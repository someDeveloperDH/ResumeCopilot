"""
모델 로딩 및 추론 함수 모음 - GPU/CPU 환경 자동 감지 통합판

- GPU(+ unsloth) 가능 시  : 생성 = Meta-Llama-3.1-8B-Instruct-bnb-4bit (Unsloth 4bit LoRA, test3/jintae_v2)
                            분석 = ModernBERT-base 멀티태스크 (test3/jintae_v2)
- GPU 불가(CPU)        시  : 생성 = skt/kogpt2-base-v2 (jinyoung_v2)
                            분석 = monologg/koelectra-small-v3-discriminator 멀티태스크 (jinyoung_v2)

분석 모델(_MultiTaskModel)은 두 환경에서 구조가 동일하므로 코드를 공유하고,
체크포인트 디렉터리(best/ vs best_cpu/)와 train_config.json의 model_name만 환경별로 다르게 선택한다.

Ollama(EXAONE 등) 로컬 LLM provider는 환경과 무관하게 STAR 분석/재작성 단계(coach.py)에서 사용한다.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None


# ── 환경 감지 ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU_MODELS = DEVICE == "cuda" and FastLanguageModel is not None

_ROOT = Path(__file__).parent.parent

# GPU 경로: test3 / jintae_v2 와 동일한 베이스/체크포인트
GPU_GEN_BASE  = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
GPU_GEN_TUNED = _ROOT / "stage4_tail_question" / "models" / "best"
GPU_ANA_TUNED = _ROOT / "stage3_multitask" / "models" / "best"
MAX_SEQ_LEN_GEN_GPU = 768

# CPU 경로: jinyoung_v2 와 동일한 베이스/체크포인트
CPU_GEN_BASE      = "skt/kogpt2-base-v2"
CPU_GEN_TUNED     = _ROOT / "stage4_tail_question" / "models" / "best_cpu"
CPU_ANA_TUNED     = _ROOT / "stage3_multitask" / "models" / "best_cpu"
CPU_ANA_TUNED_OLD = _ROOT / "stage3_multitask" / "models" / "best"
MAX_SEQ_LEN_GEN_CPU = 256
MAX_NEW_TOKENS_CPU  = 60

JOB_LABELS = ["backend", "ai_ml", "product"]

SYSTEM_Q_GEN = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량(competency)에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)
SYSTEM_TAIL = (
    "당신은 IT/AI 직무 자기소개서 전문 코치입니다. "
    "원본 질문과 지금까지의 전체 대화 맥락, 의도 반영도(intent_score), 적합 직무(job)를 모두 참고해 "
    "사용자가 더 구체적인 경험을 드러낼 수 있도록 다음 꼬리질문 한 개만 생성하세요. "
    "이미 답한 내용을 반복해서 묻지 말고, 모순되거나 부족한 부분을 구체적으로 파고드세요. "
    "Yes/No로 답할 수 없어야 하며, 반드시 한국어 질문 한 문장으로 작성하세요."
)


# ── 분석 모델 구조 (GPU/CPU 공통 - ModernBERT/KoELECTRA 둘 다 동일 인터페이스) ──
class _MultiTaskModel(nn.Module):
    """인코더 기반 intent_score 회귀 + suitable_job 분류 (ModernBERT 또는 KoELECTRA)."""

    def __init__(self, model_name: str, num_jobs: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.reg_head = nn.Linear(h, 1)
        self.cls_head = nn.Linear(h, num_jobs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.reg_head(cls).squeeze(-1), self.cls_head(cls)


class AnalysisModel:
    """답변 분석 모델 래퍼 - intent_score 회귀 + suitable_job 분류."""

    def __init__(self, model: _MultiTaskModel, tokenizer, cfg: dict, device: str):
        self._model = model
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._device = device
        self._job_labels = cfg.get("job_classes", JOB_LABELS)

    def predict(self, question: str, answer: str) -> tuple[float, str]:
        enc = self._tokenizer(
            question, answer,
            max_length=self._cfg.get("max_seq_length", 256),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(self._device)
        attention_mask = enc.attention_mask.to(self._device)

        with torch.no_grad():
            reg_out, cls_out = self._model(input_ids, attention_mask)

        score = float(reg_out.item()) * 100.0
        score = max(0.0, min(100.0, score))
        idx = int(cls_out.argmax(-1).item())
        job = self._job_labels[idx] if idx < len(self._job_labels) else JOB_LABELS[0]
        return score, job


def _fix_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ── 로더 ─────────────────────────────────────────────────────────────
def load_generation_model(use_finetuned: bool = True):
    """환경에 맞는 생성 모델을 로드한다. (GPU→Llama-3.1 / CPU→KoGPT2)"""
    if USE_GPU_MODELS:
        model_path = str(GPU_GEN_TUNED) if (use_finetuned and GPU_GEN_TUNED.exists()) else GPU_GEN_BASE
        label = f"GPU Fine-tuned ({GPU_GEN_TUNED})" if model_path != GPU_GEN_BASE else f"GPU Base ({GPU_GEN_BASE})"
        print(f"  [생성 모델] {label}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LEN_GEN_GPU,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    if use_finetuned and CPU_GEN_TUNED.exists():
        model_path = str(CPU_GEN_TUNED)
        label = f"CPU Fine-tuned ({CPU_GEN_TUNED})"
    else:
        model_path = CPU_GEN_BASE
        label = f"CPU Base ({CPU_GEN_BASE})"

    print(f"  [생성 모델] {label}")
    tokenizer = _fix_tokenizer(AutoTokenizer.from_pretrained(model_path))
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def load_analysis_model(use_finetuned: bool = True) -> AnalysisModel | None:
    """환경에 맞는 분석 모델을 로드한다. 체크포인트가 없으면 None (CLI는 폴백 사용)."""
    if not use_finetuned:
        print("  [분석 모델] 미사용")
        return None

    if USE_GPU_MODELS:
        model_dir = GPU_ANA_TUNED
    else:
        model_dir = CPU_ANA_TUNED if CPU_ANA_TUNED.exists() else CPU_ANA_TUNED_OLD

    cfg_path = model_dir / "train_config.json"
    pt_path = model_dir / "best_model.pt"

    if not cfg_path.exists() or not pt_path.exists():
        print("  [분석 모델] 파일 없음 → 건너뜀")
        print(f"             찾은 위치: {model_dir}")
        return None

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = _fix_tokenizer(AutoTokenizer.from_pretrained(cfg["model_name"]))
    model = _MultiTaskModel(cfg["model_name"], cfg.get("num_jobs", len(cfg.get("job_classes", JOB_LABELS))))
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    print(f"  [분석 모델] {cfg['model_name']} ({model_dir})")
    return AnalysisModel(model, tokenizer, cfg, DEVICE)


# ── 질문/꼬리질문 생성 ────────────────────────────────────────────────
_SAMPLE_QUESTIONS = {
    "문제해결": "기술적 문제를 발견하고 해결한 경험을 구체적으로 설명하세요.",
    "협업": "팀 프로젝트에서 의견 차이를 조율하며 성과를 만든 경험을 설명하세요.",
    "성장": "처음 접한 기술이나 환경을 학습해 실제 결과로 연결한 경험을 설명하세요.",
    "실패경험": "실패나 실수를 겪은 뒤 재발을 막기 위해 개선한 경험을 설명하세요.",
    "주도성": "스스로 문제를 발견하고 먼저 제안하거나 실행한 경험을 설명하세요.",
}


def generate_question(model, tokenizer, competency: str) -> str:
    if model is None or tokenizer is None:
        return _SAMPLE_QUESTIONS.get(competency, "IT/AI 직무 역량을 보여줄 수 있는 경험을 설명하세요.")

    if USE_GPU_MODELS:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_Q_GEN}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"[TASK:generate_question] competency: {competency}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        text = _generate_gpu(model, tokenizer, prompt)
    else:
        prompt = (
            f"{SYSTEM_Q_GEN}\n"
            f"[TASK:generate_question] competency: {competency}\n"
            f"답변:"
        )
        text = _clean_generated(_generate_cpu(model, tokenizer, prompt, max_new_tokens=50))

    if not text:
        text = f"{competency} 역량을 발휘해 문제를 해결했던 경험을 구체적으로 작성해 주세요."
    return text


_SAMPLE_TAIL = (
    "이 경험에서 사용한 구체적인 기술이나 결과 수치를 하나만 더 설명해 주세요. "
    "정확한 기술이 없다면 비슷한 도구, 자동화 방식, 협업 방식도 좋습니다."
)


def generate_tail_question(
    model, tokenizer,
    question: str, answer: str,
    intent_score: float, job: str,
    conversation_context: str = "",
    consistency_note: str = "",
) -> str:
    """전체 대화 맥락(conversation_context)과 일관성 검사 메모(consistency_note)를 반영해 꼬리질문 생성."""
    if model is None or tokenizer is None:
        return _SAMPLE_TAIL

    context_block = conversation_context.strip() or f"[최초 답변]\n{answer}"
    note_block = f"\n[일관성 검사 메모]\n{consistency_note}\n" if consistency_note else ""

    if USE_GPU_MODELS:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_TAIL}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"[TASK:tail_question] "
            f"question: {question} "
            f"intent_score: {intent_score:.0f} "
            f"job: {job}\n"
            f"[지금까지의 전체 대화]\n{context_block}\n"
            f"{note_block}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        text = _generate_gpu(model, tokenizer, prompt)
    else:
        prompt = (
            f"{SYSTEM_TAIL}\n"
            f"[TASK:tail_question]\n"
            f"원본 질문: {question}\n"
            f"현재 의도 반영도: {intent_score:.0f}\n"
            f"현재 적합 직무: {job}\n"
            f"\n[지금까지의 전체 대화]\n{context_block}\n"
            f"{note_block}"
            f"\n위 내용을 바탕으로 다음에 물어볼 꼬리질문 한 개만 작성하세요.\n"
            f"답변:"
        )
        text = _clean_generated(_generate_cpu(model, tokenizer, prompt, max_new_tokens=70))

    if not text:
        text = "지금까지의 답변을 기준으로 본인이 직접 판단하고 행동한 부분을 더 구체적으로 설명해 주실 수 있나요?"
    return text


def _generate_gpu(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def _generate_cpu(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_CPU) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN_GEN_CPU,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _clean_generated(text: str) -> str:
    """소형 CPU 모델 출력 후처리: 첫 줄/첫 질문만 남김."""
    text = text.replace("<|endoftext|>", "").strip()
    text = re.split(r"\n|답변:|질문:", text)[0].strip()
    text = text.strip(" -:：")
    m = re.search(r"(.+?[?？])", text)
    if m:
        return m.group(1).strip()
    return text[:120].strip()


# ── Ollama 로컬 LLM provider (환경 무관, STAR 분석/재작성에 사용) ─────────
def ollama_chat(prompt: str, runtime_cfg: dict, *, json_format: bool = False) -> str:
    """Ollama generate API를 통해 로컬 LLM(예: exaone3.5)을 호출한다.

    completion 기반 모델은 /api/chat 대신 /api/generate를 사용한다.
    """
    base_url = runtime_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/generate"
    payload = {
        "model": runtime_cfg.get("model", "exaone3.5:7.8b"),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": runtime_cfg.get("temperature", 0.2),
            "num_predict": runtime_cfg.get("max_tokens", 900),
        },
    }
    if json_format:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=runtime_cfg.get("timeout_seconds", 120)) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama 호출 실패: {exc}") from exc

    return body.get("response", "").strip()
