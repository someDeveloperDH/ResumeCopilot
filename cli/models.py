"""
모델 로딩 및 추론 함수 모음

- 질문 생성 / 꼬리질문 생성 : Qwen3-8B (Unsloth LoRA)  → stage4_tail_question/models/best
- 답변 분석               : ModernBERT Multi-task     → stage3_multitask/models/best
"""

import json
import urllib.error
import urllib.request
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

try:
    from unsloth import FastLanguageModel
except ImportError:  # Mac 로컬 CLI는 Ollama provider를 사용하므로 Unsloth가 없어도 실행 가능해야 함.
    FastLanguageModel = None

# ── 경로 ─────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).parent.parent
STAGE14_BASE   = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
STAGE14_TUNED  = _ROOT / "stage4_tail_question/models/best"
STAGE3_TUNED   = _ROOT / "stage3_multitask/models/best"

MAX_SEQ_LEN_GEN  = 768
MAX_SEQ_LEN_ANA  = 512
JOB_LABELS       = ["backend", "ai_ml", "product"]

SYSTEM_Q_GEN = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량(competency)에 맞는 자소서 질문을 한 개 생성하세요."
)
SYSTEM_TAIL = (
    "당신은 IT/AI 직무 자기소개서 전문 코치입니다. "
    "질문, 답변, 의도 반영도(intent_score), 적합 직무(job)를 바탕으로 "
    "사용자가 더 구체적인 경험을 작성할 수 있도록 꼬리질문 하나를 생성하세요. "
    "Yes/No로 답할 수 없어야 하며, 구체적 경험을 끌어내야 합니다."
)


# ── Stage 3 모델 구조 ─────────────────────────────────────────────────
class _MultiTaskModel(nn.Module):
    """ModernBERT 기반 intent_score 회귀 + suitable_job 분류"""

    def __init__(self, model_name: str, num_jobs: int = 3):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.reg_head = nn.Linear(h, 1)
        self.cls_head = nn.Linear(h, num_jobs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.reg_head(cls).squeeze(-1), self.cls_head(cls)


# ── Stage 3 래퍼 ─────────────────────────────────────────────────────
class AnalysisModel:
    """답변 분석 모델 래퍼 - intent_score 회귀 + suitable_job 분류"""

    def __init__(self, model: _MultiTaskModel, tokenizer, cfg: dict, device: str):
        self._model     = model
        self._tokenizer = tokenizer
        self._cfg       = cfg
        self._device    = device

    def predict(self, question: str, answer: str) -> tuple[float, str]:
        enc = self._tokenizer(
            question, answer,
            max_length=self._cfg["max_seq_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc.input_ids.to(self._device)
        attention_mask = enc.attention_mask.to(self._device)

        with torch.no_grad():
            reg_out, cls_out = self._model(input_ids, attention_mask)

        score = float(reg_out.item()) * 100.0
        score = max(0.0, min(100.0, score))
        job   = JOB_LABELS[cls_out.argmax(-1).item()]
        return score, job


# ── 로더 ─────────────────────────────────────────────────────────────
def load_generation_model(use_finetuned: bool = False):
    """Qwen3-8B 로드. use_finetuned=True 시 stage4 fine-tuned 모델 사용."""
    if FastLanguageModel is None:
        print("  [생성 모델] Unsloth 미설치 → CLI 로컬 Ollama provider 사용")
        return None, None

    model_path = str(STAGE14_TUNED) if use_finetuned else STAGE14_BASE
    label      = "Fine-tuned (Stage1+4)" if use_finetuned else "Base Qwen3-8B"
    print(f"  [생성 모델] {label}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN_GEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_analysis_model(use_finetuned: bool = False) -> AnalysisModel | None:
    """ModernBERT Multi-task 로드. 모델 파일 없으면 None 반환."""
    if not use_finetuned:
        print("  [분석 모델] 미사용 (stage3 학습 완료 후 교체)")
        return None

    cfg_path = STAGE3_TUNED / "train_config.json"
    pt_path  = STAGE3_TUNED / "best_model.pt"

    if not cfg_path.exists() or not pt_path.exists():
        print("  [분석 모델] 파일 없음 → 건너뜀")
        return None

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model     = _MultiTaskModel(cfg["model_name"], cfg["num_jobs"])
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval().to(device)

    print(f"  [분석 모델] {cfg['model_name']}")
    return AnalysisModel(model, tokenizer, cfg, device)


# ── 추론 함수 ─────────────────────────────────────────────────────────
def generate_question(model, tokenizer, competency: str) -> str:
    if model is None or tokenizer is None:
        samples = {
            "문제해결": "기술적 문제를 발견하고 해결한 경험을 구체적으로 설명하세요.",
            "협업": "팀 프로젝트에서 의견 차이를 조율하며 성과를 만든 경험을 설명하세요.",
            "성장": "처음 접한 기술이나 환경을 학습해 실제 결과로 연결한 경험을 설명하세요.",
            "실패경험": "실패나 실수를 겪은 뒤 재발을 막기 위해 개선한 경험을 설명하세요.",
            "주도성": "스스로 문제를 발견하고 먼저 제안하거나 실행한 경험을 설명하세요.",
        }
        return samples.get(competency, "IT/AI 직무 역량을 보여줄 수 있는 경험을 설명하세요.")

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_Q_GEN}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"[TASK:generate_question] competency: {competency}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return _generate(model, tokenizer, prompt)


def generate_tail_question(
    model, tokenizer,
    question: str, answer: str,
    intent_score: float, job: str,
) -> str:
    if model is None or tokenizer is None:
        return (
            "이 경험에서 사용한 구체적인 기술이나 결과 수치를 하나만 더 설명해 주세요. "
            "정확한 기술이 없다면 비슷한 도구, 자동화 방식, 협업 방식도 좋습니다."
        )

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_TAIL}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"[TASK:tail_question] "
        f"question: {question} "
        f"answer: {answer} "
        f"intent_score: {intent_score:.0f} "
        f"job: {job}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return _generate(model, tokenizer, prompt)


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
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
    # Llama-3 special token은 skip_special_tokens=True 로 정상 제거됨
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── Ollama 로컬 LLM provider ─────────────────────────────────────────
def ollama_chat(prompt: str, runtime_cfg: dict, *, json_format: bool = False) -> str:
    """Ollama generate API를 통해 로컬 LLM을 호출한다.

    exaone3.5:7.8b는 현재 Ollama에서 completion capability로 노출되므로
    /api/chat 대신 /api/generate를 사용한다.
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
