"""
모델 로딩 및 추론 함수 모음 - CPU 버전

- 질문 생성 / 꼬리질문 생성 : stage4_tail_question/models/best_cpu
  없으면 skt/kogpt2-base-v2 사용
- 답변 분석 : stage3_multitask/models/best_cpu
  없으면 None 반환 후 CLI에서 대체값 사용

주의:
- Unsloth, bitsandbytes, CUDA 사용 안 함
- .to("cuda") 없음
"""

import json
import re
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# ── 경로 ─────────────────────────────────────────────────────────────
_ROOT             = Path(__file__).parent.parent
STAGE14_BASE      = "skt/kogpt2-base-v2"
STAGE14_TUNED_CPU = _ROOT / "stage4_tail_question" / "models" / "best_cpu"
STAGE3_TUNED_CPU  = _ROOT / "stage3_multitask" / "models" / "best_cpu"
STAGE3_TUNED_OLD  = _ROOT / "stage3_multitask" / "models" / "best"

MAX_SEQ_LEN_GEN = 256
MAX_NEW_TOKENS  = 60
JOB_LABELS      = ["backend", "ai_ml", "product"]
DEVICE          = "cpu"

SYSTEM_Q_GEN = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)
SYSTEM_TAIL = (
    "당신은 IT/AI 직무 자기소개서 전문 코치입니다. "
    "질문, 답변, 의도 반영도, 적합 직무를 바탕으로 "
    "사용자가 더 구체적인 경험을 작성할 수 있도록 꼬리질문 하나를 생성하세요. "
    "Yes/No로 답할 수 없어야 하며, 반드시 한국어로 작성하세요."
)


# ── Stage 3 모델 구조 ─────────────────────────────────────────────────
class _MultiTaskModel(nn.Module):
    """intent_score 회귀 + suitable_job 분류"""

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


# ── Stage 3 래퍼 ─────────────────────────────────────────────────────
class AnalysisModel:
    """답변 분석 모델 래퍼"""

    def __init__(self, model: _MultiTaskModel, tokenizer, cfg: dict, device: str = DEVICE):
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


# ── 로더 ─────────────────────────────────────────────────────────────
def _fix_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_generation_model(use_finetuned: bool = True):
    """CPU용 생성 모델 로드."""
    if use_finetuned and STAGE14_TUNED_CPU.exists():
        model_path = str(STAGE14_TUNED_CPU)
        label = f"Fine-tuned CPU ({STAGE14_TUNED_CPU})"
    else:
        model_path = STAGE14_BASE
        label = f"Base CPU ({STAGE14_BASE})"

    print(f"  [생성 모델] {label}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = _fix_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def load_analysis_model(use_finetuned: bool = True) -> AnalysisModel | None:
    """CPU용 Stage3 분석 모델 로드. 파일이 없으면 None."""
    if not use_finetuned:
        print("  [분석 모델] 미사용")
        return None

    model_dir = STAGE3_TUNED_CPU if STAGE3_TUNED_CPU.exists() else STAGE3_TUNED_OLD
    cfg_path = model_dir / "train_config.json"
    pt_path = model_dir / "best_model.pt"

    if not cfg_path.exists() or not pt_path.exists():
        print("  [분석 모델] 파일 없음 → 건너뜀")
        print(f"             찾은 위치: {model_dir}")
        return None

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer = _fix_tokenizer(tokenizer)
    model = _MultiTaskModel(cfg["model_name"], cfg.get("num_jobs", len(cfg.get("job_classes", JOB_LABELS))))
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    print(f"  [분석 모델] {cfg['model_name']} ({model_dir})")
    return AnalysisModel(model, tokenizer, cfg, DEVICE)


# ── 추론 함수 ─────────────────────────────────────────────────────────
def generate_question(model, tokenizer, competency: str) -> str:
    prompt = (
        f"{SYSTEM_Q_GEN}\n"
        f"[TASK:generate_question] competency: {competency}\n"
        f"답변:"
    )
    text = _generate(model, tokenizer, prompt, max_new_tokens=50)
    text = _clean_generated(text)
    if not text:
        text = f"{competency} 역량을 발휘해 문제를 해결했던 경험을 구체적으로 작성해 주세요."
    return text


def generate_tail_question(
    model, tokenizer,
    question: str, answer: str,
    intent_score: float, job: str,
) -> str:
    prompt = (
        f"{SYSTEM_TAIL}\n"
        f"[TASK:tail_question]\n"
        f"question: {question}\n"
        f"answer: {answer}\n"
        f"intent_score: {intent_score:.0f}\n"
        f"job: {job}\n"
        f"답변:"
    )
    text = _generate(model, tokenizer, prompt, max_new_tokens=70)
    text = _clean_generated(text)
    if not text:
        text = "그 경험에서 본인이 직접 맡은 역할과 판단 과정을 더 구체적으로 설명해 주실 수 있나요?"
    return text


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN_GEN,
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
    """모델 출력 후처리: 첫 줄/첫 질문만 남김."""
    text = text.replace("<|endoftext|>", "").strip()
    text = re.split(r"\n|답변:|질문:", text)[0].strip()
    text = text.strip(" -:：")
    # 너무 길게 이어지면 첫 문장 수준으로 자름
    m = re.search(r"(.+?[?？])", text)
    if m:
        return m.group(1).strip()
    return text[:120].strip()
