"""
모델 로딩 및 추론 함수 모음

- 질문 생성 / 꼬리질문 생성 : Qwen3-8B (Unsloth LoRA)  → stage4_tail_question/models/best
- 답변 분석               : ModernBERT Multi-task     → stage3_multitask/models/best
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from unsloth import FastLanguageModel

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
