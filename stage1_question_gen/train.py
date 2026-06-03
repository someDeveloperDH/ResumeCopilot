"""
Stage 1 - 질문 생성 모델 학습 (Unsloth LoRA)

실행: python train.py
설정: config.yaml 에서 모델/파라미터 수정

- data/stage1/train.csv 를 train/validation으로 분리하여 학습
- 에폭마다 validation loss 기록, 최고 모델을 best_lora.pt로 저장
- evaluate.py와 독립적으로 실행 가능
"""

import os
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# ── 경로 설정 (스크립트 위치 기준 → 어디서 실행해도 동일) ──────────────
_HERE = Path(__file__).parent

def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = _load_cfg()

BASE_MODEL  = _cfg["model"]["base_model"]
MAX_SEQ_LEN = _cfg["model"]["max_seq_length"]
LORA_R      = _cfg["lora"]["r"]
LORA_ALPHA  = _cfg["lora"]["alpha"]
BATCH_SIZE  = _cfg["train"]["batch_size"]
GRAD_ACCUM  = _cfg["train"]["grad_accumulation"]
EPOCHS      = _cfg["train"]["epochs"]
LR          = _cfg["train"]["learning_rate"]
VAL_RATIO   = _cfg["train"]["val_ratio"]
SEED        = _cfg["train"]["seed"]
DATA_PATH   = str(_HERE / _cfg["data"]["train_csv"])
OUTPUT_DIR  = str(_HERE / _cfg["output"]["dir"])

SYSTEM_PROMPT = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량(competency)에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)

LORA_TARGETS = _cfg["lora"]["targets"] or [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── 데이터 포맷 ───────────────────────────────────────────────────────
def to_chat(row: dict) -> dict:
    """Llama-3 chat 형식 + task prefix로 변환"""
    return {
        "text": (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"[TASK:generate_question] competency: {row['competency']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{row['question']}<|eot_id|>"
        )
    }


def load_datasets(path: str):
    df = pd.read_csv(path)

    # competency 분포 출력
    print("\n[데이터 분포]")
    print(df["competency"].value_counts().to_string())
    print(f"\n전체: {len(df)}개\n")

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        stratify=df["competency"],
        random_state=SEED,
    )
    print(f"Train: {len(train_df)}개 | Val: {len(val_df)}개")

    train_ds = Dataset.from_list([to_chat(r) for _, r in train_df.iterrows()])
    val_ds   = Dataset.from_list([to_chat(r) for _, r in val_df.iterrows()])
    return train_ds, val_ds


def load_model_and_lora():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model, tokenizer


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 데이터 ─────────────────────────────────────────────────────────
    train_ds, val_ds = load_datasets(DATA_PATH)

    # ── 모델 ───────────────────────────────────────────────────────────
    print("\n모델 로딩 중 (4bit 양자화)...")
    model, tokenizer = load_model_and_lora()

    # ── 학습 ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=SEED,
        ),
    )

    # 어시스턴트 답변 부분에만 loss 적용 (TRL 1.x 방식)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print("\n학습 시작...")
    trainer.train()

    # ── Best 모델 저장 ──────────────────────────────────────────────────
    # 1) HuggingFace adapter 형식 (config + weights, 복구용)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 2) LoRA 가중치만 .pt로 저장 (evaluate.py에서 직접 로드)
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    pt_path = os.path.join(OUTPUT_DIR, "best_lora.pt")
    torch.save(lora_state, pt_path)

    # 3) 학습 설정 저장 (evaluate.py에서 동일 구조 재현에 사용)
    config = {
        "base_model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGETS,
        "max_seq_length": MAX_SEQ_LEN,
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_DIR}")
    print(f"  adapter : {OUTPUT_DIR}/adapter_model.safetensors")
    print(f"  LoRA .pt: {pt_path}")
    print(f"  config  : {OUTPUT_DIR}/train_config.json")


if __name__ == "__main__":
    main()
