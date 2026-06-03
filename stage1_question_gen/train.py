"""
Stage 1 - 질문 생성 모델 학습 (CPU 소형 모델 버전)

실행:
    python train.py

변경점:
- Unsloth / 4bit / CUDA 제거
- skt/kogpt2-base-v2 같은 소형 CausalLM을 Transformers Trainer로 학습
- CPU에서도 실행 가능하도록 batch_size=1, max_seq_length=256 권장
"""

import os
import json
import random
import yaml
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

_HERE = Path(__file__).parent


def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))


_cfg = _load_cfg()

BASE_MODEL = _cfg["model"]["base_model"]
MAX_SEQ_LEN = int(_cfg["model"]["max_seq_length"])
BATCH_SIZE = int(_cfg["train"]["batch_size"])
GRAD_ACCUM = int(_cfg["train"]["grad_accumulation"])
EPOCHS = float(_cfg["train"]["epochs"])
LR = float(_cfg["train"]["learning_rate"])
VAL_RATIO = float(_cfg["train"]["val_ratio"])
SEED = int(_cfg["train"]["seed"])
DATA_PATH = str(_HERE / _cfg["data"]["train_csv"])
OUTPUT_DIR = str(_HERE / _cfg["output"]["dir"])

SYSTEM_PROMPT = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(competency: str) -> str:
    return (
        f"[시스템] {SYSTEM_PROMPT}\n"
        f"[작업] 역량: {competency}\n"
        f"[질문]"
    )


def to_text(row: dict) -> dict:
    return {"text": f"{build_prompt(row['competency'])} {row['question']}"}


def load_datasets(path: str):
    df = pd.read_csv(path)
    print("\n[데이터 분포]")
    print(df["competency"].value_counts().to_string())
    print(f"\n전체: {len(df)}개\n")

    stratify = df["competency"] if df["competency"].value_counts().min() >= 2 else None
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        stratify=stratify,
        random_state=SEED,
    )
    print(f"Train: {len(train_df)}개 | Val: {len(val_df)}개")
    return Dataset.from_list([to_text(r) for _, r in train_df.iterrows()]), Dataset.from_list([to_text(r) for _, r in val_df.iterrows()])


def load_model_and_tokenizer():
    print(f"\n모델 로딩 중: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def tokenize_dataset(ds: Dataset, tokenizer):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    return ds.map(tok, batched=True, remove_columns=ds.column_names)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds = load_datasets(DATA_PATH)
    model, tokenizer = load_model_and_tokenizer()
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    device_msg = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n학습 디바이스: {device_msg}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        fp16=False,
        bf16=False,
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n학습 시작...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    train_config = {
        "base_model": BASE_MODEL,
        "max_seq_length": MAX_SEQ_LEN,
        "model_type": "cpu_transformers_causal_lm",
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
