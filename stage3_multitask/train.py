"""
Stage 3 - Multi-task 분석 모델 학습 (CPU 소형 모델 버전)

실행:
    python train.py

CPU 변경점:
- config.yaml의 model.name을 monologg/koelectra-small-v3-discriminator로 변경
- CUDA가 없어도 CPU에서 실행
- batch_size / max_seq_length를 CPU용으로 축소
"""

import os
import json
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent

def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = _load_cfg()

MODEL_NAME  = _cfg["model"]["name"]
MAX_SEQ_LEN = int(_cfg["model"]["max_seq_length"])
BATCH_SIZE  = int(_cfg["train"]["batch_size"])
EPOCHS      = int(_cfg["train"]["epochs"])
LR          = float(_cfg["train"]["learning_rate"])
VAL_RATIO   = float(_cfg["train"]["val_ratio"])
SEED        = int(_cfg["train"]["seed"])
FREEZE_ENCODER = bool(_cfg["train"].get("freeze_encoder", False))
NUM_WORKERS = int(_cfg["train"].get("num_workers", 0))

DATA_PATH   = str(_HERE / _cfg["data"]["train_csv"])
OUTPUT_DIR  = str(_HERE / _cfg["output"]["dir"])

JOB_CLASSES = _cfg["jobs"]["classes"]
NUM_JOBS    = len(JOB_CLASSES)
JOB2ID      = {j: i for i, j in enumerate(JOB_CLASSES)}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        enc = self.tokenizer(
            str(row["question"]),
            str(row["answer"]),
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "intent_labels": torch.tensor(float(row["intent_score"]) / 100.0, dtype=torch.float),
            "job_labels": torch.tensor(JOB2ID[str(row["suitable_job"])], dtype=torch.long),
        }

class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_jobs: int = NUM_JOBS, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.reg_head = nn.Linear(h, 1)
        self.cls_head = nn.Linear(h, num_jobs)

    def forward(self, input_ids, attention_mask, intent_labels=None, job_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])

        reg_out = self.reg_head(cls).squeeze(-1)
        cls_out = self.cls_head(cls)

        loss = None
        if intent_labels is not None and job_labels is not None:
            mse_loss = F.mse_loss(reg_out, intent_labels)
            ce_loss = F.cross_entropy(cls_out, job_labels)
            loss = mse_loss + ce_loss

        return loss, reg_out, cls_out

def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _, _ = model(**batch)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n사용 장치: {device}")
    print(f"모델: {MODEL_NAME}")
    print(f"max_seq_length={MAX_SEQ_LEN}, batch_size={BATCH_SIZE}, freeze_encoder={FREEZE_ENCODER}")

    df = pd.read_csv(DATA_PATH)

    required = {"question", "answer", "intent_score", "suitable_job"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df = df.dropna(subset=["question", "answer", "intent_score", "suitable_job"]).copy()
    df["suitable_job"] = df["suitable_job"].astype(str)
    unknown = sorted(set(df["suitable_job"]) - set(JOB_CLASSES))
    if unknown:
        raise ValueError(f"config.yaml jobs.classes에 없는 suitable_job 값: {unknown}")

    print("\n[데이터 분포]")
    print(df["suitable_job"].value_counts().to_string())
    print(f"intent_score 분포: mean={df['intent_score'].mean():.1f}  std={df['intent_score'].std():.1f}\n")

    stratify = df["suitable_job"] if df["suitable_job"].value_counts().min() >= 2 else None
    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, stratify=stratify, random_state=SEED
    )
    print(f"Train: {len(train_df)}개 | Val: {len(val_df)}개")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MultiTaskDataset(train_df, tokenizer, MAX_SEQ_LEN)
    val_ds = MultiTaskDataset(val_df, tokenizer, MAX_SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MultiTaskModel(MODEL_NAME, NUM_JOBS, FREEZE_ENCODER).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_loss = float("inf")
    history = []

    print("\n학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_dl, optimizer, device, train=True)
        val_loss = run_epoch(model, val_dl, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        msg = f"  Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(msg + "  ← 저장")
        else:
            print(msg)

    saved_config = {
        "model_name": MODEL_NAME,
        "num_jobs": NUM_JOBS,
        "job_classes": JOB_CLASSES,
        "max_seq_length": MAX_SEQ_LEN,
        "freeze_encoder": FREEZE_ENCODER,
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(saved_config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_DIR}")
    print(f"  best_model.pt  (val_loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    main()
