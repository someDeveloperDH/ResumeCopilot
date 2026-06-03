"""
Stage 3 - Multi-task 분석 모델 학습

실행: python train.py
설정: config.yaml 에서 모델/파라미터 수정

- data/stage3/train.csv 를 train/validation으로 분리
- Head 1 (Regression):     intent_score 예측 (0~100)
- Head 2 (Classification): suitable_job 예측 (backend / ai_ml / product)
- 에폭마다 val_loss 기록, 최고 모델을 best_model.pt로 저장
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── 경로 설정 (스크립트 위치 기준 → 어디서 실행해도 동일) ──────────────
_HERE = Path(__file__).parent

def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = _load_cfg()

MODEL_NAME  = _cfg["model"]["name"]
MAX_SEQ_LEN = _cfg["model"]["max_seq_length"]
BATCH_SIZE  = _cfg["train"]["batch_size"]
EPOCHS      = _cfg["train"]["epochs"]
LR          = _cfg["train"]["learning_rate"]
VAL_RATIO   = _cfg["train"]["val_ratio"]
SEED        = _cfg["train"]["seed"]
DATA_PATH   = str(_HERE / _cfg["data"]["train_csv"])
OUTPUT_DIR  = str(_HERE / _cfg["output"]["dir"])

JOB_CLASSES = _cfg["jobs"]["classes"]
NUM_JOBS    = len(JOB_CLASSES)
JOB2ID      = {j: i for i, j in enumerate(JOB_CLASSES)}


# ── 데이터셋 ─────────────────────────────────────────────────────────
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
            str(row["question"]), str(row["answer"]),
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            # intent_score: 0~1 정규화 (MSE 스케일 맞춤)
            "intent_labels":  torch.tensor(row["intent_score"] / 100.0, dtype=torch.float),
            "job_labels":     torch.tensor(JOB2ID[row["suitable_job"]], dtype=torch.long),
        }


# ── 모델 ─────────────────────────────────────────────────────────────
class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_jobs: int = NUM_JOBS):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.reg_head = nn.Linear(h, 1)
        self.cls_head = nn.Linear(h, num_jobs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: torch.Tensor = None,
        job_labels: torch.Tensor    = None,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]           # [CLS] 토큰

        reg_out = self.reg_head(cls).squeeze(-1)    # (B,)  0~1 스케일
        cls_out = self.cls_head(cls)                # (B, num_jobs)

        loss = None
        if intent_labels is not None:
            mse_loss = F.mse_loss(reg_out, intent_labels)
            ce_loss  = F.cross_entropy(cls_out, job_labels)
            loss     = mse_loss + ce_loss

        return loss, reg_out, cls_out


# ── 학습 / 검증 루프 ─────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _, _ = model(**batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드 & 분리
    df = pd.read_csv(DATA_PATH)
    print(f"\n[데이터 분포]")
    print(df["suitable_job"].value_counts().to_string())
    print(f"intent_score 분포: mean={df['intent_score'].mean():.1f}  std={df['intent_score'].std():.1f}\n")

    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, stratify=df["suitable_job"], random_state=SEED
    )
    print(f"Train: {len(train_df)}개 | Val: {len(val_df)}개")

    # 토크나이저 & 데이터로더
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = MultiTaskDataset(train_df, tokenizer, MAX_SEQ_LEN)
    val_ds    = MultiTaskDataset(val_df,   tokenizer, MAX_SEQ_LEN)
    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 모델 & 옵티마이저
    model     = MultiTaskModel(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 학습
    best_val_loss = float("inf")
    history = []

    print("\n학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_dl, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_dl,   optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print("  ← 저장")
        else:
            print()

    # 설정 저장 (evaluate.py에서 동일 구조 재현)
    config = {
        "model_name":   MODEL_NAME,
        "num_jobs":     NUM_JOBS,
        "job_classes":  JOB_CLASSES,
        "max_seq_length": MAX_SEQ_LEN,
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_DIR}")
    print(f"  best_model.pt  (val_loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
