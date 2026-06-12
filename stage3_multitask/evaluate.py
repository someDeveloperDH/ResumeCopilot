"""
Stage 3 - Multi-task 분석 모델 평가 (CPU 소형 모델 버전)

실행:
    python evaluate.py
    python evaluate.py --baseline
"""

import argparse
import json
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

_HERE = Path(__file__).parent

def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = _load_cfg()

_OUT = _HERE / _cfg["output"]["dir"]
PT_PATH = str(_OUT / "best_model.pt")
CONFIG_PATH = str(_OUT / "train_config.json")
TEST_DATA = str(_HERE / _cfg["data"]["test_csv"])
RESULT_PATH = str(_OUT / "eval_results.json")
BATCH_SIZE = int(_cfg["train"]["batch_size"])
_EVAL = _cfg["eval"]

class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_jobs: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.reg_head = nn.Linear(h, 1)
        self.cls_head = nn.Linear(h, num_jobs)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.reg_head(cls).squeeze(-1), self.cls_head(cls)

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

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
        }

def pass_mark(value, threshold, higher_is_better=True) -> str:
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return "✓ PASS" if ok else "✗ FAIL"

def regression_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse

def classification_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return acc, f1, cm

def load_model(baseline: bool, device):
    if baseline:
        model_name = _cfg["model"]["name"]
        max_seq_len = int(_cfg["model"]["max_seq_length"])
        job_classes = _cfg["jobs"]["classes"]
        num_jobs = len(job_classes)
        print(f"[베이스라인] {model_name} (랜덤 헤드, 파인튜닝 없음)")
    else:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        model_name = saved["model_name"]
        max_seq_len = int(saved["max_seq_length"])
        job_classes = saved["job_classes"]
        num_jobs = int(saved["num_jobs"])
        print(f"[Fine-tuned] {model_name} + {PT_PATH}")

    model = MultiTaskModel(model_name, num_jobs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not baseline:
        model.load_state_dict(torch.load(PT_PATH, map_location=device))

    model.eval()
    return model, tokenizer, job_classes, max_seq_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "BASELINE (파인튜닝 없음)" if args.baseline else "FINE-TUNED"
    print(f"\n평가 모드: {mode}")
    print(f"사용 장치: {device}\n")

    model, tokenizer, job_classes, max_seq_len = load_model(args.baseline, device)
    job2id = {j: i for i, j in enumerate(job_classes)}

    df = pd.read_csv(TEST_DATA)
    required = {"question", "answer", "intent_score", "suitable_job"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df["generated_intent_score"] = df.get("generated_intent_score", pd.Series(dtype="float64"))
    df["generated_suitable_job"] = df.get("generated_suitable_job", pd.Series(dtype="str")).astype(str).replace("nan", "")

    ds = TestDataset(df, tokenizer, max_seq_len)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_reg, all_cls = [], []
    print(f"추론 중 ({len(df)}개)...")

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            reg_out, cls_out = model(input_ids, attention_mask)
            all_reg.extend((reg_out.cpu().numpy() * 100.0).tolist())
            all_cls.extend(cls_out.argmax(-1).cpu().numpy().tolist())

    df["generated_intent_score"] = [round(float(s), 1) for s in all_reg]
    df["generated_suitable_job"] = [job_classes[i] for i in all_cls]

    y_reg_true = df["intent_score"].values.astype(float)
    y_reg_pred = np.array(all_reg)
    mae, rmse = regression_metrics(y_reg_true, y_reg_pred)

    unknown = sorted(set(df["suitable_job"].astype(str)) - set(job_classes))
    if unknown:
        raise ValueError(f"config/train_config에 없는 suitable_job 값: {unknown}")

    y_cls_true = [job2id[str(j)] for j in df["suitable_job"]]
    y_cls_pred = all_cls
    acc, f1, cm = classification_metrics(y_cls_true, y_cls_pred, job_classes)

    print("\n" + "=" * 55)
    print("  [Regression] intent_score")
    print(f"  MAE      : {mae:.2f}   (기준 ≤ {_EVAL['mae_threshold']})  {pass_mark(mae, _EVAL['mae_threshold'], False)}")
    print(f"  RMSE     : {rmse:.2f}   (기준 ≤ {_EVAL['rmse_threshold']})  {pass_mark(rmse, _EVAL['rmse_threshold'], False)}")
    print()
    print(f"  [Classification] suitable_job ({len(job_classes)}-class 분류, 랜덤 기준 {100 / len(job_classes):.1f}%)")
    print(f"  Accuracy  : {acc:.4f}   (기준 > {_EVAL['accuracy_threshold']}) {pass_mark(acc, _EVAL['accuracy_threshold'])}")
    print(f"  F1 (macro): {f1:.4f}   (기준 > {_EVAL['f1_threshold']}) {pass_mark(f1, _EVAL['f1_threshold'])}")
    print()
    print("  Confusion Matrix")
    print(f"  {'':>10}", "  ".join(f"{j:>8}" for j in job_classes))
    for i, row_cm in enumerate(cm):
        print(f"  {job_classes[i]:>10}", "  ".join(f"{v:>8}" for v in row_cm))
    print("=" * 55)

    out_path = (
        str(Path(RESULT_PATH).parent / "eval_results_baseline.json")
        if args.baseline else RESULT_PATH
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results = {
        "mode": "baseline" if args.baseline else "finetuned",
        "regression": {"mae": float(mae), "rmse": float(rmse)},
        "classification": {"accuracy": float(acc), "f1_macro": float(f1)},
        "n_samples": int(len(df)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df.to_csv(TEST_DATA, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {out_path}")
    print(f"test.csv 갱신 완료: {TEST_DATA}")

if __name__ == "__main__":
    main()
