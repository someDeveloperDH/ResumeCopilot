"""
Stage 4 CPU 버전 - Stage 1+4 통합 생성 모델 학습
실행: python train.py

- GPU/Unsloth/bitsandbytes/LoRA 사용 안 함
- 작은 KoGPT2 모델을 CPU에서 full fine-tuning
- [TASK:generate_question] 과 [TASK:tail_question] 두 형식을 함께 학습
"""
import os
import json
import random
from pathlib import Path

import yaml
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast


def load_kogpt2_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if type(tokenizer).__name__ == "GPT2Tokenizer":
        # skt/kogpt2-base-v2의 AutoTokenizer 기본 로딩은 한국어를 깨진 바이트(�)로 디코딩함.
        # PreTrainedTokenizerFast + 명시적 special token으로 로딩해야 정상 동작.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            path,
            bos_token="</s>", eos_token="</s>", unk_token="<unk>",
            pad_token="<pad>", mask_token="<mask>",
        )
    return tokenizer

_HERE = Path(__file__).parent

def load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = load_cfg()
MODEL_NAME = _cfg["model"]["base_model"]
MAX_LEN = int(_cfg["model"]["max_seq_length"])
BATCH_SIZE = int(_cfg["train"]["batch_size"])
EPOCHS = int(_cfg["train"]["epochs"])
LR = float(_cfg["train"]["learning_rate"])
VAL_RATIO = float(_cfg["train"]["val_ratio"])
SEED = int(_cfg["train"]["seed"])
MAX_SAMPLES = int(_cfg["train"].get("max_samples", 0) or 0)
LOG_EVERY = int(_cfg["train"].get("log_every", 10))
STAGE1_DATA = _HERE / _cfg["data"]["stage1_csv"]
STAGE4_DATA = _HERE / _cfg["data"]["stage4_csv"]
OUTPUT_DIR = _HERE / _cfg["output"]["dir"]

SYSTEM_Q_GEN = "IT/AI 직무 자기소개서 질문 생성. 주어진 역량에 맞는 자소서 질문을 한국어로 하나 생성."
SYSTEM_TAIL = "IT/AI 직무 자기소개서 코치. 질문, 답변, 의도점수, 직무를 보고 구체적 경험을 끌어내는 한국어 꼬리질문 하나 생성."

def format_stage1(row, eos_token: str) -> str:
    return (
        f"시스템: {SYSTEM_Q_GEN}\n"
        f"사용자: [TASK:generate_question] competency: {row['competency']}\n"
        f"어시스턴트: {row['question']}{eos_token}"
    )

def format_stage4(row, eos_token: str) -> str:
    return (
        f"시스템: {SYSTEM_TAIL}\n"
        f"사용자: [TASK:tail_question] question: {row['question']} answer: {row['answer']} "
        f"intent_score: {row['intent_score']} job: {row['suitable_job']}\n"
        f"어시스턴트: {row['tail_question']}{eos_token}"
    )

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_samples(eos_token: str):
    samples = []
    if STAGE1_DATA.exists():
        df1 = pd.read_csv(STAGE1_DATA)
        samples += [format_stage1(r, eos_token) for _, r in df1.iterrows()]
        print(f"Stage1 샘플: {len(df1)}개")
    else:
        print(f"[경고] Stage1 데이터 없음: {STAGE1_DATA}")

    if STAGE4_DATA.exists():
        df4 = pd.read_csv(STAGE4_DATA)
        samples += [format_stage4(r, eos_token) for _, r in df4.iterrows()]
        print(f"Stage4 샘플: {len(df4)}개")
    else:
        raise FileNotFoundError(f"Stage4 데이터가 없습니다: {STAGE4_DATA}")

    if not samples:
        raise ValueError("학습 샘플이 없습니다.")

    random.seed(SEED)
    random.shuffle(samples)
    if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]
        print(f"[빠른 CPU 테스트] 전체 샘플을 {MAX_SAMPLES}개로 제한")
    n_val = max(1, int(len(samples) * VAL_RATIO))
    return samples[n_val:], samples[:n_val]

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            if train and (step % LOG_EVERY == 0 or step == len(loader)):
                print(f"    batch {step}/{len(loader)} loss={loss.item():.4f}", flush=True)
    return total / max(1, len(loader))

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"모델 로딩: {MODEL_NAME} ({device})")
    tokenizer = load_kogpt2_tokenizer(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts, val_texts = load_samples(tokenizer.eos_token)
    print(f"Train: {len(train_texts)}개 | Val: {len(val_texts)}개")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # tokenizer와 model의 vocab 크기가 다르면 embedding index out of range 발생
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    train_loader = DataLoader(TextDataset(train_texts, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_texts, tokenizer), batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    history = []
    print("\n학습 시작 (CPU)...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", end="")
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(" ← 저장")
        else:
            print()

    train_config = {
        "base_model": MODEL_NAME,
        "max_seq_length": MAX_LEN,
        "tasks": ["generate_question", "tail_question"],
        "cpu_version": True,
    }
    (OUTPUT_DIR / "train_config.json").write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
