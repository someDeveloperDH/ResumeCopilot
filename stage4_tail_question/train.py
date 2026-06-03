"""
Stage 4 - 꼬리질문 생성 모델 학습 (Stage 1+4 통합 Multi-task)

실행: python train.py

PDF 설계: "Qwen3-8B 하나로 1단계 + 4단계 동시 처리 - Instruction tuning으로 두 Task 학습"
- [TASK:generate_question] : Stage 1 질문 생성  (data/stage1/train.csv)
- [TASK:tail_question]     : Stage 4 꼬리질문 생성 (data/stage4/tail_train.csv)
- 두 데이터를 합쳐 하나의 Qwen3-8B LoRA 모델 학습
- 이 모델이 stage1 모델을 대체하여 CLI에서 두 Task 모두 처리

모델 교체 후 cli/main.py:
  USE_STAGE14_FINETUNED = True
  STAGE14_TUNED = "../stage4_tail_question/models/best"
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

BASE_MODEL   = _cfg["model"]["base_model"]
MAX_SEQ_LEN  = _cfg["model"]["max_seq_length"]
LORA_R       = _cfg["lora"]["r"]
LORA_ALPHA   = _cfg["lora"]["alpha"]
LORA_TARGETS = _cfg["lora"]["targets"]
BATCH_SIZE   = _cfg["train"]["batch_size"]
GRAD_ACCUM   = _cfg["train"]["grad_accumulation"]
EPOCHS       = _cfg["train"]["epochs"]
LR           = _cfg["train"]["learning_rate"]
VAL_RATIO    = _cfg["train"]["val_ratio"]
SEED         = _cfg["train"]["seed"]
STAGE1_DATA  = str(_HERE / _cfg["data"]["stage1_csv"])
STAGE4_DATA  = str(_HERE / _cfg["data"]["stage4_csv"])
OUTPUT_DIR   = str(_HERE / _cfg["output"]["dir"])

SYSTEM_Q_GEN = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량(competency)에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)
SYSTEM_TAIL = (
    "당신은 IT/AI 직무 자기소개서 전문 코치입니다. "
    "질문, 답변, 의도 반영도(intent_score), 적합 직무(job)를 바탕으로 "
    "사용자가 더 구체적인 경험을 작성할 수 있도록 꼬리질문 하나를 생성하세요. "
    "Yes/No로 답할 수 없어야 하며, 구체적 경험을 끌어내야 합니다. "
    "반드시 한국어로만 작성하세요."
)


# ── 데이터 포맷 ───────────────────────────────────────────────────────
def format_stage1(row: dict) -> dict:
    return {
        "text": (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_Q_GEN}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"[TASK:generate_question] competency: {row['competency']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{row['question']}<|eot_id|>"
        )
    }


def format_stage4(row: dict) -> dict:
    return {
        "text": (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_TAIL}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"[TASK:tail_question] "
            f"question: {row['question']} "
            f"answer: {row['answer']} "
            f"intent_score: {row['intent_score']} "
            f"job: {row['suitable_job']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{row['tail_question']}<|eot_id|>"
        )
    }


def load_combined_dataset():
    # Stage 1 데이터 (질문 생성)
    df1 = pd.read_csv(STAGE1_DATA)
    s1_samples = [format_stage1(r) for _, r in df1.iterrows()]

    # Stage 4 데이터 (꼬리질문 생성)
    df4 = pd.read_csv(STAGE4_DATA)
    s4_samples = [format_stage4(r) for _, r in df4.iterrows()]

    all_samples = s1_samples + s4_samples
    print(f"Stage1 샘플: {len(s1_samples)}개 | Stage4 샘플: {len(s4_samples)}개")
    print(f"전체: {len(all_samples)}개")

    # 셔플 후 train/val 분리
    import random
    random.seed(SEED)
    random.shuffle(all_samples)

    n_val   = max(1, int(len(all_samples) * VAL_RATIO))
    n_train = len(all_samples) - n_val
    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:]

    print(f"Train: {len(train_samples)}개 | Val: {len(val_samples)}개")
    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds = load_combined_dataset()

    # 모델 로드
    print("\n모델 로딩 중 (4bit 양자화)...")
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

    print("\n학습 시작 (Stage 1+4 통합)...")
    trainer.train()

    # 저장
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, os.path.join(OUTPUT_DIR, "best_lora.pt"))

    config = {
        "base_model":    BASE_MODEL,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "lora_targets":  LORA_TARGETS,
        "max_seq_length": MAX_SEQ_LEN,
        "tasks":         ["generate_question", "tail_question"],
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_DIR}")
    print(f"  CLI 교체: cli/main.py → USE_STAGE14_FINETUNED = True")


if __name__ == "__main__":
    main()
