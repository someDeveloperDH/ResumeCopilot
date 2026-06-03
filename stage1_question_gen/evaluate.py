"""
Stage 1 - 질문 생성 평가

실행:
    python evaluate.py                  # fine-tuned 모델 평가 (best_lora.pt)
    python evaluate.py --baseline       # base 모델 평가 (파인튜닝 없이)

- BLEU-4, ROUGE-L, Cosine 유사도 + 통합 점수(가중 평균) 계산
- --baseline 사용 시 best_lora.pt 없이도 실행 가능 (베이스라인 수치 확인용)
"""

import os
import json
import yaml
import argparse
import warnings
import torch
import pandas as pd

# max_new_tokens / max_length 동시 설정 경고 억제 (max_new_tokens 우선 적용됨)
warnings.filterwarnings("ignore", message="Both.*max_new_tokens.*max_length")
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import SentenceTransformer, util
from unsloth import FastLanguageModel

# ── 경로 설정 (스크립트 위치 기준 → 어디서 실행해도 동일) ──────────────
_HERE = Path(__file__).parent

def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = _load_cfg()

_OUT            = _HERE / _cfg["output"]["dir"]
PT_PATH         = str(_OUT / "best_lora.pt")
CONFIG_PATH     = str(_OUT / "train_config.json")
TEST_DATA       = str(_HERE / _cfg["data"]["test_csv"])
RESULT_PATH     = str(_OUT / "eval_results.json")
VALID_THRESHOLD = _cfg["eval"]["valid_threshold"]

SYSTEM_PROMPT = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량(competency)에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)

# ── 질문 생성 ─────────────────────────────────────────────────────────
def generate_question(model, tokenizer, competency: str) -> str:
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"[TASK:generate_question] competency: {competency}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── 지표 계산 ─────────────────────────────────────────────────────────
def calc_bleu4(references: list[str], hypotheses: list[str]) -> float:
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def calc_rouge_l(references: list[str], hypotheses: list[str]) -> float:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return sum(scores) / len(scores)


def calc_cosine(references: list[str], hypotheses: list[str]) -> tuple[float, list[float]]:
    """평균 코사인 유사도 + 개별 유사도 리스트 반환"""
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    ref_emb = embedder.encode(references, convert_to_tensor=True)
    hyp_emb = embedder.encode(hypotheses, convert_to_tensor=True)
    sims = util.cos_sim(ref_emb, hyp_emb).diagonal()
    return sims.mean().item(), sims.tolist()


# ── 기준값 통과 여부 표시 ─────────────────────────────────────────────
def pass_mark(value: float, threshold: float, higher_is_better: bool = True) -> str:
    ok = value >= threshold if higher_is_better else value <= threshold
    return "✓ PASS" if ok else "✗ FAIL"


# ── 모델 로드 ────────────────────────────────────────────────────────
def load_model(baseline: bool):
    """
    baseline=True  → base 모델만 로드 (LoRA 미적용)
    baseline=False → base 모델 + best_lora.pt 로드
    """
    if baseline:
        # config.yaml의 base_model 직접 사용
        base_model  = _cfg["model"]["base_model"]
        max_seq_len = _cfg["model"]["max_seq_length"]
        print(f"[베이스라인] {base_model} 로딩 중... (LoRA 미적용)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        # train_config.json에서 학습 시 사용한 설정 복원
        with open(CONFIG_PATH, encoding="utf-8") as f:
            train_cfg = json.load(f)
        print(f"[Fine-tuned] {train_cfg['base_model']} + {PT_PATH} 로딩 중...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=train_cfg["base_model"],
            max_seq_length=train_cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=train_cfg["lora_r"],
            target_modules=train_cfg["lora_targets"],
            lora_alpha=train_cfg["lora_alpha"],
            lora_dropout=0.0,
            bias="none",
        )
        lora_state = torch.load(PT_PATH, map_location="cuda")
        model.load_state_dict(lora_state, strict=False)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="파인튜닝 없이 base 모델로 평가 (베이스라인 수치 확인)")
    args = parser.parse_args()

    mode = "BASELINE (파인튜닝 없음)" if args.baseline else "FINE-TUNED"
    print(f"\n평가 모드: {mode}\n")

    model, tokenizer = load_model(args.baseline)

    # 테스트 데이터 로드
    df = pd.read_csv(TEST_DATA)
    # 빈 컬럼이 float64(NaN)로 추론되는 경우 str로 변환
    df["generated_question"] = df["generated_question"].astype(str).replace("nan", "")
    references, hypotheses = [], []

    print(f"\n질문 생성 중 ({len(df)}개)...\n" + "=" * 55)
    for i, row in df.iterrows():
        gen = generate_question(model, tokenizer, row["competency"])
        df.at[i, "generated_question"] = gen
        references.append(row["question"])
        hypotheses.append(gen)
        print(f"[{row['question_id']}] competency: {row['competency']}")
        print(f"  정답:  {row['question']}")
        print(f"  생성:  {gen}\n")

    # 지표 계산
    bleu              = calc_bleu4(references, hypotheses)
    rouge             = calc_rouge_l(references, hypotheses)
    cosine, cos_list  = calc_cosine(references, hypotheses)

    # is_valid 판정 (샘플별 코사인 유사도 기준)
    df["is_valid"] = [sim >= VALID_THRESHOLD for sim in cos_list]

    # 통합 점수 (가중 평균)
    w      = _cfg["eval"]["weights"]
    composite = w["bleu4"] * bleu + w["rouge_l"] * rouge + w["cosine"] * cosine
    comp_thr  = _cfg["eval"]["composite_threshold"]

    # 결과 출력
    ev = _cfg["eval"]
    print("=" * 55)
    print(f"  BLEU-4        : {bleu:.4f}   (기준 > {ev['bleu4_goal']})  {pass_mark(bleu, ev['bleu4_goal'])}")
    print(f"  ROUGE-L       : {rouge:.4f}   (기준 > {ev['rouge_l_goal']})  {pass_mark(rouge, ev['rouge_l_goal'])}")
    print(f"  Cosine 유사도 : {cosine:.4f}   (기준 > {ev['cosine_goal']})  {pass_mark(cosine, ev['cosine_goal'])}")
    print(f"  is_valid 비율 : {df['is_valid'].mean():.2%}  (threshold={VALID_THRESHOLD})")
    print(f"  {'─'*47}")
    print(f"  통합 점수     : {composite:.4f}   (기준 > {comp_thr})  {pass_mark(composite, comp_thr)}")
    print(f"  가중치        : BLEU {w['bleu4']} / ROUGE {w['rouge_l']} / Cosine {w['cosine']}")
    print("=" * 55)

    # 결과 저장 (baseline이면 별도 파일로 저장)
    out_path = (
        str(Path(RESULT_PATH).parent / "eval_results_baseline.json")
        if args.baseline else RESULT_PATH
    )
    results = {
        "mode":        "baseline" if args.baseline else "finetuned",
        "bleu4":       bleu,
        "rouge_l":     rouge,
        "cosine":      cosine,
        "composite":   composite,
        "weights":     w,
        "valid_ratio": df["is_valid"].mean(),
        "n_samples":   len(df),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df.to_csv(TEST_DATA, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {out_path}")
    print(f"test.csv 갱신 완료: {TEST_DATA}")


if __name__ == "__main__":
    main()
