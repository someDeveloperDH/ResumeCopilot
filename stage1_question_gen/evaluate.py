"""
Stage 1 - 질문 생성 평가 (CPU 소형 모델 버전)

실행:
    python evaluate.py --baseline   # base 모델 평가
    python evaluate.py              # fine-tuned 모델 평가
"""

import os
import json
import yaml
import argparse
import warnings
import torch
import pandas as pd
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", message="Both.*max_new_tokens.*max_length")

_HERE = Path(__file__).parent


def _load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))


_cfg = _load_cfg()
_OUT = _HERE / _cfg["output"]["dir"]
CONFIG_PATH = str(_OUT / "train_config.json")
TEST_DATA = str(_HERE / _cfg["data"]["test_csv"])
RESULT_PATH = str(_OUT / "eval_results.json")
VALID_THRESHOLD = _cfg["eval"]["valid_threshold"]
MAX_NEW_TOKENS = int(_cfg["model"].get("max_new_tokens", 64))

SYSTEM_PROMPT = (
    "당신은 IT/AI 직무 자기소개서 질문 생성 전문가입니다. "
    "주어진 역량에 맞는 자소서 질문을 한 개 생성하세요. "
    "반드시 한국어로만 작성하세요."
)


def build_prompt(competency: str) -> str:
    return (
        f"[시스템] {SYSTEM_PROMPT}\n"
        f"[작업] 역량: {competency}\n"
        f"[질문]"
    )


def generate_question(model, tokenizer, competency: str, device: torch.device) -> str:
    prompt = build_prompt(competency)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def calc_bleu4(references: list[str], hypotheses: list[str]) -> float:
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def calc_rouge_l(references: list[str], hypotheses: list[str]) -> float:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / len(scores)


def calc_cosine(references: list[str], hypotheses: list[str]) -> tuple[float, list[float]]:
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    ref_emb = embedder.encode(references, convert_to_tensor=True)
    hyp_emb = embedder.encode(hypotheses, convert_to_tensor=True)
    sims = util.cos_sim(ref_emb, hyp_emb).diagonal()
    return sims.mean().item(), sims.tolist()


def pass_mark(value: float, threshold: float, higher_is_better: bool = True) -> str:
    ok = value >= threshold if higher_is_better else value <= threshold
    return "✓ PASS" if ok else "✗ FAIL"


def load_model(baseline: bool):
    if baseline:
        model_path = _cfg["model"]["base_model"]
        print(f"[베이스라인] {model_path} 로딩 중...")
    else:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"학습 결과가 없습니다. 먼저 python train.py 실행: {CONFIG_PATH}")
        model_path = str(_OUT)
        print(f"[Fine-tuned] {model_path} 로딩 중...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"디바이스: {device}")
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="파인튜닝 없이 base 모델로 평가")
    args = parser.parse_args()

    mode = "BASELINE (파인튜닝 없음)" if args.baseline else "FINE-TUNED"
    print(f"\n평가 모드: {mode}\n")

    model, tokenizer, device = load_model(args.baseline)

    df = pd.read_csv(TEST_DATA)
    if "generated_question" not in df.columns:
        df["generated_question"] = ""
    df["generated_question"] = df["generated_question"].astype(str).replace("nan", "")

    references, hypotheses = [], []

    print(f"\n질문 생성 중 ({len(df)}개)...\n" + "=" * 55)
    for i, row in df.iterrows():
        gen = generate_question(model, tokenizer, row["competency"], device)
        df.at[i, "generated_question"] = gen
        references.append(row["question"])
        hypotheses.append(gen)
        print(f"[{row['question_id']}] competency: {row['competency']}")
        print(f"  정답:  {row['question']}")
        print(f"  생성:  {gen}\n")

    bleu = calc_bleu4(references, hypotheses)
    rouge = calc_rouge_l(references, hypotheses)
    cosine, cos_list = calc_cosine(references, hypotheses)
    df["is_valid"] = [sim >= VALID_THRESHOLD for sim in cos_list]

    w = _cfg["eval"]["weights"]
    composite = w["bleu4"] * bleu + w["rouge_l"] * rouge + w["cosine"] * cosine
    comp_thr = _cfg["eval"]["composite_threshold"]

    ev = _cfg["eval"]
    print("=" * 55)
    print(f"  BLEU-4        : {bleu:.4f}   (기준 > {ev['bleu4_goal']})  {pass_mark(bleu, ev['bleu4_goal'])}")
    print(f"  ROUGE-L       : {rouge:.4f}   (기준 > {ev['rouge_l_goal']})  {pass_mark(rouge, ev['rouge_l_goal'])}")
    print(f"  Cosine 유사도 : {cosine:.4f}   (기준 > {ev['cosine_goal']})  {pass_mark(cosine, ev['cosine_goal'])}")
    print(f"  is_valid 비율 : {df['is_valid'].mean():.2%}  (threshold={VALID_THRESHOLD})")
    print(f"  {'─' * 47}")
    print(f"  통합 점수     : {composite:.4f}   (기준 > {comp_thr})  {pass_mark(composite, comp_thr)}")
    print("=" * 55)

    os.makedirs(str(_OUT), exist_ok=True)
    out_path = str(Path(RESULT_PATH).parent / "eval_results_baseline.json") if args.baseline else RESULT_PATH
    results = {
        "mode": "baseline" if args.baseline else "finetuned",
        "bleu4": bleu,
        "rouge_l": rouge,
        "cosine": cosine,
        "composite": composite,
        "weights": w,
        "valid_ratio": float(df["is_valid"].mean()),
        "n_samples": len(df),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df.to_csv(TEST_DATA, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {out_path}")
    print(f"test.csv 갱신 완료: {TEST_DATA}")


if __name__ == "__main__":
    main()
