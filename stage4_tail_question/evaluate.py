"""
Stage 4 CPU 버전 - 꼬리질문 생성 평가
실행:
  python evaluate.py            # fine-tuned CPU 모델 평가
  python evaluate.py --baseline # base 모델 평가
"""
import argparse
import json
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import SentenceTransformer, util

_HERE = Path(__file__).parent

def load_cfg():
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))

_cfg = load_cfg()
OUTPUT_DIR = _HERE / _cfg["output"]["dir"]
TEST_DATA = _HERE / _cfg["data"]["test_csv"]
RESULT_PATH = OUTPUT_DIR / "eval_results.json"
BASE_MODEL = _cfg["model"]["base_model"]
MAX_LEN = int(_cfg["model"]["max_seq_length"])
MAX_NEW = int(_cfg["model"].get("max_new_tokens", 80))

SYSTEM_TAIL = "IT/AI 직무 자기소개서 코치. 질문, 답변, 의도점수, 직무를 보고 구체적 경험을 끌어내는 한국어 꼬리질문 하나 생성."
YES_NO_PATTERNS = ["예", "아니요", "네", "맞아요", "아니오", "그렇습니다", "아닙니다"]

def build_prompt(row) -> str:
    return (
        f"시스템: {SYSTEM_TAIL}\n"
        f"사용자: [TASK:tail_question] question: {row['question']} answer: {row['answer']} "
        f"intent_score: {row['intent_score']} job: {row['suitable_job']}\n"
        f"어시스턴트:"
    )

def generate(model, tokenizer, row, device):
    prompt = build_prompt(row)
    enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # 뒤에 다른 프롬프트가 이어지는 경우 잘라냄
    for stop in ["사용자:", "시스템:", "어시스턴트:"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    return text

def calc_bleu4(refs, hyps):
    return corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps], smoothing_function=SmoothingFunction().method1)

def calc_rouge_l(refs, hyps):
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
    return float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]))

def calc_cosine(refs, hyps):
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device="cpu")
    ref_emb = embedder.encode(refs, convert_to_tensor=True)
    hyp_emb = embedder.encode(hyps, convert_to_tensor=True)
    sims = util.cos_sim(ref_emb, hyp_emb).diagonal()
    return sims.mean().item(), sims.tolist()

def is_open_question(text):
    text = str(text).strip()
    if len(text) <= 15:
        return False
    if "?" not in text and "요" not in text:
        return False
    no_q = text.replace("?", "").strip()
    return not any(no_q.endswith(p) for p in YES_NO_PATTERNS)

def pass_mark(value, threshold, higher=True):
    ok = value >= threshold if higher else value <= threshold
    return "✓ PASS" if ok else "✗ FAIL"

def load_model(baseline):
    device = torch.device("cpu")
    model_path = BASE_MODEL if baseline else str(OUTPUT_DIR)
    if not baseline and not (OUTPUT_DIR / "config.json").exists():
        raise FileNotFoundError(f"학습된 모델이 없습니다. 먼저 python train.py 실행: {OUTPUT_DIR}")
    print(f"모델 로딩: {model_path} (CPU)")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # 저장된 tokenizer 기준으로 embedding 크기 동기화
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.baseline)
    df = pd.read_csv(TEST_DATA)
    df["generated_tail_question"] = df.get("generated_tail_question", pd.Series(dtype="str")).astype(str).replace("nan", "")

    refs, hyps = [], []
    print(f"\n꼬리질문 생성 중 ({len(df)}개)...")
    for i, row in df.iterrows():
        gen = generate(model, tokenizer, row.to_dict(), device)
        df.at[i, "generated_tail_question"] = gen
        refs.append(str(row["tail_question"]))
        hyps.append(gen)
        print(f"[{i+1}/{len(df)}] 생성: {gen}")

    bleu = calc_bleu4(refs, hyps)
    rouge = calc_rouge_l(refs, hyps)
    cosine, _ = calc_cosine(refs, hyps)
    open_ratio = sum(is_open_question(h) for h in hyps) / max(1, len(hyps))

    ev = _cfg["eval"]
    print("\n" + "=" * 55)
    print(f"BLEU-4        : {bleu:.4f} (기준 > {ev['bleu4_threshold']}) {pass_mark(bleu, ev['bleu4_threshold'])}")
    print(f"ROUGE-L       : {rouge:.4f} (기준 > {ev['rouge_l_threshold']}) {pass_mark(rouge, ev['rouge_l_threshold'])}")
    print(f"Cosine 유사도 : {cosine:.4f} (기준 > {ev['cosine_threshold']}) {pass_mark(cosine, ev['cosine_threshold'])}")
    print(f"열린 질문 비율: {open_ratio:.2%}")
    print("=" * 55)

    out_path = OUTPUT_DIR / ("eval_results_baseline.json" if args.baseline else "eval_results.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"mode": "baseline" if args.baseline else "finetuned", "bleu4": bleu, "rouge_l": rouge, "cosine": cosine, "open_question_ratio": open_ratio, "n_samples": len(df)}
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(TEST_DATA, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {out_path}")
    print(f"tail_test.csv 갱신 완료: {TEST_DATA}")

if __name__ == "__main__":
    main()
