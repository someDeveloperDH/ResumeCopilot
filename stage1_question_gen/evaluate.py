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
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast

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
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def calc_cosine(references: list[list[str]], hypotheses: list[str]) -> tuple[float, list[float]]:
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    hyp_emb = embedder.encode(hypotheses, convert_to_tensor=True)

    ref_emb_cache = {}
    sims = []
    for ref_group, h_emb in zip(references, hyp_emb):
        key = tuple(ref_group)
        if key not in ref_emb_cache:
            ref_emb_cache[key] = embedder.encode(ref_group, convert_to_tensor=True)
        sims.append(util.cos_sim(h_emb.unsqueeze(0), ref_emb_cache[key]).max().item())
    return sum(sims) / len(sims), sims


def pass_mark(value: float, threshold: float, higher_is_better: bool = True) -> str:
    ok = value >= threshold if higher_is_better else value <= threshold
    return "✓ PASS" if ok else "✗ FAIL"


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


def load_model(baseline: bool):
    if baseline:
        model_path = _cfg["model"]["base_model"]
        print(f"[베이스라인] {model_path} 로딩 중...")
    else:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"학습 결과가 없습니다. 먼저 python train.py 실행: {CONFIG_PATH}")
        model_path = str(_OUT)
        print(f"[Fine-tuned] {model_path} 로딩 중...")

    tokenizer = load_kogpt2_tokenizer(model_path)
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

    refs_by_competency = df.groupby("competency")["question"].apply(list).to_dict()

    references, hypotheses = [], []

    print(f"\n질문 생성 중 ({len(df)}개)...\n" + "=" * 55)
    for i, row in df.iterrows():
        gen = generate_question(model, tokenizer, row["competency"], device)
        df.at[i, "generated_question"] = gen
        references.append(refs_by_competency[row["competency"]])
        hypotheses.append(gen)
        print(f"[{row['question_id']}] competency: {row['competency']}")
        print(f"  정답:  {row['question']}")
        print(f"  생성:  {gen}\n")

    cosine, cos_list = calc_cosine(references, hypotheses)
    df["is_valid"] = [sim >= VALID_THRESHOLD for sim in cos_list]

    ev = _cfg["eval"]
    print("=" * 55)
    print(f"  Cosine 유사도 : {cosine:.4f}   (기준 > {ev['cosine_goal']})  {pass_mark(cosine, ev['cosine_goal'])}")
    print(f"  is_valid 비율 : {df['is_valid'].mean():.2%}  (threshold={VALID_THRESHOLD})")
    print("=" * 55)

    os.makedirs(str(_OUT), exist_ok=True)
    out_path = str(Path(RESULT_PATH).parent / "eval_results_baseline.json") if args.baseline else RESULT_PATH
    results = {
        "mode": "baseline" if args.baseline else "finetuned",
        "cosine": cosine,
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
