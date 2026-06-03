"""
Stage 4 - 꼬리질문 생성 평가

실행:
    python evaluate.py              # fine-tuned 모델 평가 (best_lora.pt)
    python evaluate.py --baseline   # base 모델 평가 (파인튜닝 없이)

- 정량 지표: BLEU-4, ROUGE-L, Cosine 유사도
- 자동 품질 검사: Yes/No 불가 여부
"""

import warnings
warnings.filterwarnings("ignore", message="Both.*max_new_tokens.*max_length")

import argparse
import json
import yaml
import torch
import pandas as pd
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

_OUT         = _HERE / _cfg["output"]["dir"]
PT_PATH      = str(_OUT / "best_lora.pt")
CONFIG_PATH  = str(_OUT / "train_config.json")
TEST_DATA    = str(_HERE / _cfg["data"]["test_csv"])
RESULT_PATH  = str(_OUT / "eval_results.json")

SYSTEM_TAIL = (
    "당신은 IT/AI 직무 자기소개서 전문 코치입니다. "
    "질문, 답변, 의도 반영도(intent_score), 적합 직무(job)를 바탕으로 "
    "사용자가 더 구체적인 경험을 작성할 수 있도록 꼬리질문 하나를 생성하세요. "
    "Yes/No로 답할 수 없어야 하며, 구체적 경험을 끌어내야 합니다. "
    "반드시 한국어로만 작성하세요."
)

# Yes/No 단답 유도 키워드 (이 단어로만 구성된 짧은 답변 방지용 검사)
YES_NO_PATTERNS = ["예", "아니요", "네", "맞아요", "아니오", "그렇습니다", "아닙니다"]


# ── 꼬리질문 생성 ─────────────────────────────────────────────────────
def generate_tail_question(model, tokenizer, row: dict) -> str:
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_TAIL}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"[TASK:tail_question] "
        f"question: {row['question']} "
        f"answer: {row['answer']} "
        f"intent_score: {row['intent_score']} "
        f"job: {row['suitable_job']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── 지표 ─────────────────────────────────────────────────────────────
def calc_bleu4(references, hypotheses) -> float:
    refs = [[r.split()] for r in references]
    hyps = [h.split() for h in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def calc_rouge_l(references, hypotheses) -> float:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    return sum(scores) / len(scores)


def calc_cosine(references, hypotheses) -> tuple[float, list[float]]:
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    ref_emb  = embedder.encode(references, convert_to_tensor=True)
    hyp_emb  = embedder.encode(hypotheses, convert_to_tensor=True)
    sims     = util.cos_sim(ref_emb, hyp_emb).diagonal()
    return sims.mean().item(), sims.tolist()


def is_open_question(text: str) -> bool:
    """Yes/No로 답할 수 없는 열린 질문인지 자동 검사"""
    if "?" not in text and "요?" not in text:
        return False
    lower = text.strip().replace("?", "")
    for pat in YES_NO_PATTERNS:
        if lower.endswith(pat):
            return False
    return len(text) > 15  # 너무 짧은 질문 제외


def pass_mark(value, threshold, higher=True) -> str:
    return "✓ PASS" if (value >= threshold if higher else value <= threshold) else "✗ FAIL"


# ── 모델 로드 ────────────────────────────────────────────────────────
def load_model(baseline: bool):
    if baseline:
        base_model  = _cfg["model"]["base_model"]
        max_seq_len = _cfg["model"]["max_seq_length"]
        print(f"[베이스라인] {base_model} (LoRA 미적용)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[Fine-tuned] {cfg['base_model']} + {PT_PATH}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model"],
            max_seq_length=cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            target_modules=cfg["lora_targets"],
            lora_alpha=cfg["lora_alpha"],
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

    # 테스트 데이터
    df = pd.read_csv(TEST_DATA)
    df["generated_tail_question"] = df.get("generated_tail_question", pd.Series(dtype="str")).astype(str).replace("nan", "")
    references, hypotheses = [], []

    print(f"\n꼬리질문 생성 중 ({len(df)}개)...\n" + "="*60)
    for i, row in df.iterrows():
        gen = generate_tail_question(model, tokenizer, row.to_dict())
        df.at[i, "generated_tail_question"] = gen
        references.append(row["tail_question"])
        hypotheses.append(gen)

        print(f"[{row['suitable_job']} | intent={row['intent_score']}]")
        print(f"  Q:    {row['question'][:50]}...")
        print(f"  정답: {row['tail_question']}")
        print(f"  생성: {gen}\n")

    # 정량 지표
    bleu            = calc_bleu4(references, hypotheses)
    rouge           = calc_rouge_l(references, hypotheses)
    cosine, cos_lst = calc_cosine(references, hypotheses)

    # 자동 품질 검사 (Yes/No 불가 여부)
    open_q_ratio = sum(is_open_question(h) for h in hypotheses) / len(hypotheses)

    # intent_score 구간별 분석
    df["is_open_question"] = [is_open_question(h) for h in hypotheses]
    print("="*60)
    print("  [정량 지표]")
    print(f"  BLEU-4        : {bleu:.4f}   (기준 > 0.35) {pass_mark(bleu, 0.35)}")
    print(f"  ROUGE-L       : {rouge:.4f}   (기준 > 0.40) {pass_mark(rouge, 0.40)}")
    print(f"  Cosine 유사도 : {cosine:.4f}   (기준 > 0.80) {pass_mark(cosine, 0.80)}")
    print()
    print("  [자동 품질 검사]")
    print(f"  열린 질문 비율: {open_q_ratio:.2%}  (Yes/No 불가 여부)")
    print()
    print("  [intent_score 구간별 열린 질문 비율]")
    for label, cond in [("0~40(저)", df["intent_score"] <= 40),
                        ("41~70(중)", (df["intent_score"] > 40) & (df["intent_score"] <= 70)),
                        ("71~100(고)", df["intent_score"] > 70)]:
        sub = df[cond]
        if len(sub) > 0:
            ratio = sub["is_open_question"].mean()
            print(f"    {label}: {ratio:.2%}  ({len(sub)}개)")
    print("="*60)

    # 저장 (baseline이면 별도 파일)
    out_path = (
        str(Path(RESULT_PATH).parent / "eval_results_baseline.json")
        if args.baseline else RESULT_PATH
    )
    results = {
        "mode":                "baseline" if args.baseline else "finetuned",
        "bleu4":               bleu,
        "rouge_l":             rouge,
        "cosine":              cosine,
        "open_question_ratio": open_q_ratio,
        "n_samples":           len(df),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df.to_csv(TEST_DATA, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {out_path}")
    print(f"tail_test.csv 갱신 완료: {TEST_DATA}")


if __name__ == "__main__":
    main()
