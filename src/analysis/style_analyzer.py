# STEP 7 - 언어 스타일 분석 (Perplexity, TTR, 문장 길이)
# AI가 생성한 텍스트는 자연스럽고 균일한 경향이 있어 Perplexity가 낮고
# 어휘가 반복되어 TTR(어휘 다양성)이 낮으며 문장이 길어지는 경향이 있음

import re
import math
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from src.preprocessing.tokenizer import morphs

# 한국어 GPT2 모델 — Perplexity 계산용
KOGPT2_MODEL = 'skt/kogpt2-base-v2'

_gpt2_model = None
_gpt2_tokenizer = None


def _load_gpt2():
    """KoGPT2 모델과 토크나이저를 싱글톤으로 로드한다."""
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        _gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained(KOGPT2_MODEL)
        _gpt2_model = GPT2LMHeadModel.from_pretrained(KOGPT2_MODEL)
        _gpt2_model.eval()


def perplexity(text: str) -> float:
    """
    KoGPT2로 텍스트의 Perplexity를 계산한다.
    낮을수록 언어 모델이 예측하기 쉬운 문장 → AI 생성 텍스트의 특성.
    높을수록 불규칙하고 인간적인 표현.
    """
    _load_gpt2()

    tokens = _gpt2_tokenizer(text, return_tensors='pt')
    input_ids = tokens['input_ids']

    # 너무 짧으면 Perplexity가 불안정하므로 최솟값 반환
    if input_ids.shape[1] < 5:
        return 0.0

    with torch.no_grad():
        outputs = _gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss

    return round(math.exp(loss.item()), 2)


def ttr(text: str) -> float:
    """
    Type-Token Ratio: 고유 형태소 수 / 전체 형태소 수.
    낮을수록 어휘가 반복되는 경향 → AI 생성 텍스트의 특성.
    """
    tokens = morphs(text)
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 3)


def avg_sentence_length(text: str) -> float:
    """
    평균 문장 길이 (글자 수 기준).
    AI 생성 텍스트는 문장이 균일하게 길어지는 경향이 있음.
    """
    # 마침표/물음표/느낌표 기준으로 문장 분리
    sentences = re.split(r'[.!?。]\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.0
    return round(sum(len(s) for s in sentences) / len(sentences), 1)


def analyze(text: str) -> dict:
    """단일 텍스트에 대해 스타일 지표를 모두 계산한다."""
    return {
        'perplexity': perplexity(text),
        'ttr': ttr(text),
        'avg_sentence_length': avg_sentence_length(text),
    }


def analyze_delta(original: str, modified: str) -> dict:
    """원문과 수정본 간 스타일 변화량을 계산한다."""
    orig_stats = analyze(original)
    mod_stats = analyze(modified)

    return {
        'perplexity_before': orig_stats['perplexity'],
        'perplexity_after': mod_stats['perplexity'],
        # 낮아지면 AI 특성으로 변화한 것
        'perplexity_delta': round(mod_stats['perplexity'] - orig_stats['perplexity'], 2),

        'ttr_before': orig_stats['ttr'],
        'ttr_after': mod_stats['ttr'],
        'ttr_delta': round(mod_stats['ttr'] - orig_stats['ttr'], 3),

        'sent_len_before': orig_stats['avg_sentence_length'],
        'sent_len_after': mod_stats['avg_sentence_length'],
        'sent_len_delta': round(
            mod_stats['avg_sentence_length'] - orig_stats['avg_sentence_length'], 1
        ),
    }
