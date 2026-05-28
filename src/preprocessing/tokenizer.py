# STEP 1, 2 - 형태소 분석 및 문단 분리
# kiwipiepy 사용 — Java 불필요, C++ 기반으로 KoNLPy보다 빠름
# KoNLPy에서 교체 시 get_tagger() 내부만 바꾸면 되도록 인터페이스를 동일하게 유지
#
# 문단 분리 방식 2가지:
#   split_paragraphs()     - 빠른 regex 기반 (기본값)
#   split_paragraphs_nlp() - kss + KoSBERT 의미 기반 (고도화)
# main.py에서 --split-mode 옵션으로 선택

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_tagger = None
_sbert_model = None


def get_tagger():
    """
    Kiwi 인스턴스를 싱글톤으로 관리.
    초기화 비용이 크기 때문에 한 번만 생성한다.
    """
    global _tagger
    if _tagger is None:
        from kiwipiepy import Kiwi
        _tagger = Kiwi()
    return _tagger


def _get_sbert():
    """
    KoSBERT 모델을 싱글톤으로 관리.
    aggregator.py의 sbert와 동일 모델이지만 문단 분리 전용으로 따로 로드해
    파이프라인 순서에 의존하지 않도록 독립성을 유지한다.
    """
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    return _sbert_model


def morphs(text: str, stem: bool = True) -> list[str]:
    """텍스트를 형태소 리스트로 변환한다."""
    tokens = get_tagger().tokenize(text)
    return [token.form for token in tokens]


def pos(text: str) -> list[tuple[str, str]]:
    """
    텍스트의 형태소와 품사를 함께 반환한다.
    Kiwi의 tag는 enum이므로 .name으로 문자열 변환 (예: 'NNG', 'VV')
    """
    tokens = get_tagger().tokenize(text)
    # kiwipiepy 0.17+ 에서 tag는 이미 str ('NNG', 'VV' 등)
    return [(token.form, token.tag) for token in tokens]


# ─────────────────────────────────────────────
# 방식 A: regex 기반 (기본값, 빠름)
# ─────────────────────────────────────────────

def split_paragraphs(text: str) -> list[str]:
    """
    빈 줄 2개 이상을 기준으로 문단을 분리한다.
    NLP 자동 탐지 방식으로 교체할 때 이 함수만 변경하면 된다.
    """
    paragraphs = re.split(r'\n{2,}', text)
    return [p.strip() for p in paragraphs if p.strip()]


# ─────────────────────────────────────────────
# 방식 B: kss + KoSBERT 의미 기반 (고도화)
# ─────────────────────────────────────────────

def split_paragraphs_nlp(
    text: str,
    sim_threshold: float = 0.65,
    min_sentences: int = 2,
) -> list[str]:
    """
    문장 경계 모델(kss)로 문장을 분리한 뒤,
    KoSBERT로 연속 문장 간 의미 유사도를 계산해 문단 경계를 탐지한다.

    동작 원리:
        문장 i와 i+1의 유사도가 sim_threshold 미만이면 → 문단 경계
        유사도가 높으면 → 같은 문단으로 묶음

    Args:
        sim_threshold: 문단 경계 판단 임계값 (낮출수록 더 잘게 분리)
        min_sentences: 한 문단의 최소 문장 수 (너무 잘게 쪼개지는 것 방지)
    """
    import kss

    # 1단계: kss로 문장 단위 분리
    sentences = kss.split_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # 2단계: KoSBERT로 각 문장 임베딩
    model = _get_sbert()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # 3단계: 연속 문장 쌍의 코사인 유사도 계산
    #   sim[i] = 문장 i와 i+1 사이의 유사도
    similarities = [
        float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
        for i in range(len(sentences) - 1)
    ]

    # 4단계: 유사도가 threshold 미만인 지점을 문단 경계로 확정
    paragraphs = []
    current = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < sim_threshold and len(current) >= min_sentences:
            # 의미 전환점 — 현재까지 모은 문장을 하나의 문단으로 확정
            paragraphs.append(' '.join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])

    if current:
        paragraphs.append(' '.join(current))

    return paragraphs


def auto_split(text: str, sim_threshold: float = 0.65) -> tuple[list[str], str]:
    """
    텍스트에 줄바꿈이 충분하면 regex, 없으면 NLP 방식을 자동으로 선택한다.

    자소서가 줄바꿈 없이 한 덩어리로 입력된 경우를 대비한 자동 탐지 로직.

    Returns:
        (문단 리스트, 사용된 방식 이름)
    """
    # 빈 줄이 1개 이상 있으면 regex로 충분히 처리 가능
    has_blank_lines = bool(re.search(r'\n{2,}', text))

    if has_blank_lines:
        return split_paragraphs(text), 'regex'
    else:
        # 줄바꿈이 없는 연속 텍스트 → NLP 모델로 의미 경계 탐지
        return split_paragraphs_nlp(text, sim_threshold), 'nlp'


# ─────────────────────────────────────────────
# STEP 2 평가
# ─────────────────────────────────────────────

def evaluate_paragraphs(paragraphs: list[str], method: str = 'regex') -> dict:
    """
    STEP 2 평가 지표를 산출한다.
    30자 미만은 의미 있는 문단이 아닐 가능성이 높아 경고 대상으로 분류.
    """
    lengths = [len(p) for p in paragraphs]
    return {
        'split_method': method,
        'n_paragraphs': len(paragraphs),
        'avg_length': round(sum(lengths) / len(lengths), 1) if lengths else 0,
        'short_paragraphs': [i for i, l in enumerate(lengths) if l < 30],
        'long_paragraphs': [i for i, l in enumerate(lengths) if l > 500],
    }
