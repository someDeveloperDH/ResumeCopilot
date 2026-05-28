# STEP 1 - AI-based 키워드 추출 (KeyBERT)
# KoSBERT 임베딩으로 텍스트 내 핵심 키워드를 중요도 순으로 추출한다.
# Rule-based와 달리 문맥 기반이라 의미적으로 중요한 단어를 잘 잡아냄

from keybert import KeyBERT

# KoSBERT를 백본으로 사용 — 한국어에 특화된 문장 임베딩 모델
MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'

_model = None


def get_model() -> KeyBERT:
    """KeyBERT 모델을 싱글톤으로 관리. 첫 호출 시 모델을 다운로드한다."""
    global _model
    if _model is None:
        _model = KeyBERT(MODEL_NAME)
    return _model


def extract(text: str, top_n: int = 15) -> list[tuple[str, float]]:
    """
    텍스트에서 상위 top_n개 키워드를 (단어, 점수) 형태로 반환한다.
    2-gram까지 허용해 '데이터 파이프라인' 같은 복합 키워드도 추출 가능.
    """
    if not text.strip():
        return []

    model = get_model()
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),  # 단일어 + 2단어 구문 모두 추출
        stop_words=None,               # 한국어 불용어는 전처리에서 처리
        top_n=top_n,
    )
    return keywords  # [(keyword, score), ...]


def extract_by_section(sections: dict, top_n: int = 10) -> dict:
    """섹션별로 키워드를 추출한다."""
    return {
        section: extract(text, top_n)
        for section, text in sections.items()
        if text
    }


def evaluate(section_keywords: dict) -> dict:
    """STEP 1 평가 지표 산출."""
    all_kws = [kw for kws in section_keywords.values() for kw, _ in kws]
    avg_score = 0.0
    count = 0
    for kws in section_keywords.values():
        for _, score in kws:
            avg_score += score
            count += 1

    return {
        'method': 'ai_based',
        'n_keywords_per_section': {s: len(kws) for s, kws in section_keywords.items()},
        'total_keywords': len(all_kws),
        'avg_importance_score': round(avg_score / count, 3) if count else 0,
        'valid': len(all_kws) >= 5,
    }
