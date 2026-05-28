# jintea1/part1/hybrid_extractor.py 기반
# Rule-based 후보 + KeyBERT 점수 + SafeNet(HARD_SKILLS 보장) 세 단계 결합
# test1에는 없던 Hybrid 방식 — 정밀도와 재현율의 균형을 맞춤

from __future__ import annotations

from src.keyword import rule_based, ai_based
from src.keyword.constants import HARD_SKILLS, TECH_NORMALIZE, WEIGHTS

# Hybrid 점수 = ALPHA * 섹션가중치 + BETA * KeyBERT점수
HYBRID_ALPHA     = 0.3
HYBRID_BETA      = 0.7
HYBRID_THRESHOLD = 0.25   # KeyBERT 점수 이 이상이면 Hybrid로 선택


def _safe_net() -> frozenset[str]:
    """Hard Skills + 기술명 정규화 사전 → 반드시 포함할 키워드 집합."""
    return frozenset(HARD_SKILLS | set(TECH_NORMALIZE.values()))


def _score_candidates(text: str, candidates: list[str]) -> dict[str, float]:
    """Rule 후보를 KeyBERT에 넘겨 점수를 매긴다."""
    if not candidates:
        return {}
    try:
        model = ai_based.get_model()
        scored = model.extract_keywords(
            text,
            candidates=candidates,
            top_n=len(candidates),
            use_mmr=True,
            diversity=0.5,
        )
        return {str(kw): float(score) for kw, score in scored}
    except Exception:
        return {kw: 0.0 for kw in candidates}


def extract_keywords(sections: dict[str, str]) -> dict[str, list[dict]]:
    """
    Rule 후보를 추출한 뒤 KeyBERT 점수로 필터링한다.
    SafeNet에 속하는 키워드는 KeyBERT 점수가 낮아도 포함된다.

    결과 각 항목:
        keyword, pos, type, weight, score (KeyBERT), hybrid_score, source
    """
    safe = _safe_net()
    result = {}

    for section, text in sections.items():
        if not text:
            result[section] = []
            continue

        weight = WEIGHTS.get(section, 1.0)
        rule_result = rule_based.extract_keywords({section: text})
        candidates = [item['keyword'] for item in rule_result.get(section, [])]
        score_map  = _score_candidates(text, candidates)

        items = []
        w_norm = weight / 1.5   # 최대 가중치(1.5)로 정규화

        for item in rule_result.get(section, []):
            kw    = item['keyword']
            score = score_map.get(kw, 0.0)
            h_score = (w_norm * HYBRID_ALPHA) + (score * HYBRID_BETA)

            if score >= HYBRID_THRESHOLD:
                source = 'Hybrid'
            elif kw in safe:
                source = 'SafeNet'
            else:
                continue   # 점수 낮고 SafeNet도 아니면 제외

            items.append({
                'keyword':      kw,
                'pos':          item.get('pos'),
                'type':         item.get('type'),
                'weight':       weight,
                'score':        round(score, 4),
                'hybrid_score': round(h_score, 4),
                'source':       source,
            })

        # hybrid_score 내림차순 정렬
        result[section] = sorted(items, key=lambda x: x['hybrid_score'], reverse=True)

    return result


def to_keyword_list(hybrid_result: dict[str, list[dict]]) -> list[str]:
    """Hybrid 결과에서 키워드 문자열 리스트만 추출한다 (aggregator에서 사용)."""
    seen: set[str] = set()
    keywords: list[str] = []
    for items in hybrid_result.values():
        for item in items:
            kw = item['keyword']
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
    return keywords


def evaluate(section_keywords: dict) -> dict:
    """STEP 1 Hybrid 평가 지표."""
    all_kws = [item['keyword'] for items in section_keywords.values() for item in items]
    sources = {'Hybrid': 0, 'SafeNet': 0}
    for items in section_keywords.values():
        for item in items:
            src = item.get('source', 'Hybrid')
            sources[src] = sources.get(src, 0) + 1
    avg_score = 0.0
    count = sum(len(v) for v in section_keywords.values())
    for items in section_keywords.values():
        avg_score += sum(item.get('hybrid_score', 0) for item in items)
    return {
        'method': 'hybrid',
        'n_keywords_per_section': {s: len(v) for s, v in section_keywords.items()},
        'total_keywords': len(all_kws),
        'source_breakdown': sources,
        'avg_hybrid_score': round(avg_score / count, 4) if count else 0,
        'valid': len(all_kws) >= 5,
    }
