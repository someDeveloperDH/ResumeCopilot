# jintea1/part1/evaluator.py 기반
# Rule-based / AI-based / Hybrid 세 방법의 Precision / Recall / F1을 비교한다
# test1에는 없던 정량 평가 — 방법 선택의 근거를 수치로 제공

from __future__ import annotations

from statistics import mean
from src.keyword import rule_based, ai_based, hybrid
from src.preprocessing.cleaner import clean_job_posting
from src.preprocessing.section_splitter import split_sections


# ─── 내부 유틸 ────────────────────────────────────────────────────────────────

def _flatten(result: dict[str, list[dict]]) -> set[str]:
    return {item['keyword'] for items in result.values() for item in items if item.get('keyword')}


def _norm(s: str) -> str:
    return ''.join(s.lower().split())


def _match(pred: str, gt: str) -> bool:
    p, g = _norm(pred), _norm(gt)
    if not p or not g or len(p) <= 1 or len(g) <= 1:
        return False
    return p == g or p in g or g in p


def _metrics(predictions: set[str], ground_truth: set[str]) -> dict[str, float]:
    matched_p = {p for p in predictions if any(_match(p, g) for g in ground_truth)}
    matched_g = {g for g in ground_truth if any(_match(p, g) for p in predictions)}
    precision = len(matched_p) / len(predictions) if predictions else 0.0
    recall    = len(matched_g) / len(ground_truth) if ground_truth else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': round(precision, 3),
        'recall':    round(recall, 3),
        'f1':        round(f1, 3),
    }


def _safe_extract(extractor_fn, sections: dict) -> tuple[dict, str | None]:
    try:
        return extractor_fn(sections), None
    except Exception as e:
        return {s: [] for s in sections}, str(e)


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def compare(jd_text: str, ground_truth_keywords: list[str]) -> dict:
    """
    하나의 채용공고 텍스트에 대해 세 방법의 성능을 비교한다.

    Args:
        jd_text: 원본 채용공고 텍스트
        ground_truth_keywords: 정답 키워드 리스트 (수동 레이블 또는 샘플)

    Returns:
        {
            'rule':   {'predictions': set, 'metrics': {'precision', 'recall', 'f1'}},
            'ai':     {...},
            'hybrid': {...},
            'best':   'hybrid',   # F1 기준 최고 방법
        }
    """
    cleaned  = clean_job_posting(jd_text)
    sections = split_sections(cleaned)
    gt       = set(ground_truth_keywords)

    rule_result,   rule_err   = _safe_extract(rule_based.extract_keywords, sections)
    ai_result,     ai_err     = _safe_extract(
        lambda s: {sec: [{'keyword': kw, 'pos': '', 'type': '', 'weight': 1.0,
                           'score': sc, 'hybrid_score': sc, 'source': 'AI'}
                          for kw, sc in pairs]
                   for sec, pairs in ai_based.extract_by_section(s).items()},
        sections,
    )
    hybrid_result, hybrid_err = _safe_extract(hybrid.extract_keywords, sections)

    results = {
        'rule':   {'predictions': _flatten(rule_result),   'metrics': _metrics(_flatten(rule_result),   gt), 'error': rule_err},
        'ai':     {'predictions': _flatten(ai_result),     'metrics': _metrics(_flatten(ai_result),     gt), 'error': ai_err},
        'hybrid': {'predictions': _flatten(hybrid_result), 'metrics': _metrics(_flatten(hybrid_result), gt), 'error': hybrid_err},
    }

    # F1 기준 최고 방법 결정
    best = max(('rule', 'ai', 'hybrid'), key=lambda k: results[k]['metrics']['f1'])
    results['best'] = best
    return results


def print_report(comparison: dict) -> None:
    """비교 결과를 터미널에 표로 출력한다."""
    print('\n[키워드 추출 방법 비교]')
    print(f"{'방법':<10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print('-' * 42)
    for method in ('rule', 'ai', 'hybrid'):
        m = comparison[method]['metrics']
        mark = ' ← 최고' if method == comparison['best'] else ''
        print(f"{method:<10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}{mark}")
        if comparison[method].get('error'):
            print(f"  ⚠ 오류: {comparison[method]['error']}")
    print()
