# STEP 1 - 키워드 추출 방법 비교 및 통합
# Rule-based와 AI-based 결과를 비교해 최종 키워드 집합을 결정한다.
# overlap 비율이 낮으면 두 방법이 다른 관점을 포착했다는 뜻이므로 합산이 유리함


def compare_and_merge(
    rule_by_section: dict,
    ai_by_section: dict,
) -> dict:
    """
    두 방법의 섹션별 결과를 비교하고 합산한다.

    Returns:
        {
            'merged': dict,         섹션별 최종 키워드 리스트
            'all_keywords': list,   전체 키워드 (중복 제거)
            'eval': dict            비교 평가 지표
        }
    """
    merged = {}
    eval_per_section = {}

    for section in rule_by_section:
        rule_set = set(rule_by_section.get(section, []))
        ai_set = set(kw for kw, _ in ai_by_section.get(section, []))

        overlap = rule_set & ai_set
        union = rule_set | ai_set
        overlap_ratio = len(overlap) / len(union) if union else 0.0

        merged[section] = list(union)
        eval_per_section[section] = {
            'n_rule': len(rule_set),
            'n_ai': len(ai_set),
            'n_overlap': len(overlap),
            'overlap_ratio': round(overlap_ratio, 3),
            'n_merged': len(union),
        }

    # 섹션 경계를 넘어 중복 제거한 전체 키워드
    all_keywords = list({kw for kws in merged.values() for kw in kws})

    # 전체 overlap 비율 (신뢰도 지표 — 40% 이상이면 두 방법이 일관됨)
    total_rule = sum(e['n_rule'] for e in eval_per_section.values())
    total_ai = sum(e['n_ai'] for e in eval_per_section.values())
    total_overlap = sum(e['n_overlap'] for e in eval_per_section.values())
    total_union = total_rule + total_ai - total_overlap

    return {
        'merged': merged,
        'all_keywords': all_keywords,
        'eval': {
            'per_section': eval_per_section,
            'total_keywords': len(all_keywords),
            'overall_overlap_ratio': round(total_overlap / total_union, 3) if total_union else 0,
            # 40% 미만이면 두 방법 결과가 많이 달라 추가 검토 권장
            'consistent': (total_overlap / total_union >= 0.4) if total_union else False,
        },
    }
