# STEP 7 - 텍스트 변화율 측정 (Levenshtein 편집 거리)
# 원문과 최종본의 차이를 문자 단위로 측정해 AI 개입 정도를 수치화한다

import Levenshtein


def para_edit_ratio(original: str, modified: str) -> float:
    """
    두 문자열 간 Levenshtein 편집 거리를 최대 길이로 나눠 변화율(0~1)을 반환한다.
    0.0 = 원문 그대로, 1.0 = 완전히 다른 텍스트.
    """
    if not original and not modified:
        return 0.0

    distance = Levenshtein.distance(original, modified)
    max_len = max(len(original), len(modified))

    return round(distance / max_len, 3)


def classify_change(ratio: float) -> str:
    """
    변화율을 사람이 읽기 쉬운 범주로 분류한다.
    임계값은 실제 사용 데이터 없이 경험적으로 설정했으므로 추후 조정 가능.
    """
    if ratio < 0.01:
        return '원문 유지'
    elif ratio < 0.40:
        return '부분 수정'
    elif ratio < 0.80:
        return '대폭 수정'
    else:
        return '전면 재작성'


def evaluate_all(originals: list[str], finals: list[str]) -> list[dict]:
    """문단별 변화율과 분류를 계산해 반환한다."""
    results = []
    for i, (orig, final) in enumerate(zip(originals, finals)):
        ratio = para_edit_ratio(orig, final)
        results.append({
            'paragraph_idx': i,
            'edit_ratio': ratio,
            'change_level': classify_change(ratio),
        })
    return results
