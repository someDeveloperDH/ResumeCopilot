# STEP 4 - 수치 기반 피드백 텍스트 생성
# ComparisonResult의 숫자를 사람이 읽을 수 있는 피드백으로 변환한다

from src.similarity.schema import ComparisonResult


def make_feedback(result: ComparisonResult) -> str:
    """
    ComparisonResult를 CLI에 표시할 피드백 텍스트로 변환한다.
    점수가 낮은 이유와 어떤 키워드가 필요한지를 명확히 전달하는 것이 목적.
    """
    stars = _to_stars(result.overall_score)
    urgency_label = {'high': '🔴 HIGH', 'medium': '🟡 MEDIUM', 'low': '🟢 LOW'}

    lines = [
        f"공고 부합도: {stars} ({result.overall_score:.2f})  [{urgency_label[result.fix_urgency]}]",
        f"키워드 커버리지: {result.coverage_ratio * 100:.0f}%  "
        f"({len(result.covered_keywords)}개 포함 / {len(result.missing_keywords)}개 누락)",
        f"가장 관련된 섹션: {result.section_scores.dominant()}",
    ]

    if result.priority_keywords:
        lines.append(f"삽입 필요 키워드: {', '.join(result.priority_keywords)}")

    if result.covered_keywords:
        lines.append(f"이미 포함된 키워드: {', '.join(result.covered_keywords[:5])}")

    return '\n'.join(lines)


def make_eval_report(result: ComparisonResult, rewritten: str) -> dict:
    """
    STEP 4 평가 지표: 재구성 전후 유사도 변화와 STAR 구조 충족 여부를 확인한다.
    """
    # STAR 구조 키워드가 재구성 문단에 있는지 간단히 확인
    star_markers = {
        'S': ['상황', '환경', '배경', '당시'],
        'T': ['담당', '역할', '과제', '목표'],
        'A': ['하여', '통해', '개발', '구현', '설계', '분석'],
        'R': ['결과', '단축', '향상', '개선', '달성'],
    }
    star_check = {
        k: any(m in rewritten for m in markers)
        for k, markers in star_markers.items()
    }

    # 재구성 후 키워드 삽입 여부 확인
    injected = [kw for kw in result.priority_keywords if kw in rewritten]

    return {
        'star_fulfilled': star_check,
        'all_star_met': all(star_check.values()),
        'keywords_injected': injected,
        'n_injected': len(injected),
    }


def _to_stars(score: float) -> str:
    """0~1 점수를 별 5개로 시각화한다."""
    filled = round(score * 5)
    return '★' * filled + '☆' * (5 - filled)
