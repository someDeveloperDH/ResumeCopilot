"""저장된 자소서 카드 기반 준비도 지도 계산."""

from __future__ import annotations

from coach import extract_metrics, extract_tools


COMPETENCIES = ["문제해결", "협업", "성장", "실패경험", "주도성"]


def calculate_readiness(sessions: list[dict]) -> dict[str, int]:
    cards = [s.get("answer_card", {}) for s in sessions if s.get("answer_card")]
    if not cards:
        return {
            "경험 다양성": 0,
            "정량 성과": 0,
            "기술 키워드": 0,
            "STAR 구조": 0,
            "직무 연관성": 0,
        }

    all_competencies = set()
    metric_cards = 0
    tool_cards = 0
    star_scores = []
    job_scores = []

    for card in cards:
        all_competencies.update(card.get("competencies", []))
        text = " ".join(
            [
                card.get("final_answer", ""),
                " ".join(card.get("metrics", [])),
                " ".join(card.get("tools", [])),
            ]
        )
        if card.get("metrics") or extract_metrics(text):
            metric_cards += 1
        if card.get("tools") or extract_tools(text):
            tool_cards += 1

        star = card.get("star", {})
        filled = sum(1 for key in ["situation", "task", "action", "result"] if star.get(key))
        star_scores.append(filled / 4)
        job_scores.append(min(1.0, (card.get("after_score", 0) or 0) / 100))

    total = max(1, len(cards))
    return {
        "경험 다양성": min(100, round(len(all_competencies) / len(COMPETENCIES) * 100)),
        "정량 성과": round(metric_cards / total * 100),
        "기술 키워드": round(tool_cards / total * 100),
        "STAR 구조": round(sum(star_scores) / total * 100),
        "직무 연관성": round(sum(job_scores) / total * 100),
    }


def weakest_area(readiness: dict[str, int]) -> str:
    if not readiness:
        return "정량 성과"
    return min(readiness.items(), key=lambda item: item[1])[0]


def mission_for(area: str) -> str:
    missions = {
        "경험 다양성": "아직 부족한 역량 카드가 있습니다. 오늘은 새로운 관점의 경험 소재를 찾아볼게요.",
        "정량 성과": "정량 성과가 부족합니다. 시간 단축, 오류 감소, 사용자 수처럼 숫자가 있는 경험을 찾아볼게요.",
        "기술 키워드": "기술 키워드가 부족합니다. 사용한 도구나 라이브러리가 드러나는 경험을 찾아볼게요.",
        "STAR 구조": "STAR 구조가 약합니다. 상황-역할-행동-결과가 이어지는 경험을 정리해볼게요.",
        "직무 연관성": "직무 연관성이 약합니다. 질문과 직무 키워드에 더 가까운 경험을 찾아볼게요.",
    }
    return missions.get(area, missions["정량 성과"])


def render_bar(value: int) -> str:
    filled = max(0, min(10, round(value / 10)))
    return "█" * filled + "░" * (10 - filled)


def print_readiness(title: str, readiness: dict[str, int]) -> None:
    print(f"\n{title}")
    for label, value in readiness.items():
        print(f"  {label:<8} {render_bar(value)} {value}%")


def print_readiness_delta(before: dict[str, int], after: dict[str, int]) -> None:
    print("\n오늘의 변화")
    changed = False
    for label in after:
        if after[label] != before.get(label, 0):
            changed = True
            print(f"  {label:<8} {before.get(label, 0)}% → {after[label]}%")
    if not changed:
        print("  준비도 수치는 유지되었습니다. 소재 저장 자체가 다음 확장의 기반이 됩니다.")
