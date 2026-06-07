"""소재 냉장고와 기존 자소서 카드 재활용 도우미."""

from __future__ import annotations

import re


MEMORY_PROMPTS = [
    "반복 작업을 줄인 일",
    "실수해서 다시 고친 일",
    "누군가를 도와준 일",
    "처음 배운 기술",
    "팀에서 의견이 갈린 일",
]


def print_memory_prompts() -> None:
    print("\n완성된 경험이 없어도 됩니다. 아래 중 최근 기억나는 것 하나를 골라 한 줄로 적어주세요.")
    for idx, prompt in enumerate(MEMORY_PROMPTS, start=1):
        print(f"  [{idx}] {prompt}")


def recommend_cards(question: str, sessions: list[dict], limit: int = 3) -> list[dict]:
    cards = [s.get("answer_card", {}) for s in sessions if s.get("answer_card")]
    scored = []
    q_tokens = _tokens(question)
    for card in cards:
        text = " ".join(
            [
                card.get("core_experience", ""),
                card.get("final_answer", ""),
                " ".join(card.get("competencies", [])),
                " ".join(card.get("reuse_angles", [])),
            ]
        )
        overlap = len(q_tokens & _tokens(text))
        if overlap:
            scored.append((overlap, card))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [card for _, card in scored[:limit]]


def print_recommendations(cards: list[dict]) -> None:
    if not cards:
        return
    print("\n오늘 질문과 연결 가능한 기존 소재")
    for idx, card in enumerate(cards, start=1):
        angles = ", ".join(card.get("reuse_angles", [])) or "다른 관점"
        print(f"  [{idx}] {card.get('core_experience', '이전 경험')}")
        print(f"      → {angles} 관점으로 재활용 가능")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", text)}
