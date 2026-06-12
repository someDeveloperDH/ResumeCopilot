"""ConsistencyAgent — 일관성 검사 (specialized 티어, 변경 없음)."""

from __future__ import annotations

from consistency import check_consistency_pairwise
from session import Session

from .base import AgentResult


class ConsistencyAgent:
    """직전/누적 답변과의 일관성을 검사한다.

    고정 티어: specialized (rule 키워드 충돌 + NLI(mDeBERTa) + SBERT(KR-SBERT), 변경 없음).
    """

    def check(self, session: Session, latest_answer: str) -> AgentResult[dict]:
        result = check_consistency_pairwise(session, latest_answer)
        return AgentResult(data=result, tier_used="specialized")
