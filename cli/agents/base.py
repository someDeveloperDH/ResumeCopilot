"""오케스트레이터-서브에이전트 공통 인터페이스.

run_session()(오케스트레이터)이 각 서브에이전트를 호출할 때 받는 표준 반환 타입을 정의한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class AgentResult(Generic[T]):
    """서브에이전트 호출 결과.

    tier_used: 실제로 응답을 생성한 모델 티어 ("smart" | "light" | "specialized" | "rule")
    fallback_used: 원래 의도한 티어가 아닌 폴백 경로로 처리되었는지 여부
    """

    data: T
    tier_used: str
    fallback_used: bool = False
