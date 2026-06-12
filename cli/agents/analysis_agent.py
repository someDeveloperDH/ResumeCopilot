"""AnalysisAgent — intent_score/적합 직무 예측 (specialized 티어, 변경 없음)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import models
from coach import estimate_score

from .base import AgentResult


@dataclass
class AnalysisOutput:
    intent_score: float
    job: str


class AnalysisAgent:
    """답변의 의도 반영도(intent_score)와 적합 직무를 예측한다.

    고정 티어: specialized (ModernBERT/KoELECTRA 멀티태스크 모델, 변경 없음).
    체크포인트가 없으면 rule 기반 임시값으로 폴백한다.
    """

    def __init__(self, analysis_model: models.AnalysisModel | None):
        self._model = analysis_model

    def predict(self, question: str, answer: str) -> AgentResult[AnalysisOutput]:
        if self._model is not None:
            score, job = self._model.predict(question, answer)
            if score <= 0.0:
                # stage3 회귀 헤드는 pearson_r≈-0.025로 거의 상수만 출력하므로 0 이하면 보정한다.
                print("  [분석 보정] intent_score 회귀값이 비정상(<=0) → rule 기반 추정치로 대체합니다.")
                score = estimate_score(question, answer)
                return AgentResult(data=AnalysisOutput(score, job), tier_used="specialized", fallback_used=True)
            return AgentResult(data=AnalysisOutput(score, job), tier_used="specialized")

        print("  [분석 대체값] 분석 모델 체크포인트가 없어 임시값을 사용합니다.")
        score, job = 50.0, random.choice(["backend", "ai_ml", "product"])
        return AgentResult(data=AnalysisOutput(score, job), tier_used="rule", fallback_used=True)
