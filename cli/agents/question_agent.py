"""QuestionAgent — 오늘의 질문 생성 (light 티어, 변경 없음)."""

from __future__ import annotations

import models

from .base import AgentResult


class QuestionAgent:
    """오늘의 질문을 생성한다.

    고정 티어: light (기존 생성 모델 — GPU: Llama-3.1-8B-4bit / CPU: KoGPT2, 변경 없음).
    모델이 없으면 샘플 질문으로 폴백한다.
    """

    def __init__(self, gen_model, gen_tokenizer):
        self._model = gen_model
        self._tokenizer = gen_tokenizer

    def ask(self, competency: str) -> AgentResult[str]:
        question = models.generate_question(self._model, self._tokenizer, competency)
        tier = "light" if self._model is not None else "rule"
        return AgentResult(data=question, tier_used=tier, fallback_used=self._model is None)
