"""InterviewerAgent — 꼬리질문 생성 (smart 티어: Ollama 한국어 LLM, 폴백: light → rule)."""

from __future__ import annotations

from dataclasses import dataclass

import models
from coach import build_tail_question_prompt, clean_tail_question

from .base import AgentResult


@dataclass
class InterviewContext:
    """꼬리질문 생성에 필요한 입력 — 기존 generate_tail_question의 파라미터를 타입화."""

    question: str
    answer: str
    intent_score: float
    job: str
    conversation_context: str = ""
    consistency_note: str = ""


class InterviewerAgent:
    """전체 대화 맥락을 바탕으로 꼬리질문을 생성한다.

    고정 티어: smart (Ollama 한국어 파인튜닝 LLM).
    Ollama 호출 실패/빈 응답 시 light(기존 생성 모델) → rule(샘플 질문) 순으로 폴백한다.
    """

    def __init__(self, smart_cfg: dict, gen_model, gen_tokenizer):
        self._smart_cfg = smart_cfg
        self._gen_model = gen_model
        self._gen_tokenizer = gen_tokenizer

    def ask_followup(self, ctx: InterviewContext) -> AgentResult[str]:
        prompt = build_tail_question_prompt(ctx)
        try:
            raw = models.ollama_chat(prompt, self._smart_cfg, json_format=False)
            question = clean_tail_question(raw)
            if question:
                return AgentResult(data=question, tier_used="smart")
            reason = "빈 응답"
        except RuntimeError as exc:
            if not self._smart_cfg.get("allow_rule_fallback", True):
                raise
            reason = str(exc)

        print(f"\n  [안내] 스마트 모델 꼬리질문 생성 실패 → 경량 모델로 진행합니다. ({reason})")
        question = models.generate_tail_question(
            self._gen_model, self._gen_tokenizer,
            question=ctx.question, answer=ctx.answer,
            intent_score=ctx.intent_score, job=ctx.job,
            conversation_context=ctx.conversation_context,
            consistency_note=ctx.consistency_note,
        )
        tier = "light" if self._gen_model is not None else "rule"
        return AgentResult(data=question, tier_used=tier, fallback_used=True)
