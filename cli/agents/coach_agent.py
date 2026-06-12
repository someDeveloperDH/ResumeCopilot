"""CoachAgent — STAR 추출/재작성 (smart 티어: Ollama 한국어 LLM, 폴백: rule)."""

from __future__ import annotations

from dataclasses import dataclass

import models
from coach import (
    CoachAnalysis,
    build_extraction_prompt,
    build_rewrite_prompt,
    classify_input,
    parse_analysis_json,
    rule_analyze,
    split_rewrite_response,
    target_chars_for,
)

from .base import AgentResult


@dataclass
class RewriteResult:
    text: str
    improvement_summary: list[str]


class CoachAgent:
    """STAR 추출(analyze)과 재작성(rewrite)을 담당한다.

    고정 티어: smart (Ollama 한국어 파인튜닝 LLM). 실패 시 rule 기반(coach.py)으로 폴백한다.
    """

    def __init__(self, smart_cfg: dict, cfg: dict):
        self._smart_cfg = smart_cfg
        self._cfg = cfg

    def analyze(self, question: str, competency: str, answer: str) -> AgentResult[CoachAnalysis]:
        fallback = rule_analyze(question, answer, self._cfg)
        prompt = build_extraction_prompt(question, competency, answer, classify_input(answer, self._cfg))
        try:
            raw = models.ollama_chat(prompt, self._smart_cfg, json_format=True)
        except RuntimeError as exc:
            if not self._smart_cfg.get("allow_rule_fallback", True):
                raise
            print(f"\n  [안내] 스마트 모델 분석 실패 → rule 기반으로 진행합니다. ({exc})")
            return AgentResult(data=fallback, tier_used="rule", fallback_used=True)
        return AgentResult(data=parse_analysis_json(raw, fallback), tier_used="smart")

    def rewrite(
        self,
        question: str,
        competency: str,
        original: str,
        analysis: CoachAnalysis,
        followup: str,
    ) -> AgentResult[RewriteResult]:
        target_chars = target_chars_for(analysis.input_type, self._cfg)
        prompt = build_rewrite_prompt(question, competency, original, analysis, followup, target_chars)
        try:
            raw = models.ollama_chat(prompt, self._smart_cfg, json_format=False)
        except RuntimeError as exc:
            if not self._smart_cfg.get("allow_rule_fallback", True):
                raise
            print(f"\n  [안내] 스마트 모델 재작성 실패 → rule 기반 문단으로 진행합니다. ({exc})")
            text = _rule_rewrite(original, analysis, followup)
            return AgentResult(
                data=RewriteResult(text, analysis.improvement_summary),
                tier_used="rule",
                fallback_used=True,
            )
        text, summary = split_rewrite_response(raw, analysis.improvement_summary)
        return AgentResult(data=RewriteResult(text, summary), tier_used="smart")


def _rule_rewrite(original: str, analysis: CoachAnalysis, followup: str) -> str:
    if analysis.input_type == "seed":
        return (
            f"{analysis.core_experience} 경험을 자소서 소재로 저장했습니다. "
            "추후 상황, 행동, 결과를 보강해 완성 문단으로 확장할 수 있습니다."
        )

    parts = []
    if analysis.situation:
        parts.append(analysis.situation)
    if analysis.task:
        parts.append(f"이 과정에서 저는 {analysis.task}")
    if analysis.action:
        parts.append(f"{analysis.action}을 중심으로 문제를 해결했습니다.")
    if followup:
        parts.append(f"추가로 {followup}")
    if analysis.result:
        parts.append(f"그 결과 {analysis.result}라는 성과를 만들었습니다.")
    if not parts:
        parts.append(original)
    return " ".join(parts)
