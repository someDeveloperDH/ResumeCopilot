"""cli/agents — 오케스트레이터(run_session)가 호출하는 서브에이전트 모음.

서브에이전트별 고정 모델 티어:
  - QuestionAgent    : light       (기존 생성 모델, 변경 없음)
  - InterviewerAgent : smart        (Ollama 한국어 파인튜닝 LLM)
  - CoachAgent       : smart        (Ollama 한국어 파인튜닝 LLM)
  - AnalysisAgent    : specialized   (ModernBERT/KoELECTRA, 변경 없음)
  - ConsistencyAgent : specialized   (NLI/SBERT, 변경 없음)

모든 메서드는 AgentResult[T](data, tier_used, fallback_used)를 반환한다.
"""

from .analysis_agent import AnalysisAgent, AnalysisOutput
from .base import AgentResult
from .coach_agent import CoachAgent, RewriteResult
from .consistency_agent import ConsistencyAgent
from .interviewer_agent import InterviewContext, InterviewerAgent
from .question_agent import QuestionAgent

__all__ = [
    "AgentResult",
    "AnalysisAgent",
    "AnalysisOutput",
    "CoachAgent",
    "RewriteResult",
    "ConsistencyAgent",
    "InterviewContext",
    "InterviewerAgent",
    "QuestionAgent",
]
