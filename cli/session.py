"""세션 상태 관리 - 한 번의 Q&A 사이클 데이터

jinyoung_v2 Session(멀티턴 추적: job_history/consistency_checks/conversation_context)과
jintae_v2 Session(코칭 채점: input_type/score_before/after/retry_used/answer_card/versions)을
하나로 합본한 버전.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:6])
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    competency: str = ""
    question: str = ""
    answer: str = ""          # 최초 답변
    intent_score: float = 0.0
    suitable_jobs: list = field(default_factory=list)
    conversation: list = field(default_factory=list)  # 꼬리질문 대화 이력
    final_answer: str = ""

    # jinyoung_v2: 멀티턴 추적 필드
    job_history: list = field(default_factory=list)          # turn별 직무/점수 변화
    consistency_checks: list = field(default_factory=list)   # 일관성 검사 결과
    conversation_context: str = ""                           # 저장 시점 전체 대화 맥락

    # jintae_v2: 코칭/채점 필드
    input_type: str = ""
    score_before: float = 0.0
    score_after: float = 0.0
    retry_used: bool = False
    answer_card: dict = field(default_factory=dict)
    versions: list = field(default_factory=list)

    def add_turn(self, tail_question: str, user_response: str):
        self.conversation.append({
            "tail_question": tail_question,
            "response": user_response,
        })
        self.final_answer = user_response

    def add_job_history(self, turn: int, source: str, intent_score: float, job: str):
        self.job_history.append({
            "turn": turn,
            "source": source,
            "intent_score": round(float(intent_score), 1),
            "job": job,
        })

    def add_consistency_check(self, turn: int, result: dict):
        self.consistency_checks.append({
            "turn": turn,
            **result,
        })

    def set_answer_card(self, card: dict):
        self.answer_card = card
        self.versions = card.get("versions", [])
        self.final_answer = card.get("final_answer", self.final_answer)

    def to_dict(self) -> dict:
        return asdict(self)
