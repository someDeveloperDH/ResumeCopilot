"""세션 상태 관리 - 한 번의 Q&A 사이클 데이터"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:6])
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    competency: str = ""
    question: str = ""
    answer: str = ""          # 최종 답변 (꼬리질문 거친 후)
    intent_score: float = 0.0
    suitable_jobs: list = field(default_factory=list)
    conversation: list = field(default_factory=list)  # 꼬리질문 대화 이력
    final_answer: str = ""
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

    def set_answer_card(self, card: dict):
        self.answer_card = card
        self.versions = card.get("versions", [])
        self.final_answer = card.get("final_answer", self.final_answer)

    def to_dict(self) -> dict:
        return asdict(self)
