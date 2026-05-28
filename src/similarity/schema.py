# STEP 3 - 비교 결과 데이터 구조 정의
# 모든 유사도 방법이 이 스키마를 출력해야 다음 단계(rewriter)에서 통일된 형식으로 받을 수 있다

from dataclasses import dataclass, field


@dataclass
class SectionScore:
    """채용공고 섹션별 유사도 점수."""
    주요업무: float = 0.0
    자격요건: float = 0.0
    우대사항: float = 0.0

    def dominant(self) -> str:
        """가장 관련도가 높은 섹션을 반환한다."""
        scores = {
            '주요업무': self.주요업무,
            '자격요건': self.자격요건,
            '우대사항': self.우대사항,
        }
        return max(scores, key=scores.get)

    def to_dict(self) -> dict:
        return {'주요업무': self.주요업무, '자격요건': self.자격요건, '우대사항': self.우대사항}


@dataclass
class ComparisonResult:
    """
    단일 문단에 대한 채용공고 부합도 분석 결과.
    이 객체가 scorer, rewriter로 전달되어 피드백과 재구성의 입력이 된다.
    """
    paragraph_idx: int
    paragraph_text: str

    # 전체 유사도 (4가지 방법 가중 평균)
    overall_score: float

    # 섹션별 유사도
    section_scores: SectionScore

    # 키워드 분석
    covered_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    # 주요업무 섹션 기준으로 정렬된 누락 키워드 우선순위
    priority_keywords: list[str] = field(default_factory=list)
    coverage_ratio: float = 0.0

    # 수정 긴급도
    fix_urgency: str = 'low'     # 'high' | 'medium' | 'low'
    needs_rewrite: bool = False

    # 방법별 개별 점수 (투명성 확보 및 STEP 3 평가용)
    jaccard_score: float = 0.0
    tfidf_score: float = 0.0
    bm25_score: float = 0.0
    sbert_score: float = 0.0

    def score_variance(self) -> float:
        """4가지 방법 점수의 분산 — 낮을수록 방법 간 일관성이 높음."""
        scores = [self.jaccard_score, self.tfidf_score, self.bm25_score, self.sbert_score]
        mean = sum(scores) / len(scores)
        return round(sum((s - mean) ** 2 for s in scores) / len(scores), 4)
