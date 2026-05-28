# STEP 3 - BM25 유사도 (섹션별 관련도 측정)
# 검색 엔진에서 사용하는 BM25 알고리즘으로 섹션별 관련도를 측정한다.
# TF-IDF와 달리 단어 빈도가 높아도 점수가 무한히 올라가지 않음 (포화 처리)

from rank_bm25 import BM25Okapi


class BM25Similarity:
    def __init__(self, section_tokens: dict):
        """
        섹션별 토큰 리스트로 BM25 인덱스를 구성한다.
        각 섹션을 독립적인 문서로 취급해 섹션별 관련도를 따로 계산할 수 있다.
        """
        # 섹션마다 단일 문서 코퍼스로 BM25 인덱스 생성
        self._indices = {
            section: BM25Okapi([tokens])
            for section, tokens in section_tokens.items()
            if tokens
        }

    def score(self, para_tokens: list[str]) -> dict:
        """
        문단 토큰으로 각 섹션과의 BM25 점수를 계산한다.
        반환값을 통해 이 문단이 주요업무/자격요건/우대사항 중
        어느 섹션과 가장 관련 있는지 판단할 수 있다.
        """
        scores = {}
        for section, bm25 in self._indices.items():
            raw = bm25.get_scores(para_tokens)
            # 단일 문서 코퍼스이므로 인덱스 0만 사용
            scores[section] = float(raw[0])
        return scores
