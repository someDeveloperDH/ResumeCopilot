# STEP 3 - KoSBERT 문장 임베딩 유사도
# 문장 전체의 의미를 벡터로 변환해 코사인 유사도를 계산한다.
# "데이터 구축"과 "파이프라인 개발"처럼 표현은 다르지만 의미가 가까운 경우를 잡아냄

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 한국어 NLI + STS 데이터로 학습된 KoSBERT — 한국어 문장 유사도에 최적화
MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'


class SBERTSimilarity:
    def __init__(self):
        # 모델은 한 번만 로드하고 이후 재사용 (GPU 메모리 절약)
        self._model = SentenceTransformer(MODEL_NAME)
        self._job_vec = None

    def encode_job(self, job_text: str) -> None:
        """채용공고 전체를 임베딩해 저장한다."""
        self._job_vec = self._model.encode([job_text])

    def batch_score(self, paragraphs: list[str]) -> list[float]:
        """
        자소서 전체 문단을 한 번에 배치 추론한다.
        문단마다 개별 추론하면 GPU 활용률이 낮아지므로
        전체를 모아 한 번에 처리해 GPU 효율을 극대화한다.
        """
        if self._job_vec is None:
            raise RuntimeError("encode_job()을 먼저 호출해야 합니다.")

        para_vecs = self._model.encode(
            paragraphs,
            batch_size=16,
            show_progress_bar=False,
        )
        scores = cosine_similarity(self._job_vec, para_vecs)[0]
        return scores.tolist()
