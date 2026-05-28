# STEP 3 - TF-IDF + Cosine 유사도
# 전체 텍스트를 벡터로 변환해 코사인 유사도를 계산한다.
# 어휘 기반이라 동의어에 약하지만 전체적인 주제 유사도를 빠르게 측정할 수 있음

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSimilarity:
    def __init__(self):
        # 한국어는 형태소 분석 후 공백으로 구분된 토큰 문자열을 받으므로
        # 기본 토크나이저(공백 분리)를 그대로 사용
        self._vectorizer = TfidfVectorizer()
        self._job_vec = None

    def fit(self, job_text: str) -> None:
        """채용공고 텍스트로 TF-IDF 벡터라이저를 학습시킨다."""
        self._vectorizer.fit([job_text])
        self._job_vec = self._vectorizer.transform([job_text])

    def score(self, paragraph: str) -> float:
        """문단과 채용공고의 코사인 유사도를 반환한다."""
        if self._job_vec is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        para_vec = self._vectorizer.transform([paragraph])
        return float(cosine_similarity(self._job_vec, para_vec)[0][0])
