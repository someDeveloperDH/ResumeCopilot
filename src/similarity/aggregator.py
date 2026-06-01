# STEP 3 - 병렬 유사도 집계 및 ComparisonResult 생성
# Jaccard/TF-IDF/BM25는 CPU에서 병렬 실행, KoSBERT는 사전 배치 추론 후 캐시 조회

from concurrent.futures import ThreadPoolExecutor

from src.similarity.schema import ComparisonResult, SectionScore
from src.similarity import jaccard, tfidf as tfidf_mod, bm25 as bm25_mod, sbert as sbert_mod
from src.preprocessing.tokenizer import morphs

# 방법별 가중치 — 의미를 더 잘 반영할수록 높게 설정
WEIGHTS = {
    'jaccard': 0.10,
    'tfidf':   0.25,
    'bm25':    0.30,
    'sbert':   0.35,
}

# 이 값 미만이면 재작성 필요로 판단
REWRITE_THRESHOLD = 0.5


class Aggregator:
    def __init__(self, job_sections: dict, all_keywords: list[str]):
        """
        채용공고 섹션 딕셔너리와 전체 키워드 리스트로 초기화한다.
        모델/인덱스를 여기서 한 번만 생성해 문단마다 재생성하는 비용을 없앤다.
        """
        self.all_keywords = all_keywords
        self._sbert_cache: dict[int, float] = {}

        # 각 방법의 입력 형태가 달라 따로 준비
        job_full_text = ' '.join(job_sections.values())
        section_tokens = {s: morphs(t) for s, t in job_sections.items() if t}

        self._tfidf = tfidf_mod.TFIDFSimilarity()
        self._tfidf.fit(job_full_text)

        self._bm25 = bm25_mod.BM25Similarity(section_tokens)

        self._sbert = sbert_mod.SBERTSimilarity()
        self._sbert.encode_job(job_full_text)

    def precompute_sbert(self, paragraphs: list[str]) -> None:
        """
        자소서 전체 문단을 한 번에 KoSBERT로 추론해 캐시에 저장한다.
        문단 루프 전에 한 번만 호출하면 이후 GPU 대기 없이 캐시 조회로 처리된다.
        """
        scores = self._sbert.batch_score(paragraphs)
        self._sbert_cache = {i: s for i, s in enumerate(scores)}

    def compare(self, paragraph: str, idx: int) -> ComparisonResult:
        """
        단일 문단에 대해 4가지 유사도를 측정하고 ComparisonResult를 반환한다.
        Jaccard/TF-IDF/BM25는 ThreadPoolExecutor로 병렬 실행한다.
        """
        para_tokens = morphs(paragraph)

        # CPU 3개 병렬 실행
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_j = executor.submit(jaccard.score, self.all_keywords, para_tokens)
            f_t = executor.submit(self._tfidf.score, paragraph)
            f_b = executor.submit(self._bm25.score, para_tokens)

        j = f_j.result()
        t = f_t.result()
        b_scores = f_b.result()  # {주요업무: float, 자격요건: float, 우대사항: float}
        s = self._sbert_cache.get(idx, 0.0)

        # BM25 점수 정규화 (섹션별 절대값이 달라 비율로 변환)
        bm25_max = max(b_scores.values()) if b_scores else 1.0
        section_scores = SectionScore(
            주요업무=round(b_scores.get('주요업무', 0.0) / (bm25_max + 1e-9), 3),
            자격요건=round(b_scores.get('자격요건', 0.0) / (bm25_max + 1e-9), 3),
            우대사항=round(b_scores.get('우대사항', 0.0) / (bm25_max + 1e-9), 3),
        )

        # BM25 평균 (overall 계산용)
        # bm25_avg = sum(b_scores.values()) / len(b_scores) if b_scores else 0.0

        if b_scores:
            bm25_min = min(b_scores.values())
            bm25_max = max(b_scores.values())

            if abs(bm25_max - bm25_min) < 1e-9:
                bm25_norm = 0.0
            else:
                vals = [
                    (v - bm25_min) / (bm25_max - bm25_min)
                    for v in b_scores.values()
                ]
                bm25_norm = sum(vals) / len(vals)
        else:
            bm25_norm = 0.0

        # overall = (
        #     WEIGHTS['jaccard'] * j
        #     + WEIGHTS['tfidf'] * t
        #     + WEIGHTS['bm25'] * bm25_avg
        #     + WEIGHTS['sbert'] * s
        # )
        overall = (
                WEIGHTS['jaccard'] * j
                + WEIGHTS['tfidf'] * t
                + WEIGHTS['bm25'] * bm25_norm
                + WEIGHTS['sbert'] * s
        )

        # 키워드 커버리지
        cov = jaccard.coverage(self.all_keywords, para_tokens)

        # 누락 키워드 중 주요업무 섹션 키워드를 앞에 배치 (우선순위)
        priority = cov['missing']

        urgency = 'high' if overall < 0.3 else ('medium' if overall < 0.5 else 'low')

        return ComparisonResult(
            paragraph_idx=idx,
            paragraph_text=paragraph,
            overall_score=round(overall, 3),
            section_scores=section_scores,
            covered_keywords=cov['covered'],
            missing_keywords=cov['missing'],
            priority_keywords=priority,
            coverage_ratio=cov['ratio'],
            fix_urgency=urgency,
            needs_rewrite=overall < REWRITE_THRESHOLD,
            jaccard_score=round(j, 3),
            tfidf_score=round(t, 3),
            bm25_score=round(bm25_norm, 3),
            sbert_score=round(s, 3),
        )

    def evaluate_all(self, results: list[ComparisonResult]) -> dict:
        """STEP 3 전체 평가 지표를 산출한다."""
        scores = [r.overall_score for r in results]
        return {
            'n_paragraphs': len(results),
            'avg_overall_score': round(sum(scores) / len(scores), 3) if scores else 0,
            'needs_rewrite_count': sum(1 for r in results if r.needs_rewrite),
            'avg_score_variance': round(
                sum(r.score_variance() for r in results) / len(results), 4
            ) if results else 0,
            'per_paragraph': [
                {
                    'idx': r.paragraph_idx,
                    'overall': r.overall_score,
                    'jaccard': r.jaccard_score,
                    'tfidf': r.tfidf_score,
                    'bm25': r.bm25_score,
                    'sbert': r.sbert_score,
                    'urgency': r.fix_urgency,
                }
                for r in results
            ],
        }
