# JobFit — 프로젝트 기술 문서

> 채용공고 기반 자기소개서 자동 최적화 CLI 도구  
> 작성일: 2026-05-13

---

## 전체 파이프라인 개요

```
[사용자 입력]
  마스터 자소서 파일 (.txt / .pdf)
  채용공고 URL 또는 파일 (.txt)
          ↓
  STEP 0  입력 수집          fetcher.py / parser.py
          ↓
  STEP 1  키워드 추출         rule_based.py / ai_based.py / comparator.py
          ↓
  STEP 2  문단 분리           tokenizer.py
          ↓
  STEP 3  유사도 분석 (병렬)  jaccard / tfidf / bm25 / sbert → aggregator.py
          ↓
  STEP 4  피드백 & LLM 재구성 scorer.py / rewriter.py (EXAONE-3.5-7.8B)
          ↓
  STEP 5  CLI 대화형 루프     session.py (y/n/r/s)
          ↓
  STEP 6  최종 저장           session.py
          ↓
  STEP 7  AI 기여도 분석      edit_distance / style_analyzer / ai_reporter
          ↓
[출력]
  공고 맞춤 수정 자소서 (.txt) + AI 기여도 리포트
```

---

## STEP 0 — 입력 수집

### 해결 방식

채용공고와 자소서를 다양한 형태(URL, txt, pdf)에서 통일된 텍스트로 변환한다.  
URL 입력 시 `requests` + `BeautifulSoup`으로 HTML을 파싱하고, `[주요업무]` `[자격요건]` `[우대사항]` 패턴을 정규식으로 탐지해 섹션을 분리한다.

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/input/fetcher.py` | URL → 텍스트 + 섹션 딕셔너리 |
| `src/input/parser.py` | txt / pdf → 텍스트 |

### Input

```
채용공고: URL  예) https://saramin.co.kr/...
          또는 파일 예) data/sample_job.txt

자소서:   파일 예) data/sample_cover.txt
```

### Output

```python
# 채용공고 (URL 입력 시)
{
    'raw_text': '전체 텍스트...',
    'sections': {
        '주요업무': '대용량 데이터 파이프라인 설계 및 구축...',
        '자격요건': 'Python 프로그래밍 능숙자...',
        '우대사항': 'Apache Spark, Hadoop...'
    },
    'eval': {
        'fetch_success': True,
        'text_length': 842,
        'section_detected': {'주요업무': True, '자격요건': True, '우대사항': True}
    }
}

# 자소서 (파일 입력 시)
{
    'text': '데이터가 세상을 바꾼다는 말을...',
    'eval': {
        'fetch_success': True,
        'text_length': 1204,
        'likely_valid': True
    }
}
```

### 평가 지표

| 지표 | 정상 기준 |
|---|---|
| `fetch_success` | True |
| `text_length` | 200자 이상 |
| `section_detected` | 1개 이상 True |

---

## STEP 1 — 채용공고 키워드 추출

### 해결 방식

두 가지 방법으로 키워드를 추출한 뒤 합산하고 overlap 비율로 신뢰도를 평가한다.

**방법 A — Rule-based**  
`kiwipiepy`로 형태소 분석 후 품사 필터링.  
추출 대상: `NNG`(일반명사), `NNP`(고유명사), `SL`(외래어), `VV`(동사), `VA`(형용사)

**방법 B — AI-based**  
`KeyBERT` + `KoSBERT` 임베딩으로 텍스트 내 중요도 높은 키워드를 점수 순으로 추출.  
1-gram + 2-gram 모두 허용 (예: `데이터 파이프라인`)

**통합**  
두 방법의 합집합을 최종 키워드로 사용. overlap 비율로 두 방법의 일관성 평가.

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/preprocessing/cleaner.py` | 특수문자 제거, 정규화 |
| `src/preprocessing/tokenizer.py` | kiwipiepy 형태소 분석 |
| `src/keyword/rule_based.py` | POS 필터 방식 |
| `src/keyword/ai_based.py` | KeyBERT 방식 |
| `src/keyword/comparator.py` | 두 결과 비교 및 합산 |

### Input

```python
# STEP 0 output의 sections
{
    '주요업무': '대용량 데이터 파이프라인 설계 및 구축 SQL ETL...',
    '자격요건': 'Python 프로그래밍 능숙자 SQL 중급...',
    '우대사항': 'Apache Spark Hadoop AWS GCP...'
}
```

### Output

```python
{
    'merged': {
        '주요업무': ['파이프라인', 'ETL', 'SQL', '데이터품질', '모니터링'],
        '자격요건': ['Python', 'SQL', '데이터처리', '문제해결'],
        '우대사항': ['Spark', 'AWS', '클라우드', '협업']
    },
    'all_keywords': ['파이프라인', 'ETL', 'SQL', 'Python', 'Spark', ...],  # 전체 18개
    'eval': {
        'per_section': {
            '주요업무': {'n_rule': 8, 'n_ai': 7, 'n_overlap': 5, 'overlap_ratio': 0.625}
        },
        'total_keywords': 18,
        'overall_overlap_ratio': 0.61,
        'consistent': True   # 0.4 이상이면 두 방법 일관
    }
}
```

### 평가 지표

| 지표 | 정상 기준 |
|---|---|
| `total_keywords` | 10개 이상 |
| `overall_overlap_ratio` | 0.4 이상 (두 방법 일관) |
| `consistent` | True |

---

## STEP 2 — 자소서 문단 분리

### 해결 방식

자소서 텍스트를 처리 단위인 문단으로 분리한다.  
입력 형태에 따라 두 방식 중 하나를 자동 선택하며, `--split-mode` 옵션으로 강제 지정도 가능.

**방식 A — regex (기본)**  
`\n{2,}` 패턴으로 빈 줄 2개 이상 기준 분리. 빠르고 단순.

**방식 B — NLP (고도화)**  
`kss`로 문장 단위 분리 → `KoSBERT`로 각 문장 임베딩 → 연속 문장 간 코사인 유사도 계산 → 유사도 0.65 미만이면 문단 경계로 판정.  
빈 줄 없는 자소서나 연속 텍스트에 효과적.

**자동 선택 (auto)**  
`\n{2,}` 패턴이 있으면 regex, 없으면 NLP 방식 자동 적용.

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/preprocessing/tokenizer.py` | `split_paragraphs()` / `split_paragraphs_nlp()` / `auto_split()` |

### Input

```python
# STEP 0의 자소서 텍스트 (cleaner 전처리 후)
"데이터가 세상을 바꾼다는 말을 처음 들었을 때...\n\n대학원 연구실에서 2년간..."
```

### Output

```python
# 문단 리스트
[
    "데이터가 세상을 바꾼다는 말을 처음 들었을 때, 저는 단순히...",
    "대학원 연구실에서 2년간 연구 데이터 관리 시스템을 담당하며...",
    "학부 시절 교내 동아리에서 웹 서비스 개발 프로젝트를...",
    "데이터브릿지에 합류한다면, 처음에는..."
]

# 평가 지표
{
    'split_method': 'regex',    # 사용된 방식
    'n_paragraphs': 4,
    'avg_length': 230.5,
    'short_paragraphs': [],     # 30자 미만 인덱스
    'long_paragraphs': []       # 500자 초과 인덱스
}
```

### 평가 지표

| 지표 | 정상 기준 |
|---|---|
| `n_paragraphs` | 1개 이상 |
| `avg_length` | 50자 이상 |
| `short_paragraphs` | 비어있으면 정상 |

---

## STEP 3 — 채용공고 ↔ 자소서 유사도 분석

### 해결 방식

4가지 NLP 방법을 **병렬**로 실행해 각 문단의 공고 부합도를 다각도로 측정한다.  
KoSBERT만 시작 시 **전체 배치 추론** 후 캐시에 저장, 나머지 3개는 **ThreadPoolExecutor** 병렬 실행.

| 방법 | 담당 | 기술 |
|---|---|---|
| **Jaccard** | 키워드 커버리지 (covered/missing) | 집합 교집합 / 합집합 |
| **TF-IDF + Cosine** | 전체 텍스트 유사도 (베이스라인) | sklearn TfidfVectorizer |
| **BM25** | 섹션별 관련도 (주요업무/자격요건/우대사항) | rank_bm25 |
| **KoSBERT** | 의미 기반 유사도 (동의어·문맥 반영) | sentence-transformers |

**가중치**: Jaccard 0.10 / TF-IDF 0.25 / BM25 0.30 / KoSBERT 0.35  
→ 의미를 더 잘 반영할수록 높은 가중치 부여

**실행 구조**:
```
시작 시  →  KoSBERT 전체 문단 배치 추론 (GPU)  →  캐시 저장
문단별   →  Jaccard ─┐
             TF-IDF  ─┤  ThreadPoolExecutor (CPU 3개 병렬)
             BM25    ─┘
             KoSBERT →  캐시 즉시 조회
                 ↓
           가중 평균 → ComparisonResult
```

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/similarity/schema.py` | `ComparisonResult` 데이터 구조 정의 |
| `src/similarity/jaccard.py` | 키워드 커버리지 계산 |
| `src/similarity/tfidf.py` | TF-IDF 코사인 유사도 |
| `src/similarity/bm25.py` | 섹션별 BM25 점수 |
| `src/similarity/sbert.py` | KoSBERT 배치 추론 |
| `src/similarity/aggregator.py` | 4개 방법 병렬 집계 + ComparisonResult 생성 |

### Input

```python
# STEP 1 output
job_sections  = {'주요업무': [...], '자격요건': [...], '우대사항': [...]}
all_keywords  = ['파이프라인', 'ETL', 'SQL', 'Python', ...]

# STEP 2 output
paragraphs = ["문단0 텍스트...", "문단1 텍스트...", ...]
```

### Output

```python
# 문단별 ComparisonResult 리스트
[
    ComparisonResult(
        paragraph_idx     = 0,
        paragraph_text    = "데이터가 세상을 바꾼다는...",

        # 전체 유사도 (4가지 가중 평균)
        overall_score     = 0.31,

        # 섹션별 관련도 (BM25 기반)
        section_scores    = SectionScore(주요업무=0.20, 자격요건=0.35, 우대사항=0.10),

        # 키워드 분석 (Jaccard 기반)
        covered_keywords  = ['데이터', '분석'],
        missing_keywords  = ['파이프라인', 'SQL', 'ETL', 'Python'],
        priority_keywords = ['파이프라인', 'ETL'],   # 주요업무 기준 우선순위
        coverage_ratio    = 0.29,

        # 수정 판단
        fix_urgency       = 'high',    # overall < 0.3
        needs_rewrite     = True,      # overall < 0.5 (threshold)

        # 방법별 개별 점수 (투명성)
        jaccard_score     = 0.29,
        tfidf_score       = 0.33,
        bm25_score        = 0.31,
        sbert_score       = 0.34,
    ),
    ...
]
```

### 평가 지표

| 지표 | 정상 기준 |
|---|---|
| `avg_overall_score` | 높을수록 공고-자소서 부합 |
| `score_variance` (4개 방법 분산) | 낮을수록 방법 간 일관성 높음 |
| `needs_rewrite_count` | 수정 필요 문단 수 |

---

## STEP 4 — 피드백 생성 & STAR 재구성

### 해결 방식

ComparisonResult의 수치를 사람이 읽는 피드백으로 변환하고,  
로컬 LLM(EXAONE-3.5-7.8B)을 통해 STAR 구조로 자소서 문단을 재작성한다.

**피드백 생성 (scorer.py)**  
점수 → 별 5개 시각화, 누락 키워드 목록, 관련 섹션 표시

**STAR 재구성 (rewriter.py)**  
- 모델: `EXAONE-3.5-7.8B` (LG AI Research, 한국어 특화)
- 실행: Ollama 로컬 서버 (localhost:11434), 완전 오프라인
- GPU: RTX 5070 Ti에서 4-bit 양자화로 ~5GB VRAM 사용
- 자동 준비: 첫 호출 시 Ollama 설치 → 서버 시작 → 모델 다운로드 자동 처리
- 프롬프트: 원문 + 누락 키워드 + 관련 섹션 + STAR 규칙 명시

**재구성 평가**  
재작성 전후 키워드 커버리지 변화 및 STAR 구조 충족 여부 검증.

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/feedback/scorer.py` | 수치 → 피드백 텍스트 변환, STAR 충족 평가 |
| `src/feedback/rewriter.py` | EXAONE 로컬 LLM STAR 재구성 |
| `src/setup/ollama_manager.py` | Ollama 자동 설치/서버 시작/모델 다운로드 |

### Input

```python
# STEP 3 output의 단일 ComparisonResult
ComparisonResult(
    paragraph_text    = "학부 시절 교내 동아리에서 웹 서비스...",
    overall_score     = 0.08,
    priority_keywords = ['파이프라인', 'SQL', 'ETL'],
    section_scores    = SectionScore(주요업무=0.05, ...),
    ...
)

# 선택적
user_request = "ETL 경험을 더 구체적으로 강조해줘"  # r 옵션 입력 시
```

### Output

```python
# 재작성된 문단 (str)
"""
연구 데이터가 수작업으로 처리되어 비효율이 발생하던 환경에서(S),
실험 데이터 자동화 시스템 구축을 담당하게 되었습니다(T).
Python 기반 ETL 파이프라인을 설계하고 SQL 쿼리를 최적화하여
데이터 수집부터 정제까지 자동화하였으며(A),
기존 수동 처리 대비 처리 시간을 70% 단축하는 성과를 달성했습니다(R).
"""

# 재구성 평가
{
    'star_fulfilled': {'S': True, 'T': True, 'A': True, 'R': True},
    'all_star_met': True,
    'keywords_injected': ['파이프라인', 'ETL', 'SQL'],
    'n_injected': 3
}
```

### 평가 지표

| 지표 | 정상 기준 |
|---|---|
| `score_delta` | +0.1 이상 |
| `all_star_met` | True |
| `n_injected` | 1 이상 |

---

## STEP 5 — CLI 대화형 루프

### 해결 방식

각 문단을 순서대로 처리하며 사용자가 4가지 명령으로 수정 여부를 결정한다.  
`y` 즉시 수락 / `n` 재생성 / `r` 요구사항 반영 재생성 / `s` 원문 유지

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/cli/display.py` | Rich 기반 화면 출력 (패널, 표, 진행 상황) |
| `src/cli/session.py` | y/n/r/s 루프 처리, 세션 로그 기록 |

### Input

```python
paragraphs = ["문단0...", "문단1...", ...]       # STEP 2 output
results    = [ComparisonResult(...), ...]         # STEP 3 output
```

### CLI 화면 예시

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[문단 3/4]  공고 부합도: ★☆☆☆☆ (0.08)  [🔴 HIGH]
누락 키워드: 파이프라인, SQL, ETL, Python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[원문]  학부 시절 교내 동아리에서 웹 서비스...
[제안]  데이터 처리 환경에서(S) ETL 파이프라인을...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
> [y] 적용  [n] 재생성  [r] 요구사항 입력  [s] 건너뜀
```

### Output

```python
{
    'final_paragraphs': ["수정된 문단0...", "원문 유지 문단1...", ...],
    'session_log': [
        {'idx': 0, 'choice': 'y', 'score_before': 0.31, 'score_after': 0.51},
        {'idx': 1, 'choice': 's', 'score_before': 0.74, 'score_after': 0.74},
        {'idx': 2, 'choice': 'r', 'score_before': 0.08, 'score_after': 0.28},
        {'idx': 3, 'choice': 'y', 'score_before': 0.45, 'score_after': 0.65},
    ]
}
```

---

## STEP 6 — 최종 저장

### 해결 방식

모든 문단 처리 완료 후 요약 화면을 출력하고 사용자가 저장 여부를 결정한다.  
`w` 저장 / `p` 전체 미리보기 / `d` 취소

### Output

```
# 저장 파일 (output_cover.txt)
수정된 문단0...

수정된 문단1(원문)...

수정된 문단2...

수정된 문단3...
```

---

## STEP 7 — AI 기여도 분석

### 해결 방식

최종 자소서에서 AI가 실제로 얼마나 개입했는지를 3가지 관점으로 측정한다.

**관점 1 — 텍스트 변화율 (가중치 40%)**  
`python-Levenshtein`으로 원문과 최종본의 편집 거리를 계산해 변화율로 정규화.

```
변화율 = 편집 거리 / max(원문 길이, 최종 길이)
0~1%: 원문 유지  /  40~79%: 대폭 수정  /  80%+: 전면 재작성
```

**관점 2 — 세션 행동 로그 (가중치 35%)**  
STEP 5에서 축적된 y/n/r/s 선택 기록을 AI 기여도 점수로 변환.

```
y(즉시 수락) → 0.9   n(재생성 수락) → 0.7
r(요구 반영) → 0.4   s(원문 유지)   → 0.0
```

**관점 3 — 언어 스타일 변화 (가중치 25%)**  
`KoGPT2`로 Perplexity 측정 (낮을수록 AI 특성), TTR 어휘 다양성, 평균 문장 길이 변화.

**최종 AI 기여도**  
3가지 관점의 가중 평균을 문단 길이 비율로 다시 가중 평균.

### 관련 파일

| 파일 | 역할 |
|---|---|
| `src/analysis/edit_distance.py` | Levenshtein 변화율 계산 |
| `src/analysis/style_analyzer.py` | Perplexity, TTR, 문장 길이 분석 |
| `src/analysis/ai_reporter.py` | 3가지 관점 통합, Rich 리포트 출력 |

### Input

```python
originals    = ["원문 문단0...", "원문 문단1...", ...]  # STEP 2 output
finals       = ["최종 문단0...", "최종 문단1...", ...]  # STEP 6 output
session_log  = [{'idx':0,'choice':'y',...}, ...]        # STEP 5 output
```

### Output

```
🤖 AI 작성 기여도 분석 리포트

문단   변화율   변화수준      선택   AI기여도   판정
1      38%     부분 수정     y      42%       혼합 작성
2       0%     원문 유지     s       0%       자필 중심
3      91%     전면 재작성   y      87%       AI 작성
4      44%     부분 수정     r      31%       혼합 작성

전체 AI 기여도: 42%  →  혼합 작성 (사용자 + AI)
```

```python
{
    'per_paragraph': [
        {
            'paragraph_idx':      0,
            'edit_ratio':         0.38,
            'change_level':       '부분 수정',
            'session_choice':     'y',
            'ai_contribution_pct': 42.0,
            'verdict':            '혼합 작성 (사용자 + AI)',
        },
        ...
    ],
    'overall_ai_contribution': 42.0,
    'verdict': '혼합 작성 (사용자 + AI)',
}
```

### 판정 기준

| AI 기여도 | 판정 |
|---|---|
| 0 ~ 20% | 자필 중심 (AI 보조) |
| 21 ~ 50% | 혼합 작성 (사용자 + AI) |
| 51 ~ 80% | AI 주도 (사용자 검토) |
| 81 ~ 100% | AI 작성 (사용자 확인만) |

---

## 프로젝트 구조

```
test1/
├── main.py                          # CLI 진입점, STEP 0~7 오케스트레이션
├── requirements.txt
├── presentation.md
├── project_doc.md                   # 이 문서
├── data/
│   ├── sample_job.txt               # 채용공고 테스트 데이터
│   └── sample_cover.txt             # 마스터 자소서 테스트 데이터
├── src/
│   ├── input/
│   │   ├── fetcher.py               # STEP 0 — URL 스크래핑
│   │   └── parser.py                # STEP 0 — 파일 파싱
│   ├── preprocessing/
│   │   ├── cleaner.py               # STEP 1,2 — 텍스트 정제
│   │   └── tokenizer.py             # STEP 1,2 — 형태소 분석, 문단 분리
│   ├── keyword/
│   │   ├── rule_based.py            # STEP 1 — POS 필터 키워드 추출
│   │   ├── ai_based.py              # STEP 1 — KeyBERT 키워드 추출
│   │   └── comparator.py            # STEP 1 — 두 방법 비교 및 통합
│   ├── similarity/
│   │   ├── schema.py                # STEP 3 — ComparisonResult 구조
│   │   ├── jaccard.py               # STEP 3 — 키워드 커버리지
│   │   ├── tfidf.py                 # STEP 3 — TF-IDF 코사인 유사도
│   │   ├── bm25.py                  # STEP 3 — BM25 섹션별 관련도
│   │   ├── sbert.py                 # STEP 3 — KoSBERT 의미 유사도
│   │   └── aggregator.py            # STEP 3 — 병렬 집계
│   ├── feedback/
│   │   ├── scorer.py                # STEP 4 — 피드백 텍스트 생성
│   │   └── rewriter.py              # STEP 4 — EXAONE LLM STAR 재구성
│   ├── cli/
│   │   ├── display.py               # STEP 5,6 — Rich 화면 출력
│   │   └── session.py               # STEP 5,6 — 대화형 루프, 저장
│   ├── analysis/
│   │   ├── edit_distance.py         # STEP 7 — Levenshtein 변화율
│   │   ├── style_analyzer.py        # STEP 7 — Perplexity, TTR
│   │   └── ai_reporter.py           # STEP 7 — AI 기여도 통합 리포트
│   └── setup/
│       └── ollama_manager.py        # Ollama 자동 설치/서버/모델 관리
└── logs/
    └── changelog.md
```

---

## 기술 스택

| 분류 | 기술 | 용도 |
|---|---|---|
| 웹 수집 | requests, beautifulsoup4 | URL 채용공고 스크래핑 |
| 문서 파싱 | pdfminer.six | PDF 자소서 파싱 |
| 형태소 분석 | kiwipiepy | 한국어 형태소 분석 (Java 불필요) |
| 문장 분리 | kss | 한국어 문장 경계 탐지 |
| 키워드 추출 | KeyBERT | AI 기반 중요 키워드 추출 |
| 유사도 | scikit-learn | TF-IDF + Cosine |
| 유사도 | rank-bm25 | BM25 검색 관련도 |
| 임베딩 | sentence-transformers | KoSBERT 문장 임베딩 |
| 로컬 LLM | Ollama + EXAONE-3.5-7.8B | STAR 재구성 (오프라인, 무료) |
| AI 분석 | transformers + KoGPT2 | Perplexity 측정 |
| 편집 거리 | python-Levenshtein | 텍스트 변화율 계산 |
| CLI | click, rich | 인터페이스 및 시각화 |

---

## 실행 방법

```bash
# 환경 활성화
conda activate jobfit

# 기본 실행 (파일 입력)
python main.py --cover data/sample_cover.txt --job data/sample_job.txt

# URL 입력
python main.py --cover data/sample_cover.txt --job https://...

# 문단 분리 방식 지정
python main.py --cover data/sample_cover.txt --job data/sample_job.txt --split-mode nlp

# 출력 파일 지정
python main.py --cover data/sample_cover.txt --job data/sample_job.txt --output result.txt
```
