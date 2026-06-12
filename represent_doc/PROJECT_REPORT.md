# 하루 1문답 — 발표용 정리 자료

> 자연어처리를 활용한 직무(Job) 관련 프로젝트 — 평가 기준에 맞춘 발표 소스 자료
> 작성일: 2026-06-12 / 대상 시스템: `test4` (test3 + jintae_v2 + jinyoung_v2 통합본)

---

## 1. 제목

**하루 1문답 — Orchestrator-Subagent 기반 AI 자기소개서 코칭 CLI**

- 부제: *매일 질문 하나, AI 분석 · STAR 코칭 · 꼬리질문 인터뷰로 완성하는 나만의 자소서 소스*
- 발표 슬라이드 제목으로 그대로 사용 가능. 팀명/슬로건이 정해지면 부제만 교체.

---

## 2. 문제 배경

> (체크리스트에 기존 작성된 내용)

**"하루 1문답. AI 기반 자기소개서 훈련 시스템"**

IT/AI 직무를 준비하는 취업준비생(대학 1학년 ~ 장기 취준생)은 자기소개서를 작성할 때 다음과 같은 어려움을 겪는다.

- 빈 화면에서 자소서를 "한 번에 완성된 글"로 써야 한다는 부담 때문에 작성을 미루게 된다.
- 작성한 답변이 질문의 의도를 얼마나 충족하는지, 어떤 직무(백엔드/AI·ML/프로덕트)에 더 적합한 경험인지 스스로 판단하기 어렵다.
- 주변에 첨삭해 줄 멘토·현직자가 없으면 경험을 STAR(상황-과제-행동-결과) 구조로 정리하는 법을 배우기 어렵다.
- 여러 번에 걸쳐 자소서를 쓰다 보면, 이전에 쓴 내용과 지금 쓴 내용이 서로 다른 주장(역할·기여도 등)을 하게 되는지 확인할 방법이 없다.

---

## 3. 해결하고자 하는 문제

위 배경에서 도출한 5가지 구체적 문제와, 본 시스템이 각 문제에 대응하는 방식은 다음과 같다.

| # | 문제 | 해결 방식 |
|---|------|-----------|
| 1 | "빈 화면 공포" — 무엇부터 써야 할지 모름 | 매일 5대 역량(문제해결/협업/성장/실패경험/주도성) 중 하나에 대한 **구체적 질문 1개**를 AI가 생성해 제공 |
| 2 | 작성한 답변이 질문 의도에 맞는지, 어떤 직무에 적합한지 모름 | AI 분석 모델이 **intent_score(0~100)** 와 **suitable_job(backend/ai_ml/product)** 을 즉시 제공 |
| 3 | 경험을 STAR(상황-과제-행동-결과)로 구조화하지 못해 답변이 추상적 | **STAR 코칭(CoachAgent)** 이 답변에서 부족한 요소를 추출 → 보완 질문 1개 → 재작성까지 자동 수행 |
| 4 | 한 번 쓴 답변에서 더 깊은 디테일을 끌어내지 못함 | 대화 맥락 전체를 보는 **꼬리질문 인터뷰(최대 5턴)** 로 경험을 점진적으로 구체화 |
| 5 | 여러 답변 사이에서 서로 다른 주장을 해도 알아채지 못함(일관성 붕괴) | **rule + NLI(mDeBERTa) + SBERT** 기반 pairwise/global 일관성 검사로 모순 가능성을 즉시 안내 |

**해결 방식 한 줄 요약**: "매일 1문항 + 5개의 역할별 AI 서브에이전트(질문 생성 → AI 분석 → STAR 코칭 → 꼬리질문 인터뷰 → 일관성 검사)"로 구성된 셀프 트레이닝 루프를 통해, 사용자가 반복적으로 자소서 소스를 누적·개선할 수 있게 한다.

---

## 4. 팀 역할 분배



| 구성원 | 역할 |
|---|---|
| A. 박동혁 | 아키텍처 설계 / 데이터 구축 / 베이스라인 구축 |
| B. 안진영 | 아이디어 설계 및 고도화 / CLI 구축 / 모델 설정 |
| C. 이진태 | CLI 구축 및 고도화 / 모델 설정 / 평가 설계 |

---

## 5. 서비스 개요

| 항목 | 내용 |
|------|------|
| **대상** | IT/AI 직무를 준비하는 대학교 1학년 ~ 장기 취준생 |
| **MVP 범위** | IT/AI 직무 한정(backend / ai_ml / product 3개 트랙), CLI 기반 1인 세션, 로컬 실행(CPU/GPU 자동 대응) |
| **세션 단위** | 1세션 = 오늘의 질문 1개 + 최초 답변 + STAR 코칭 + 꼬리질문 인터뷰(최대 5턴) + 결과 저장 |

### 핵심 기능

1. **오늘의 질문 생성** — 5대 역량(문제해결/협업/성장/실패경험/주도성) 중 무작위 선택 후 질문 생성
2. **AI 답변 분석** — intent_score(질문 의도 반영도, 0~100) + suitable_job(적합 직무 3클래스) 예측
3. **STAR 코칭** — 답변에서 상황/과제/행동/결과 요소를 추출하고, 부족한 부분에 대해 1개의 보완 질문 → 재작성
4. **꼬리질문 인터뷰** — 전체 대화 맥락 + 분석 결과 + 일관성 메모를 반영해 최대 5턴까지 후속 질문 진행
5. **일관성 검사** — 현재 답변을 이전 답변들과 pairwise/global로 비교(rule + NLI + SBERT)하여 모순 가능성 안내
6. **적합 직무 변화 추적 + 역량 준비도 대시보드** — 세션 시작/종료 시 직무·점수 변화, 5대 역량 준비도 표시
7. **세션 저장/조회** — JSON + 사람이 읽는 `_final.txt`로 저장, `--history`로 과거 세션 조회

---

## 6. 사용자 플로우

```
[앱 시작: python cli/main.py]
        │
        ▼
[1] 역량 준비도 대시보드 + 약점 영역 추천 미션 표시
        │
        ▼
[2] 오늘의 질문 생성 (5대 역량 중 무작위 선택 → QuestionAgent)
        │
        ▼
[3] 과거 자소서 카드 추천 (유사한 이전 경험이 있으면 함께 안내)
        │
        ▼
[4] conversation.txt 파일 생성 → 사용자가 에디터에서 최초 답변 작성
        │
        ▼
[5] 최초 답변 1차 분석 (AnalysisAgent: intent_score / suitable_job)
        + 일관성 검사 (ConsistencyAgent: 첫 답변은 비교 대상 없어 보통 '양호')
        │
        ▼
[6] STAR 코칭 (CoachAgent)
        ├─ analyze(): 답변에서 상황/과제/행동/결과/도구/수치 추출
        ├─ 부족한 요소에 대한 보완 질문 1개 제시 → 사용자 보완 입력
        ├─ rewrite(): 1차 재작성본 생성
        └─ 재시도 여부 확인 (수용 / 재시도 / 건너뛰기)
        │
        ▼
[7] 꼬리질문 인터뷰 루프 (최대 5턴, InterviewerAgent)
        ├─ conversation_context + intent_score + job + consistency_note 기반 꼬리질문 생성
        ├─ 같은 conversation.txt에 [꼬리질문 n]/[답변 n] 추가 → 사용자 작성
        ├─ 매 턴마다 AnalysisAgent 재분석 + ConsistencyAgent 재검사
        ├─ job_history / suitable_jobs 갱신
        └─ "저장하고 종료? (y/n)" 확인
        │
        ▼
[8] 자소서 카드 빌드 (원본 → 코칭본 → 인터뷰 최종본 3단계 버전 이력,
        STAR 요소, 사용 도구/수치, before-after 점수, 개선 요약)
        │
        ▼
[9] 저장 — storage/sessions/{session_id}.json + {session_id}_final.txt
        │
        ▼
[10] 마무리 대시보드 — 역량 준비도 변화 + 적합 직무 변화 추적 + 세션 요약
```

---

## 7. 데이터 수집 방법

데이터 수집은 링커리어(Linkareer) 자기소개서 게시물(`https://linkareer.com/cover-letter/{id}`, ID 30000~37000 범위)을 대상으로 한다.

| 단계 | 스크립트 | 수집 내용 | 목표 건수 | 실제 수집 건수 (train/test) | 비고 |
|---|---|---|---|---|---|
| Stage1 (질문 생성) | `crawler/crawler.py` | `question_id, question, url, competency` | train 500 / test 100 | **500 / 100** (`data/stage1/train.csv`, `test.csv`) | `IT_KEYWORDS`(IT, 개발, 소프트웨어, AI, 백엔드, 프론트, 데이터, 클라우드, 보안 등)로 IT/AI 관련 공고만 필터링 |
| Stage3 (분석) | `crawler/crawler_stage3.py` | `question, answer, intent_score, suitable_job, url` | 클래스별 train 100 / test 25 (backend·ai_ml·product, 총 300/75) | **300 / 75** (`data/stage3/train.csv`, `test.csv`, 클래스별 100/25) | `--label-method keyword\|llm`로 suitable_job 자동 라벨링. `intent_score`는 크롤링 시 빈 칸으로 두고 별도 라벨링. `ANSWER_MIN_LEN=50`, `ANSWER_MAX_LEN=2000` |
| Stage4 (꼬리질문) | (Stage3 데이터 재사용 + 수동 라벨링) | `question, answer, intent_score, suitable_job, tail_question, url` | train 300 / test 75 | **300 / 75** (`data/stage4/tail_train.csv`, `tail_test.csv`, Stage3 데이터와 1:1) | Stage3 데이터에 intent_score 구간별(0~40 / 41~70 / 71~100) 꼬리질문 작성 규칙(Yes/No 불가, 구체적 경험 유도)을 적용해 `tail_question` 컬럼을 추가 라벨링 |

병합: `crawler/merge.py`가 여러 `batch_*.csv`와 크롤러 결과 CSV를 질문 기준으로 중복 제거한 뒤, `TRAIN_TARGET=500`/`TEST_TARGET=100`에 맞춰 `data/stage1/train.csv`·`test.csv`를 생성한다. 위 "실제 수집 건수"는 현재 `data/` 내 CSV 파일의 실제 행 수 기준이며, 모든 단계에서 목표 건수와 동일하게 달성되었다.

---

## 8. 데이터 가공 방법

1. **intent_score 라벨링** — 크롤링 단계에서는 빈 칸으로 두고, 사람이 0~100점을 구간별 균등(0~40 / 41~70 / 71~100)하게 직접 채점.
2. **suitable_job 라벨링** — `crawler_stage3.py`의 `JOB_KEYWORDS`(backend/ai_ml/product) 키워드 매칭 또는 `--label-method llm` 옵션으로 LLM 기반 분류.
3. **답변 길이 정규화** — `ANSWER_MIN_LEN=50`자 미만 답변 제외, `ANSWER_MAX_LEN=2000`자 초과 시 `ANSWER_SAVE_LEN=1500`자로 절단.
4. **중복 제거 및 분할** — `merge.py`가 질문 텍스트 기준 중복을 제거하고 train/test 목표 건수(500:100, 300:75)에 맞춰 분할.
5. **학습 포맷 변환**
   - Stage1: `build_prompt(competency)` → `"[시스템] {SYSTEM_PROMPT}\n[작업] 역량: {competency}\n[질문] {question}"`
   - Stage4: `format_stage1(row)`/`format_stage4(row)` → `"시스템: {SYSTEM_*}\n사용자: [TASK:generate_question|tail_question] ...\n어시스턴트: {target}{eos}"`
   - 두 포맷 모두 KoGPT2 tokenizer로 토큰화(`TextDataset`, max_length=1024)하며, prompt 영역은 손실 계산에서 `-100`으로 마스킹.
6. **Stage3 입력 포맷** — `[CLS] {question} [SEP] {answer} [SEP]` 형태로 encoder에 입력, 하나의 encoder 출력에서 회귀 head(intent_score)와 분류 head(suitable_job)를 동시에 학습.
7. **KoGPT2 tokenizer 이슈 처리** — `skt/kogpt2-base-v2`를 기본 `AutoTokenizer`로 로드하면 한글이 깨지는 문제가 있어, `PreTrainedTokenizerFast`로 `bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"`를 명시적으로 지정해 해결.

---

## 9. 전체 아키텍처

### 9.1 데이터·학습 파이프라인

```
[Linkareer 자기소개서 게시물 (ID 30000~37000)]
        │
        ├─ crawler.py        ──▶ question_id/question/url/competency
        └─ crawler_stage3.py ──▶ question/answer/suitable_job/url (intent_score는 추후 라벨링)
                │
                ▼
        merge.py (중복 제거 + train/test 목표 건수 맞춤)
                │
   ┌────────────┼─────────────────────────────┐
   ▼            ▼                              ▼
stage1_train/   stage3_train/test.csv          tail_train/test.csv
test.csv        (intent_score 사람 라벨링)      (stage3 데이터 + 구간별 꼬리질문 라벨링)
   │                  │                              │
   ▼                  ▼                              ▼
Stage1 학습        Stage3 학습                    Stage4 학습
(KoGPT2 CPU /      (KoELECTRA-small / ModernBERT  (KoGPT2 CPU 통합 /
 Llama-3.1 LoRA)    멀티태스크: 회귀+분류)          Llama-3.1 LoRA 통합)
   │                  │                              │
   ▼                  ▼                              ▼
stage1_question_gen/  stage3_multitask/             stage4_tail_question/
models/best(_cpu)     models/best(_cpu)             models/best(_cpu)
```

### 9.2 런타임(CLI) 아키텍처 — Orchestrator-Subagent

```
┌───────────────────────────────────────────────────────────────────┐
│                cli/main.py : run_session()  (Orchestrator)          │
│   - Session 상태(질문/답변/intent_score/job_history/                │
│     consistency_checks/conversation_context) 관리                   │
│   - 5개 서브에이전트를 순서대로 호출, AgentResult[T]로 결과 수신     │
└───────┬─────────────┬─────────────┬─────────────┬──────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌──────────────┐ ┌───────────────┐ ┌───────────────┐ ┌────────────────────┐
│QuestionAgent  │ │ AnalysisAgent │ │ConsistencyAgent│ │   CoachAgent        │
│ 티어: light   │ │ 티어:specialized│ │ 티어:specialized│ │ 티어: smart→rule    │
│ KoGPT2 /      │ │ KoELECTRA /   │ │ rule + NLI     │ │ EEVE-Korean-10.8B   │
│ Llama-3.1     │ │ ModernBERT    │ │ (mDeBERTa) +   │ │ (Ollama, JSON 모드) │
│ (4bit/LoRA)   │ │ 멀티태스크 +  │ │ SBERT(KR-SBERT)│ │                     │
│               │ │ rule 보정     │ │ pairwise+global│ │                     │
└──────────────┘ └───────────────┘ └────────────────┘ └─────────┬───────────┘
                                                                  ▼
                                                       ┌─────────────────────┐
                                                       │ InterviewerAgent     │
                                                       │ 티어: smart→light→rule│
                                                       │ EEVE-Korean-10.8B    │
                                                       │ (Ollama)             │
                                                       └─────────────────────┘

           모든 Agent는 AgentResult(data, tier_used, fallback_used) 반환
                                   │
                                   ▼
        storage/drafts/{session_id}_conversation.txt   (사용자 입력 파일)
        storage/sessions/{session_id}.json,            (구조화된 결과)
        storage/sessions/{session_id}_final.txt        (사람이 읽는 요약)
```

---

## 10. 사용 기술/스택

| 영역 | 기술 / 모델 | 비고 |
|---|---|---|
| 데이터 수집 | Python, `requests`/`playwright`/`selenium`(선택) | Linkareer 자소서 게시물 크롤링 |
| 데이터 처리 | pandas, scikit-learn | CSV 병합/중복 제거/train-test 분할 |
| Stage1·4 생성 모델 (CPU) | `skt/kogpt2-base-v2` (full fine-tuning) | 실제 사용 중인 체크포인트 |
| Stage1·4 생성 모델 (GPU) | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (LoRA r=16) | 코드상 준비됨. 체크포인트 미존재로 GPU에서도 현재는 base 모델 사용 (15번 한계점 참고) |
| Stage3 분석 모델 (CPU) | `monologg/koelectra-small-v3-discriminator` (멀티태스크) | 실제 사용 중인 체크포인트 |
| Stage3 분석 모델 (GPU) | `answerdotai/ModernBERT-base` (멀티태스크) | 동일 아키텍처(`_MultiTaskModel`) 공유 |
| 평가 | sentence-transformers(Cosine 유사도), numpy(MAE/RMSE), scikit-learn(Accuracy/F1-macro/Confusion Matrix) | `evaluate.py --baseline` / 기본 실행으로 파인튜닝 전후 비교. Stage별로 산출 지표가 다름(11번 참고) |
| ⭐ "smart" 코칭/인터뷰 LLM | `yanolja/EEVE-Korean-Instruct-10.8B-v1.0` Q4_K_M GGUF (Ollama 서빙) | GGUF를 직접 받아 `Modelfile`로 `ollama create` 등록 — 본 프로젝트의 핵심 차별화 요소 |
| ⭐ 일관성 검사 | rule 기반 키워드 충돌 + `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`(NLI) + `snunlp/KR-SBERT-V40K-klueNLI-augSTS`(SBERT), pairwise + global 비교 | 세 방식을 조합한 자체 설계 |
| ⭐ Orchestrator-Subagent 구조 | `AgentResult[T](data, tier_used, fallback_used)` 공통 계약 + `cli/config.yaml`의 `agents:` 섹션으로 역할별 모델 티어 고정 배정 | 5개 서브에이전트(질문/인터뷰/코칭/분석/일관성)로 분리 |
| ⭐ intent_score rule 보정 | `AnalysisAgent`가 회귀 출력이 `<=0`일 때 `coach.estimate_score()`로 대체 | Stage3 회귀 헤드 성능 저하(11번) 대응 |
| 모델 로딩/추론 | PyTorch, Transformers, Unsloth(LoRA), `torch.cuda.is_available()` 기반 GPU/CPU 자동 분기 (`cli/models.py`) | |
| 저장소 | 표준 라이브러리(json, os, uuid) 기반 자체 JSON 저장소 (`cli/storage.py`) | |

---

## 11. 각 단계별 모델 설계 / 학습 / 평가

### Stage 1 — 질문 생성

**A. Task 정의**
- 입력: `competency` (문제해결 / 협업 / 성장 / 실패경험 / 주도성 중 1개)
- 출력: 해당 역량을 묻는 자기소개서 질문 1개
- 모델: `skt/kogpt2-base-v2` CPU full fine-tuning (실사용) / `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` GPU LoRA r=16 (코드상 준비, 체크포인트 미존재)
- ⚠️ 이 섹션의 체크포인트(`stage1_question_gen/models/best_cpu`)는 Stage1 단독 평가용으로 별도 학습된 것이다. **CLI 런타임(`QuestionAgent`)이 실제로 로드하는 모델은 Stage4에서 Task1+Task2를 함께 학습한 `stage4_tail_question/models/best_cpu`**이다 (`cli/models.py`의 `CPU_GEN_TUNED`). 자세한 내용은 Stage 4 섹션 참고.

**B. 평가지표 정의 / 목적 / 좋은 기준**

> `stage1_question_gen/evaluate.py` 기준 — 현재는 **Cosine 유사도**와 **valid_ratio**만 산출한다. (BLEU-4/ROUGE-L/composite은 계산하지 않음 — `eval_results.json`에 남아있는 해당 값은 과거 버전의 잔여 필드)

| 지표 | 정의 | 목적 / 좋은 기준 |
|---|---|---|
| Cosine | 생성 질문과 동일 역량(competency)의 정답 질문들 간 SBERT 임베딩 코사인 유사도 최대값의 평균 | 표현이 달라도 해당 역량의 의도에 맞는 질문인지 (cosine_min=0.75, cosine_goal=0.80) |
| valid_ratio | Cosine 유사도가 valid_threshold(0.70) 이상인 샘플의 비율 | 생성 질문이 "역량에 맞는 유효한 질문"으로 인정되는 비율 |

**결과** (`stage1_question_gen/models/best_cpu/eval_results*.json`, n=100)

| 모드 | Cosine | valid_ratio |
|---|---|---|
| baseline (파인튜닝 전) | 0.485 | 0.00 |
| finetuned | 0.671 | 0.54 |

→ Cosine 0.485 → 0.671로 개선되었으나 목표(0.80)에는 아직 미달. `valid_ratio`는 0 → 0.54로 크게 향상되어 "역량에 맞는 질문" 생성이 절반 이상의 샘플에서 안정화되었다. 500건 규모의 학습 데이터 한계로 추정.

---

### Stage 3 — 분석 (intent_score + suitable_job)

**A. Task 정의**
- 입력: `(question, answer)`
- 출력: `intent_score`(0~100, 회귀) + `suitable_job`(backend/ai_ml/product, 3클래스 분류)
- 모델: 단일 encoder(`monologg/koelectra-small-v3-discriminator` CPU / `answerdotai/ModernBERT-base` GPU) + 회귀 head(Linear→1) + 분류 head(Linear→3)

**B. 평가지표 정의 / 목적 / 좋은 기준**

> `stage3_multitask/evaluate.py` 기준 — 회귀는 **MAE/RMSE**, 분류는 **Accuracy/F1-macro**(+ Confusion Matrix, 콘솔 출력만) 산출. (Pearson r / Precision / Recall은 계산하지 않음 — `eval_results.json`에 남아있는 해당 값은 과거 버전의 잔여 필드)

| 지표 | 정의 | 목적 / 좋은 기준 |
|---|---|---|
| MAE | intent_score 예측-정답 평균 절대오차 | 점수 오차 크기 (mae_threshold=10) |
| RMSE | 평균 제곱근 오차 | 큰 오차에 민감 (rmse_threshold=15) |
| Accuracy | suitable_job 분류 정확도 | accuracy_threshold=0.75 |
| F1-macro | 3클래스 평균 F1 (클래스 불균형 보정) | f1_threshold=0.70 |
| Confusion Matrix | 클래스별 예측/정답 분포 (콘솔 출력) | 분류 오류 유형 파악 |

**결과** (`stage3_multitask/models/best/eval_results*.json`, n=75)

| 모드 | MAE | RMSE | Accuracy | F1-macro |
|---|---|---|---|---|
| baseline | 3.11 | 3.82 | 0.267 | 0.140 |
| finetuned | 7.55 | 9.40 | 0.533 | 0.533 |

→ **분류(suitable_job)** 는 파인튜닝 후 크게 개선(Accuracy 0.267 → 0.533, F1-macro 0.140 → 0.533, 둘 다 목표 미달이지만 큰 향상). 그러나 **회귀(intent_score)** 는 오히려 악화(MAE 3.11 → 7.55, RMSE 3.82 → 9.40, 둘 다 threshold 초과) — 회귀 head가 거의 일정한 값을 출력하는 것으로 추정되며, 300건 규모의 라벨 데이터로는 0~100 연속 점수 학습이 부족했던 것으로 보인다. 이 문제와 대응 방안은 14번/15번에서 자세히 다룬다.

---

### Stage 4 — 꼬리질문(+질문) 통합 생성

**A. Task 정의**
- Task 1 (Stage1 재사용): `competency → question`
- Task 2 (신규): `(question, answer, intent_score, suitable_job) → tail_question` — Yes/No로 답할 수 없는, 구체적 경험을 끌어내는 후속 질문
- 모델: `skt/kogpt2-base-v2` CPU full fine-tuning, 두 Task를 하나의 모델에 함께 학습 (실사용) / `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` GPU LoRA (코드상 준비, 체크포인트 미존재)
- ⭐ **이 체크포인트(`stage4_tail_question/models/best_cpu` / GPU는 `models/best`)가 CLI 런타임에서 질문·꼬리질문 생성을 모두 담당한다.** `cli/models.py`의 `load_generation_model()`이 `CPU_GEN_TUNED`/`GPU_GEN_TUNED`로 이 경로를 로드하고, `QuestionAgent.ask()`(`[TASK:generate_question]`)와 `InterviewerAgent`의 light 폴백(`[TASK:tail_question]`)이 동일한 모델·토크나이저 인스턴스를 공유한다. `train.py`가 `data/stage1/train.csv`(Task1, 500건) + `data/stage4/tail_train.csv`(Task2, 300건)를 함께 학습하기 때문에 가능한 구조다.

**B. 평가지표 정의 / 목적 / 좋은 기준**

> `stage4_tail_question/evaluate.py` 기준 — 현재는 **Cosine 유사도**와 **open_question_ratio**만 산출한다. (BLEU-4/ROUGE-L은 계산하지 않음 — `eval_results.json`에 남아있는 해당 값은 과거 버전의 잔여 필드)

| 지표 | 정의 | 목적 / 좋은 기준 |
|---|---|---|
| Cosine | 생성 꼬리질문과 정답 꼬리질문 간 SBERT 임베딩 코사인 유사도 | 정답과 같은 의도의 질문인지 (cosine_threshold=0.60) |
| open_question_ratio | 생성 꼬리질문 중 길이 15자 초과 + "?"/"요" 포함 + Yes/No 패턴으로 끝나지 않는(개방형) 질문 비율 | 인터뷰 효과의 핵심 지표 — "더 말하게 하는" 질문 |

**결과** (`stage4_tail_question/models/best_cpu/eval_results*.json`, n=75)

| 모드 | Cosine | open_question_ratio |
|---|---|---|
| baseline | 0.247 | 0.373 |
| finetuned | 0.539 | 1.000 |

→ `open_question_ratio`가 0.373 → 1.000으로 개선되어, 파인튜닝 후 생성된 꼬리질문은 전부 개방형 질문. Cosine도 0.247 → 0.539로 개선되었으나 목표(cosine_threshold=0.60)에는 근소하게 미달 — "표현은 다르지만 의미는 더 유사한" 질문을 생성하는 방향으로 개선되었으나, 정답과의 의미적 정합도는 추가 개선 여지가 있다.

---

## 12. 핵심 기능

### A. CLI 방식

- `python cli/main.py`로 세션 시작 → 역량 준비도 대시보드 + 추천 미션 + 오늘의 질문 출력
- 답변은 `storage/drafts/{session_id}_conversation.txt`라는 텍스트 파일에 작성 (`$EDITOR`/`nano`/`vi`, Windows는 `notepad`)
- `[역량]` / `[원본 질문]` / `[최초 답변]` / `[꼬리질문 n]` / `[답변 n]` 섹션 구조로 누적 — 기존 내용은 지우지 않고 아래에 이어 씀
- 입력 규칙: 최초 답변 30자 이상, 꼬리질문 답변 10자 이상(`MIN_ANSWER_LEN`), `#`으로 시작하는 줄은 주석 처리되어 답변에서 제외
- `python cli/main.py --history`로 과거 세션 기록 조회

### B. 가벼운 파인튜닝

- `skt/kogpt2-base-v2`(생성, Stage1·4 통합)와 `monologg/koelectra-small-v3-discriminator`(분석, Stage3)를 **CPU에서 full fine-tuning** — GPU 없이도 동작
- 각각 수백 건 규모(Stage1 500/100, Stage3·4 각 300/75)의 자체 수집·라벨링 데이터로 학습
- KoGPT2 tokenizer의 한글 디코딩 깨짐 문제를 `PreTrainedTokenizerFast` 명시적 설정으로 해결
- GPU 환경(Unsloth)에서 `Llama-3.1-8B-4bit`를 LoRA(r=16)로 파인튜닝하는 경로도 코드에 준비되어 있으나, 현재 제출 시점에는 GPU LoRA 체크포인트가 없어 CPU(KoGPT2/KoELECTRA) 체크포인트가 실제 CLI에서 사용된다.

### C. AI agentic Orchestrator-Subagent 구조

- **Orchestrator**: `cli/main.py`의 `run_session()` — Session 상태를 들고 전체 흐름(질문 → 분석 → 코칭 → 인터뷰 → 저장)을 제어하는 순수 Python 함수. 직접 LLM을 호출하지 않고 모든 모델 호출을 서브에이전트에 위임.
- **5개 서브에이전트** (`cli/agents/`), 모두 `AgentResult[T](data, tier_used, fallback_used)` 공통 타입으로 응답:

| 서브에이전트 | 책임 | 고정 티어 | 폴백 체인 |
|---|---|---|---|
| `QuestionAgent` | 오늘의 질문 생성 | light (CPU: KoGPT2 `stage4_tail_question/models/best_cpu` / GPU: Llama-3.1-8B-4bit `models/best`, 없으면 base — Stage1+Stage4 통합 학습 체크포인트, `InterviewerAgent`와 공유) | 모델 로드 자체가 실패한 경우만 → 샘플 질문 |
| `InterviewerAgent` | 꼬리질문 생성 | smart (EEVE-Korean-10.8B, Ollama) | Ollama 실패 → light → rule(샘플) |
| `CoachAgent` | STAR 추출(analyze)/재작성(rewrite) | smart (EEVE-Korean-10.8B, Ollama) | Ollama 실패 → rule 기반 |
| `AnalysisAgent` | intent_score/적합 직무 예측 | specialized (KoELECTRA 멀티태스크) | 회귀값 ≤0 → rule(`estimate_score`) 보정 / 체크포인트 없음 → rule |
| `ConsistencyAgent` | 일관성 검사 | specialized (rule+NLI+SBERT) | (내부 단계적 폴백) |

- **모델 간 연결/소통**: Orchestrator가 Session 상태(`conversation_context`, `intent_score`, `job`, `consistency_checks` 등)를 각 에이전트의 입력 타입(예: `InterviewContext`)으로 직렬화해 전달 → 에이전트는 `AgentResult`로 결과 반환 → Orchestrator가 Session에 누적해 다음 호출에 재사용한다. 예를 들어 `AnalysisAgent`가 계산한 `intent_score`/`job`은 `InterviewerAgent`의 꼬리질문 생성 입력(`InterviewContext`)으로 직접 전달된다.
- **역할별 모델 티어 고정 배정**은 `cli/config.yaml`의 `agents:` 섹션 한 곳에서 관리 — 어떤 서브에이전트가 어떤 모델을 쓸지 일괄 확인/변경 가능.
- 모든 "smart" 호출은 Ollama 장애/빈 응답 시 자동으로 light 또는 rule 단계로 graceful fallback되며, `fallback_used=True`일 때 `"[안내] ... 로 진행합니다"` 메시지를 출력한다.

---

## 13. 시연 재현 설명

### A. 초기 셋팅 명령어

```bash
# 1. conda 환경 생성
conda create -n jobfit python=3.11 -y
conda activate jobfit

# 2. PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Unsloth 설치 (GPU 환경에서 LoRA 학습/추론용)
pip install unsloth

# 4. 나머지 의존성 설치
pip install -r requirements.txt

# 5. (선택) 크롤러 의존성
pip install playwright selenium webdriver-manager
playwright install chromium

# 6. "smart" 코칭/인터뷰 모델 — EEVE-Korean-Instruct-10.8B를 Ollama에 등록
#    Modelfile 예시:
#    FROM /path/to/EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf
ollama create eeve-korean-10.8b -f Modelfile
ollama list   # eeve-korean-10.8b:latest 확인
```

### B. 각 과정 실행 명령어

```bash
# 1) 데이터 수집
python crawler/crawler.py --output-dir ./data/stage1
python crawler/crawler_stage3.py --output-dir ./data/stage3 --label-method keyword
python crawler/merge.py

# 2) Stage1 — 질문 생성 모델
cd stage1_question_gen
python train.py
python evaluate.py --baseline   # 파인튜닝 전
python evaluate.py              # 파인튜닝 후

# 3) Stage3 — 분석 모델
cd ../stage3_multitask
python train.py
python evaluate.py --baseline
python evaluate.py

# 4) Stage4 — 꼬리질문(+질문) 통합 생성 모델
cd ../stage4_tail_question
python train.py
python evaluate.py --baseline
python evaluate.py

# 5) CLI 실행 (test4/ 루트에서)
cd ..
python cli/main.py            # 새 세션: 질문 → 답변 → AI 분석 → STAR 코칭 → 꼬리질문 인터뷰 → 저장
python cli/main.py --history   # 저장된 세션 기록 조회
```

> `cli/main.py` 상단의 `USE_STAGE14_FINETUNED`/`USE_STAGE3_FINETUNED` 플래그가 `True`로 설정되어 있어야 학습된 체크포인트(`models/best`, `models/best_cpu`)를 사용한다. `cli/config.yaml`은 `test4/` 루트에서 실행해야 상대 경로가 정상 동작한다.

---

## 14. 진행 중 문제점 / 해결 방법

### 핵심 문제 3가지

**1) Stage3 intent_score 회귀 헤드 성능 저하 (MAE/RMSE가 파인튜닝 후 오히려 악화)**

- **문제**: `stage3_multitask` 모델의 회귀 head가 사실상 거의 일정한(≤0으로 clamp되는) 값만 출력 → CLI에서 "의도 반영도"가 항상 `0.0/100`으로 표시됨
- **원인 분석**: `eval_results.json` 기준 baseline(파인튜닝 전, MAE=3.11/RMSE=3.82)보다 finetuned(MAE=7.55/RMSE=9.40)가 더 큰 오차를 보임. 300건(클래스 균등 100×3) 규모의 intent_score 라벨 데이터로는 0~100 연속 점수를 학습하기에 부족했던 것으로 추정
- **해결**: `cli/agents/analysis_agent.py`의 `AnalysisAgent.predict()`에 보정 로직 추가 — 모델 출력 `score <= 0.0`일 때 `coach.estimate_score(question, answer)`(rule 기반 점수 추정)로 대체하고 `tier_used="specialized", fallback_used=True`로 기록. 근본적인 재학습(라벨 데이터 확충) 전까지의 임시 보정 조치.

**2) 로컬 한국어 "smart" LLM 티어 구축 — Ollama 모델 부재 → EEVE-Korean-10.8B 직접 빌드**

- **문제**: 기존 코칭(STAR 추출/재작성) 로직이 Ollama의 특정 모델을 호출했으나 로컬 환경에 해당 모델이 등록되어 있지 않아 호출이 항상 실패하고 rule 기반 폴백만 동작
- **해결**: `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`의 Q4_K_M GGUF(약 6.5GB)를 직접 받아 `Modelfile`을 작성하고 `ollama create eeve-korean-10.8b -f Modelfile`로 로컬에 등록. `cli/config.yaml`의 `models.smart.model: eeve-korean-10.8b`로 지정해 `CoachAgent`/`InterviewerAgent`의 "smart" 티어가 정상 동작하도록 구성

**3) 3개 브랜치(test3 / jintae_v2 / jinyoung_v2) 통합 — 서로 다른 인터페이스를 공통 계약으로 재설계**

- **문제**: 세 브랜치가 서로 다른 `Session` 필드, 서로 다른 모델 호출 방식(GPU Llama-3.1 LoRA vs CPU KoGPT2/KoELECTRA), 서로 다른 코칭 로직(rule-only vs Ollama)을 갖고 있어 그대로 합치면 인터페이스가 충돌
- **해결**:
  - `Session`을 jintae_v2(코칭 채점 필드: `input_type`/`score_before`/`score_after`/`retry_used`/`answer_card`/`versions`) + jinyoung_v2(멀티턴/일관성/직무 추적 필드: `job_history`/`consistency_checks`/`conversation_context`)의 합집합으로 재설계
  - `cli/models.py`를 GPU/CPU 자동 분기 + 동일 공개 API(`generate_question`/`generate_tail_question`/`AnalysisModel`)로 통합
  - Orchestrator-Subagent 구조(`AgentResult[T]`)를 도입해 5개 역할별로 모델 티어를 고정 배정하고 graceful fallback을 표준화

### 그 외 문제점 (간단 나열)

- KoGPT2 tokenizer 한글 디코딩 깨짐(`�`) → `PreTrainedTokenizerFast`로 special token(bos/eos/unk/pad/mask) 명시 지정해 해결
- `requirements.txt` 병합 시 `bitsandbytes` 누락 발견 → GPU 4bit 추론을 위해 `bitsandbytes>=0.43.0` 추가
- `conversation.txt` 기반 입력에서 답변 길이 미달(최초 30자/꼬리 10자 미만) 시 에디터가 재오픈됨 — 의도된 검증 로직이나, 처음 사용하는 사람에게는 "왜 안 닫히지?"라는 혼란을 줄 수 있어 안내 문구/문서화 필요
- GPU/CPU 환경에 따라 생성 모델 프롬프트 포맷(Llama-3 채팅 템플릿 vs 평문 한국어 프롬프트 + 후처리)이 달라 `models.py`에서 이중 분기 처리 필요

---

## 15. 한계점 / 향후 발전 가능성

### 한계점

- **데이터 규모**: Stage1 500/100건, Stage3·4 각 300/75건 — 소규모 자체 수집·라벨링 데이터로 인해 Cosine(Stage1·4)·MAE/RMSE(Stage3 회귀) 등 지표가 전반적으로 목표치에 미달
- **Stage3 회귀(intent_score) 학습 실패**: MAE/RMSE가 파인튜닝 후 오히려 증가(3.11→7.55 / 3.82→9.40) — 현재는 AnalysisAgent의 rule 기반 보정으로 우회 중이며 근본적으로는 해결되지 않음
- **GPU LoRA 체크포인트 부재**: GPU 환경에서도 Stage1/4의 Llama-3.1 LoRA 체크포인트가 없어, 실제로는 파인튜닝되지 않은 base `Llama-3.1-8B-4bit`가 질문/꼬리질문 생성에 사용됨
- **"smart" 티어의 Ollama 의존성**: Ollama 로컬 서버가 설치/실행되지 않은 환경에서는 코칭/꼬리질문이 light 또는 rule 기반으로만 동작해 품질이 저하됨
- **Stage4 Cosine 미달**: finetuned cosine(0.539)이 목표(cosine_threshold=0.60)에 근소하게 못 미침 — 추가 데이터/학습 epoch로 개선 가능성 있음
- **일관성 검사 임계값의 경험적 설정**: NLI contradiction(≥0.65/0.80), SBERT similarity(<0.35 등) 임계값이 경험적 기본값이며, 라벨링된 모순 데이터셋 기반의 정량적 검증은 수행되지 않음

### 향후 발전 가능성

- Stage3 intent_score 라벨 데이터를 확충(현재 300건 → 1,000건 이상)하여 회귀 head 재학습 → intent_score 신뢰도 확보
- Stage1/4 GPU LoRA 학습을 실제로 진행해 `Llama-3.1-8B` 기반 생성 품질 향상 (현재 KoGPT2 대비 더 자연스러운 한국어 생성 기대)
- `EEVE-Korean-10.8B`를 본 프로젝트의 코칭/꼬리질문 데이터로 추가 SFT/DPO하여 "smart" 티어의 도메인 적합도 향상
- 일관성 검사용 모순 데이터셋을 직접 구축해 NLI/SBERT 임계값을 정량적으로 튜닝
- CLI → 웹/모바일 UI 확장, 영문 자소서 등 다국어 지원
- 누적 세션 데이터를 활용한 개인화 추천(약점 역량 집중 트레이닝 스케줄링) 고도화
