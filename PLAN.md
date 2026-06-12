# NLP 프로젝트 - 하루 1 문답 : 개발 단계 계획

> 기간: 4월 ~ 6월 | 멤버: 안진영 이진태 박동혁

---

## 프로젝트 개요

**목표**: 자소서 글을 쓰는 힘을 길러주는 훈련기 (메타인지)  
**대상**: 대학교 1학년 ~ 장기 취준생  
**범위(MVP)**: IT / AI 관련 직무 한정

### 핵심 유저 플로우

```
1. 무작위 질문 제공 (AI 생성)
      ↓
2. 사용자가 자신의 경험 작성
      ↓
3. AI 분석 - 의도 반영도(intent_score) + 적합 직무(suitable_job)
      ↓
4. CLI 형식 꼬리질문으로 답변 개선
      ↓
5. 저장 (적합 직무 TOP3 + 최종 답변)
```

---

## 전체 단계 구조

| 단계 | 내용 | 핵심 모델 | 우선순위 |
|------|------|-----------|----------|
| STEP 0 | 데이터 수집 및 구축 | - | 필수 |
| STEP 1 | EDA 및 전처리 | - | 필수 |
| STEP 2 | 질문 생성 모델 Fine-tuning | Qwen3-8B (LoRA) | 필수 |
| STEP 3 | Multi-task 분석 모델 Fine-tuning | ModernBERT / klue/roberta-large | 필수 (main) |
| STEP 4 | 꼬리질문 생성 모델 Fine-tuning | Qwen3-8B (LoRA, 동일 모델) | 필수 |
| STEP 5 | CLI 구축 (개인별) | LangChain Agent | 필수 |
| STEP 6 | 평가 및 지표 산출 | - | 필수 |
| STEP 7 | 팀 비교 및 최종 통합 | - | 필수 |

---

## STEP 0. 데이터 수집

### 목적
모델 학습에 필요한 원천 데이터 구축

### 수집 대상

#### 1단계용 - 질문 생성 훈련 데이터 (train.csv)

| 필드 | 설명 |
|------|------|
| `question_id` | Q001, Q002 ... |
| `competency` | 문제해결 / 협업 / 성장 / 실패경험 / 주도성 |
| `question` | 실제 자소서 질문 |

**수집처**
- 잡코리아, 원티드, 링크드인 IT/AI 직무 채용공고
- 블라인드, 취준생 오픈채팅 자소서 질문 모음
- 자소서 가이드북 / 블로그 질문 예시
- 수집 시 회사명도 함께 기록 (훈련용 아니더라도 보관)

**목표 수량**: competency별 균등 분배, EDA로 최종 결정

#### 3단계용 - Multi-task 훈련 데이터

**intent_score Regression train set**

| 필드 | 설명 |
|------|------|
| `question` | 자소서 질문 |
| `answer` | 사용자 답변 |
| `intent_score` | 0~100 (의도 반영도, gold label) |
| `generated_intent_score` | 모델 예측값 (처음엔 비워둠) |

- 수집량: **60~80개**
- intent_score 구간별 균등 분포
  - 0~40: 20개
  - 41~70: 20개
  - 71~100: 20개

**suitable_job Classification train set**

| 필드 | 설명 |
|------|------|
| `question` | 자소서 질문 |
| `answer` | 사용자 답변 |
| `suitable_job` | backend / ai_ml / product (gold label) |

- 수집량: **60~90개**
- 클래스별 균등 분포 (불균형 방지)
  - backend: 20~30개
  - ai_ml: 20~30개
  - product: 20~30개

#### 꼬리질문 생성 데이터

**훈련용**

| 필드 | 설명 |
|------|------|
| `question` | 원본 질문 |
| `answer` | 사용자 답변 |
| `intent_score` | 3단계 분석 결과 (조건) |
| `suitable_job` | 3단계 분석 결과 (조건) |
| `tail_question` | 정답 꼬리질문 (gold standard) |

- 수집량: **훈련용 300개 / 검증용 75개**
- intent_score 구간별 균등 분포
  - 0~40: 100개
  - 41~70: 100개
  - 71~100: 100개

**꼬리질문 규칙 (훈련 데이터 품질 기준)**
- Yes/No로 답할 수 없는 질문
- 구체적 경험을 끌어내는 질문
- intent_score 낮을수록 → 기본 경험 유도
- intent_score 높을수록 → 더 깊은 사고 유도
- suitable_job에 맞는 전문성 반영

**intent_score별 꼬리질문 전략**
- 0~40 (저): "그 상황에서 본인의 역할은 구체적으로 무엇이었나요?"
- 41~70 (중): "그 방법을 선택한 근거와 대안을 검토한 과정은?"
- 71~100 (고): "그 경험이 이후 의사결정 방식에 어떤 영향을 줬나요?"

### 산출물
```
data/
├── raw/               # 원천 수집 데이터
├── train.csv          # 질문 생성 훈련용
├── test.csv           # 질문 생성 테스트용 (generated_question 비어있음)
├── intent_train.csv   # intent_score 회귀 훈련용
├── intent_test.csv    # intent_score 회귀 테스트용
├── job_train.csv      # suitable_job 분류 훈련용
├── job_test.csv       # suitable_job 분류 테스트용
├── tail_train.csv     # 꼬리질문 생성 훈련용
└── tail_test.csv      # 꼬리질문 생성 테스트용
```

---

## STEP 1. EDA 및 전처리

### 목적
데이터 품질 확인 및 학습 적합 형태로 가공

### 작업 내용

1. **분포 분석**
   - competency별 질문 수 분포 시각화
   - intent_score 구간별 분포 확인
   - suitable_job 클래스 불균형 확인

2. **텍스트 전처리**
   - 중복 제거
   - 최소 길이 필터 (너무 짧은 답변 제거)
   - 특수문자 정리

3. **훈련/검증/테스트 분리**
   - 각 데이터셋별 split 전략 수립

4. **Instruction 포맷 변환** (Stage 1, 4용)
   ```
   [TASK:generate_question] competency: {competency} → {question}
   [TASK:tail_question] question: {q} answer: {a} intent_score: {s} job: {j} → {tail_q}
   ```

### 산출물
```
notebooks/
└── 01_EDA.ipynb
```

---

## STEP 2. 질문 생성 모델 Fine-tuning (Stage 1)

### 목적
주어진 competency에 맞는 자소서 질문을 생성하는 모델 학습

### 선택 모델

| 모델 | 크기 | 학습 방식 | VRAM | 한국어 | 추천 |
|------|------|-----------|------|--------|------|
| Qwen3-4B | 4B | Full FT | 14GB | 상 | ★★★★ |
| **Qwen3-8B** | **8B** | **LoRA (Unsloth)** | **~8GB** | **상** | **★★★★★** |
| Gemma 3 4B | 4B | Full FT | 14GB | 상 | ★★★★ |
| Qwen3-14B | 14B | QLoRA | 14GB | 최상 | ★★★★ |

**최종 선택: Qwen3-8B + Unsloth LoRA**

**근거**
- Unsloth가 Blackwell 지원 + LoRA로 8GB만 사용
- Qwen3-8B는 동급 모델 중 한국어 생성 품질 최상위
- 14B는 QLoRA로 가능하지만 학습 시간 1.5배

### 학습 설정
```python
# Instruction tuning - 두 Task 동시 학습
task_prefix = "[TASK:generate_question]"  # Stage 2용
# Stage 4 꼬리질문과 동일 모델에서 multi-task 처리
```

### 평가 지표 (정량)

| 지표 | 설명 | 기준값 |
|------|------|--------|
| BLEU-4 | n-gram 겹침으로 유사도 측정 | > 0.35 |
| ROUGE-L | 최장 공통 부분 수열 유사도 | > 0.40 |
| 코사인 유사도 | 임베딩 벡터 간 유사도 | > 0.80 |

### 평가 지표 (정성 - 샘플 30개 human eval)

| 지표 | 설명 | 척도 |
|------|------|------|
| 경험 유도력 | 구체적 경험을 끌어낼 수 있는가 | 1~5점 |
| 역량 적합성 | 해당 역량을 측정하기에 적절한가 | 1~5점 |
| 자연스러움 | 실제 채용공고에 나올 법한가 | 1~5점 |

### 산출물
```
models/stage1_question_gen/
notebooks/
└── 02_stage1_question_gen.ipynb
```

---

## STEP 3. Multi-task 분석 모델 Fine-tuning (Stage 3) ← Main

### 목적
사용자 답변을 분석하여 의도 반영도와 적합 직무를 동시에 예측

### 선택 모델

| 모델 | 크기 | 학습 | VRAM | 추천 |
|------|------|------|------|------|
| klue/roberta-large | 337M | Full FT | 8GB | ★★★★ |
| **ModernBERT-base (multilingual)** | **150M** | **Full FT** | **4GB** | **★★★★★** |
| kykim/bert-kor-base | 110M | Full FT | 3GB | ★★★★ |

**최종 선택: answerdotai/ModernBERT-base (또는 한국어 데이터 많으면 klue/roberta-large)**

**근거**
- ModernBERT는 8K 컨텍스트 (자소서 길어져도 안전) + 빠른 학습
- 한국어 특화가 절대 필요하면 klue/roberta-large 안전책
- Encoder-only는 분류/회귀에 정석 (LLM보다 효율적)

### Multi-task 구조

```
입력: [question] + [answer]
        ↓
   ModernBERT Encoder
        ↓
   ┌────────────────┐
   │  Head 1        │  → intent_score (Regression, 0~100)
   │  Head 2        │  → suitable_job (Classification, 3 classes)
   └────────────────┘
```

### Task 1: intent_score Regression

**평가 지표**

| 지표 | 설명 | 기준값 |
|------|------|--------|
| MAE | 정답과 예측값 평균 절대 오차 | ≤ 10 |
| RMSE | 오차 제곱 평균의 제곱근 | ≤ 15 |
| Pearson r | 정답-예측 상관계수 | ≥ 0.80 |

### Task 2: suitable_job Classification

**클래스**: `backend` / `ai_ml` / `product`

**평가 지표**

| 지표 | 설명 | 기준값 |
|------|------|--------|
| Accuracy | 전체 정답률 | > 0.75 |
| F1-score (macro) | 클래스 불균형 고려 F1 | > 0.70 |
| Precision (macro) | 예측한 것 중 맞은 비율 | > 0.70 |
| Recall (macro) | 실제 정답 중 맞춘 비율 | > 0.70 |
| Confusion Matrix | 오분류 패턴 시각화 | - |

### 산출물
```
models/stage3_multitask/
notebooks/
└── 03_stage3_multitask.ipynb
```

---

## STEP 4. 꼬리질문 생성 모델 Fine-tuning (Stage 4)

### 목적
분석 결과(intent_score, suitable_job)를 조건으로 적절한 꼬리질문 생성

### 모델
**Qwen3-8B (STEP 2와 동일 모델, Instruction Multi-task)**

```python
task_prefix = "[TASK:tail_question]"

# 입력 형식
f"[TASK:tail_question] question: {q} answer: {a} intent_score: {score} job: {job}"
# → tail_question 생성
```

**적절 수집 개수**
- 훈련: 300개 (intent_score 구간별 100개씩)
- 검증: 75개 (구간별 25개씩)

### 평가 지표 (정량)

| 지표 | 설명 | 기준값 |
|------|------|--------|
| BLEU-4 | 정답과 생성 꼬리질문 n-gram 겹침 | > 0.35 |
| ROUGE-L | 최장 공통 부분 수열 유사도 | > 0.40 |
| 코사인 유사도 | 임베딩 벡터 간 유사도 | > 0.80 |

### 평가 지표 (정성 - Human Eval / LLM-as-a-Judge)

| 지표 | 설명 | 방법 |
|------|------|------|
| 사고 유도력 | 더 깊은 답변을 끌어낼 수 있는가 | Human Eval / LLM-as-a-Judge |
| intent_score 반영 | 점수에 맞는 난이도의 질문인가 | Human Eval / LLM-as-a-Judge |
| 직무 연관성 | suitable_job에 맞는 전문성이 담겼는가 | Human Eval / LLM-as-a-Judge |
| Yes/No 불가 여부 | 단답형으로 답할 수 없는가 | 자동 검증 가능 |

### 산출물
```
models/stage4_tail_question/
notebooks/
└── 04_stage4_tail_question.ipynb
```

---

## STEP 5. CLI 구축 (개인별 독립 개발)

### 목적
3명이 각자 독립적인 CLI를 구축하여 비교 후 최종 통합

### 전체 아키텍처 (직접 호출 파이프라인)

```
[CLI 입력]
     │
     ▼
models.load_generation_model()   Qwen3-8B LoRA  (질문 생성 + 꼬리질문)
models.load_analysis_model()     ModernBERT     (intent_score + suitable_job)
     │
     ▼
generate_question(competency)    → 질문 출력
     │
     ▼
get_user_answer()                → 답변 입력
     │
     ▼
analysis.predict(q, a)           → intent_score, suitable_job
     │
     ▼
[꼬리질문 루프 × MAX_TURNS]
  generate_tail_question(...)    → 꼬리질문 출력
  get_user_input()               → 답변 보완
  analysis.predict(q, new_a)    → 재분석
     │
     ▼
storage.save_session()           → JSON 저장
```

### CLI 필수 기능

- [ ] 무작위 질문 1개 출력
- [ ] 사용자 답변 입력 받기
- [ ] 분석 결과 출력 (intent_score + suitable_job)
- [ ] 꼬리질문 생성 및 출력
- [ ] 답변 수정 후 재분석 루프
- [ ] 최종 저장 (JSON, 적합 직무 TOP3 포함)
- [ ] 세션 종료 처리

### 산출물
```
cli/
├── main.py
├── agent.py
├── tools.py
└── storage.py
```

---

## STEP 6. 평가 및 지표 산출

### 작업 내용

1. **정량 평가 자동화 스크립트 작성**
   - BLEU-4, ROUGE-L, 코사인 유사도
   - MAE, RMSE, Pearson r (intent_score)
   - Accuracy, F1, Precision, Recall, Confusion Matrix (suitable_job)

2. **정성 평가 (샘플 30개)**
   - 경험 유도력, 역량 적합성, 자연스러움 (1~5점)

3. **결과 정리**: 각 stage별 평가 결과 표 + 기준값 달성 여부 체크

### 산출물
```
notebooks/
└── 05_evaluation.ipynb
evaluation/
└── results.md
```

---

## STEP 7. 팀 비교 및 최종 통합

### 비교 항목

| 항목 | 내용 |
|------|------|
| 장점 | 잘 된 점 |
| 단점 | 부족한 점 |
| 꼬리질문 품질 | 사용자를 얼마나 잘 유도하는가 |
| 대화 자연스러움 | CLI 흐름이 매끄러운가 |
| 저장 구조 | 추후 활용하기 편한가 |

**결정 기준**: 가장 적절한 꼬리질문을 통하여 사용자를 위한 좋은 답변을 이끌어낸 것

### 산출물
```
docs/
└── comparison.md
```

---

## 디렉토리 구조 (전체)

```
test3/
├── PLAN.md
├── USER_FLOW_PLAN.md
├── data/
│   ├── raw/
│   ├── train.csv
│   ├── test.csv
│   ├── intent_train.csv / intent_test.csv
│   ├── job_train.csv / job_test.csv
│   └── tail_train.csv / tail_test.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_stage1_question_gen.ipynb
│   ├── 03_stage3_multitask.ipynb
│   ├── 04_stage4_tail_question.ipynb
│   └── 05_evaluation.ipynb
├── models/
│   ├── stage1_question_gen/
│   ├── stage3_multitask/
│   └── stage4_tail_question/
├── cli/
│   ├── main.py
│   ├── agent.py
│   ├── tools.py
│   └── storage.py
├── evaluation/
│   └── results.md
└── docs/
    └── comparison.md
```

---

## 타임라인

| 주차 | 작업 |
|------|------|
| 1주차 | STEP 0 - 데이터 수집 (분담) |
| 2주차 | STEP 1 - EDA / STEP 2 - 질문 생성 모델 학습 시작 |
| 3주차 | STEP 3 - Multi-task / STEP 4 - 꼬리질문 모델 학습 |
| 4주차 | STEP 5 - CLI 구축 (개인별) + GitHub Push |
| 5주차 | STEP 6 - 평가 / STEP 7 - 비교 및 통합 |

> **주말 내로 개인 CLI GitHub Push 필수**

---

## 이 프로젝트에서 얻어갈 것

1. Multi-task Fine-tuning 경험
2. NLP 모델 평가 경험 (BLEU, ROUGE, Regression, Classification)
3. CLI 구축 경험 (LangChain Agent)
4. 데이터 수집 및 구축 경험
5. 팀 협업 기반 결정 경험 (여러 안 → 장단점 분석 → 최종 솔루션)
6. (가능 시) AI Agent 구조 설계 경험
