# 유저 플로우 중심 개발 단계 계획

> 하루 1 문답 - 사용자가 실제로 경험하는 흐름을 기준으로 개발 순서를 정의

---

## 유저 플로우 전체

```
[앱 시작]
    │
    ▼
[PHASE 1] 오늘의 질문 받기
    │  Qwen3-8B → competency 기반 질문 생성
    │
    ▼
[PHASE 2] 내 경험 작성
    │  사용자 자유 입력
    │
    ▼
[PHASE 3] AI 분석 결과 확인
    │  ModernBERT → intent_score + suitable_job 동시 예측
    │
    ▼
[PHASE 4] 꼬리질문과 대화하며 개선
    │  Qwen3-8B → 꼬리질문 생성 → 사용자 보완 → 반복
    │
    ▼
[PHASE 5] 저장
       JSON 저장소 → 적합 직무 TOP3 + 최종 답변 보관
```

---

## 개발 단계 (유저 플로우 순서 기준)

---

### DEV-1. 질문 제공 기능 (PHASE 1)

**사용자가 경험하는 것**: 앱을 켜면 오늘의 자소서 질문 1개가 나온다.

#### 개발 순서

**1-A. 질문 데이터 수집**
- competency 5종 (문제해결 / 협업 / 성장 / 실패경험 / 주도성) 기준 질문 수집
- 잡코리아, 원티드, 블라인드 등 IT/AI 직무 공고에서 수집
- CSV 형태로 저장: `data/train.csv`

```
question_id | competency | question
Q001        | 문제해결   | 개발 중 예상치 못한 기술적 오류를 마주했을 때...
Q002        | 협업       | 팀 프로젝트에서 의견 충돌이 발생했을 때...
```

**1-B. EDA 및 분포 확인**
- competency별 수량 균형 확인
- 중복 / 너무 짧은 질문 제거
- Instruction 포맷으로 변환

**1-C. 질문 생성 모델 학습 (Qwen3-8B + Unsloth LoRA)**
```
입력: [TASK:generate_question] competency: 문제해결
출력: 개발 중 예상치 못한 기술적 오류를 마주했을 때 어떻게 정의하고 해결했는지...
```
- 학습: LoRA Fine-tuning (~8GB VRAM)
- 평가: BLEU-4 > 0.35, ROUGE-L > 0.40, 코사인 유사도 > 0.80

**1-D. CLI 연결**
```python
# 실행 시 competency 랜덤 선택 → 모델이 질문 생성
question = model.generate(f"[TASK:generate_question] competency: {random_competency}")
print(f"\n오늘의 질문: {question}\n")
```

**완료 기준**: `python main.py` 실행 시 자소서 질문 1개가 출력된다.

---

### DEV-2. 답변 입력 기능 (PHASE 2)

**사용자가 경험하는 것**: 질문을 보고 자신의 경험을 자유롭게 입력한다.

#### 개발 순서

**2-A. CLI 입력 처리**
```python
print("답변을 입력하세요 (입력 완료: 빈 줄 후 Enter):")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
answer = "\n".join(lines)
```

**2-B. 입력값 검증**
- 최소 길이 검사 (너무 짧으면 재입력 요청)
- 세션 컨텍스트에 저장 (question + answer 쌍)

**완료 기준**: 사용자가 여러 줄 답변을 입력하고 저장된다.

---

### DEV-3. AI 분석 기능 (PHASE 3)

**사용자가 경험하는 것**: 내 답변이 질문의 의도를 얼마나 반영했는지, 어떤 직무에 어울리는지 점수/라벨로 나온다.

#### 개발 순서

**3-A. Multi-task 훈련 데이터 수집**

intent_score 데이터 (60~80개):
```
question | answer | intent_score
기술적 문제를 해결한 경험을 서술하세요 | API 오류를 로그 분석으로... | 72
```
- 구간별 균등: 0~40 (20개) / 41~70 (20개) / 71~100 (20개)
- 라벨링 기준: 경험 구체성, 질문 의도 부합도 직접 채점

suitable_job 데이터 (60~90개):
```
question | answer | suitable_job
데이터 분석 경험을 서술하세요 | pandas로 결측치 제거... | ai_ml
```
- 클래스별 균등: backend / ai_ml / product 각 20~30개

**3-B. EDA 및 전처리**
- intent_score 분포 시각화 (구간 쏠림 방지)
- 클래스 불균형 확인
- 입력 포맷: `[CLS] {question} [SEP] {answer} [SEP]`

**3-C. Multi-task 모델 학습 (ModernBERT-base)**

```
입력: [question] + [answer]
         ↓
   ModernBERT Encoder
         ↓
   ┌─────────────────┐
   │ Head 1 (회귀)   │ → intent_score (0~100)
   │ Head 2 (분류)   │ → suitable_job (backend/ai_ml/product)
   └─────────────────┘
```

**평가 지표 (intent_score)**
- MAE ≤ 10, RMSE ≤ 15, Pearson r ≥ 0.80

**평가 지표 (suitable_job)**
- Accuracy > 0.75, F1-macro > 0.70

**3-D. CLI 연결**
```python
result = analyzer.predict(question=question, answer=answer)
print(f"\n분석 결과")
print(f"  의도 반영도 : {result.intent_score}/100")
print(f"  적합 직무   : {result.suitable_job}")
```

**완료 기준**: 답변 입력 후 `intent_score`와 `suitable_job`이 출력된다.

---

### DEV-4. 꼬리질문 대화 기능 (PHASE 4)

**사용자가 경험하는 것**: AI가 내 점수와 직무에 맞는 질문을 던지고, 나는 답을 보완하며 반복한다.

#### 개발 순서

**4-A. 꼬리질문 훈련 데이터 수집 (300개)**

```
question | answer | intent_score | suitable_job | tail_question
기술적 문제를 해결한 경험 | API 오류를 로그 분석으로... | 72 | backend | 로그에서 원인을 찾기까지 어떤 순서로 접근했나요?
```

구간별 100개씩 (0~40 / 41~70 / 71~100)

**꼬리질문 규칙 (데이터 품질 기준)**
- Yes/No 불가
- 구체적 경험 유도
- 0~40점: 기본 상황 파악 → "그 상황에서 본인의 역할은 구체적으로?"
- 41~70점: 선택 근거 요구 → "그 방법을 선택한 이유는?"
- 71~100점: 사고 심화 → "이후 의사결정 방식에 어떤 영향을 줬나요?"

**4-B. 꼬리질문 모델 학습 (Qwen3-8B, DEV-1 모델과 동일 / Multi-task)**

```python
# DEV-1과 같은 Qwen3-8B 모델에 두 Task 동시 학습
입력: [TASK:tail_question] question: {q} answer: {a} intent_score: {s} job: {j}
출력: 로그에서 원인을 찾기까지 어떤 순서로 접근했나요?
```

**평가 지표 (정량)**
- BLEU-4 > 0.35, ROUGE-L > 0.40, 코사인 유사도 > 0.80

**평가 지표 (정성)**
- 사고 유도력, intent_score 반영 여부, 직무 연관성

**4-C. 대화 루프 CLI 구현**

```
[꼬리질문 루프]

꼬리질문: 로그에서 원인을 찾기까지 어떤 순서로 접근했나요?
> (사용자 입력)

[재분석]
의도 반영도: 85/100 ↑
적합 직무: backend

다음 꼬리질문: 그 경험이 이후 디버깅 방식에 어떤 영향을 줬나요?
> (사용자 입력)

저장하시겠습니까? (y/n)
```

**루프 종료 조건**
- 사용자가 저장 선택
- 꼬리질문 최대 횟수 초과 (예: 5회)
- intent_score가 기준값 이상 도달 시 저장 유도

**4-D. CLI 통합 (직접 호출)**

```python
# cli/main.py - run_session()
gen_model, tokenizer = models.load_generation_model(use_finetuned)
analysis             = models.load_analysis_model(use_finetuned)

for turn in range(MAX_TAIL_TURNS):
    tail_q  = models.generate_tail_question(gen_model, tokenizer, ...)
    reply   = input()
    score, job = analysis.predict(question, reply)  # 재분석

# cli/models.py  — 모든 모델 코드 분리
# cli/session.py — 세션 상태 (대화 이력 포함)
# cli/storage.py — JSON 저장
```

**완료 기준**: 꼬리질문 → 답변 보완 → 재분석 루프가 자연스럽게 동작한다.

---

### DEV-5. 저장 기능 (PHASE 5)

**사용자가 경험하는 것**: 오늘 쓴 내용이 내 자소서 저장함에 저장되고, 나중에 꺼내볼 수 있다.

#### 개발 순서

**5-A. 저장 데이터 구조 설계**

```json
{
  "session_id": "20240602_001",
  "date": "2024-06-02",
  "question": "기술적 문제를 해결한 경험을 서술하세요",
  "competency": "문제해결",
  "final_answer": "API 오류를 로그 분석으로 해결했습니다...",
  "intent_score": 85,
  "suitable_jobs": ["backend", "ai_ml", "product"],
  "conversation_history": [
    {"role": "tail_question", "content": "로그에서 원인을 찾기까지..."},
    {"role": "user", "content": "우선 에러 메시지를 확인하고..."}
  ]
}
```

**5-B. 저장소 구현**

```python
class CoverLetterStorage:
    def save_session(self, session: dict) -> str
    def load_all(self) -> list[dict]
    def search_by_competency(self, competency: str) -> list[dict]
    def get_top_jobs(self) -> list[str]  # 전체 세션 기반 TOP3 집계
```

**5-C. CLI 종료 흐름**
```
저장하시겠습니까? (y/n): y

저장 완료!
  날짜: 2024-06-02
  적합 직무 TOP3: backend > ai_ml > product
  저장 위치: storage/sessions/20240602_001.json

오늘도 수고했습니다. 내일 또 써봐요!
```

**완료 기준**: 세션이 JSON으로 저장되고, 이전 기록을 불러올 수 있다.

---

## 통합 체크리스트

```
DEV-1. 질문 제공
  [ ] 데이터 수집 완료 (train.csv)
  [ ] EDA 완료
  [ ] Qwen3-8B LoRA 학습 완료
  [ ] CLI 연결 및 출력 확인

DEV-2. 답변 입력
  [ ] 멀티라인 입력 처리
  [ ] 최소 길이 검증

DEV-3. AI 분석
  [ ] intent + job 데이터 수집 완료
  [ ] ModernBERT Multi-task 학습 완료
  [ ] 평가 지표 기준값 통과
  [ ] CLI 분석 결과 출력 확인

DEV-4. 꼬리질문 대화
  [ ] 꼬리질문 데이터 수집 완료 (300개)
  [ ] Qwen3-8B [TASK:tail_question] 학습 완료
  [ ] 루프 로직 구현
  [ ] LangChain Agent 통합

DEV-5. 저장
  [ ] JSON 저장 구조 구현
  [ ] TOP3 직무 집계 로직
  [ ] 이전 기록 조회 기능
```

---

## 개인 CLI 개발 → 팀 비교 순서

```
[각자 DEV-1 ~ DEV-5 구현]
         ↓
[주말 내 GitHub Push]
         ↓
[3명이 서로의 CLI 사용해보기]
         ↓
[장단점 기록 (꼬리질문 품질 중심)]
         ↓
[최종 1개 선정 + 나머지에서 장점 조합]
         ↓
[최종 버전 Push]
```

**선정 기준**: 가장 적절한 꼬리질문을 통해 사용자의 좋은 답변을 이끌어낸 것
