# 하루 1문답 자기소개서 코칭 CLI 프로세스 README

이 프로젝트는 CPU 환경에서 실행되는 자기소개서 코칭 CLI입니다. 사용자의 최초 답변을 받고, 답변 분석 결과와 전체 대화 맥락을 바탕으로 꼬리질문을 생성합니다. 이후 각 답변 사이의 일관성을 `규칙 기반 + NLI contradiction + SBERT 유사도` 방식으로 검사하고, 세션 전체를 JSON과 TXT로 저장합니다.

---

## 1. 전체 구조

```text
project/
├─ cli/
│  ├─ main.py          # CLI 실행 흐름, 답변 입력, 일관성 검사, 저장 처리
│  ├─ models.py        # 생성 모델/분석 모델 로딩 및 추론 함수
│  ├─ session.py       # 한 세션의 상태 데이터 구조
│  └─ storage.py       # 세션 JSON 저장 및 history 출력
│
├─ stage1_question_generation/
│  ├─ train.py         # Stage1: 최초 질문 생성 모델 학습
│  ├─ evaluate.py      # Stage1 평가
│  └─ config.yaml
│
├─ stage3_multitask/
│  ├─ train.py         # Stage3: intent_score + suitable_job 분석 모델 학습
│  ├─ evaluate.py      # Stage3 평가
│  └─ config.yaml
│
├─ stage4_tail_question/
│  ├─ train.py         # Stage4: Stage1 + Stage4 통합 생성 모델 학습
│  ├─ evaluate.py      # Stage4 꼬리질문 평가
│  └─ config.yaml
│
└─ storage/
   ├─ drafts/          # 답변 작성용 conversation txt
   └─ sessions/        # 최종 JSON, 최종 TXT 저장
```

---

## 2. 학습 단계 개요

### 2.1 Stage1: 최초 질문 생성 모델

Stage1은 역량을 입력받아 자기소개서 질문을 생성하는 모델입니다.

```text
입력: competency
출력: 자기소개서 질문
```

예시:

```text
competency: 문제해결
→ 문제해결 역량을 발휘해 어려운 문제를 해결했던 경험을 구체적으로 작성해 주세요.
```

실행:

```bash
cd stage1_question_generation
python train.py
python evaluate.py
```

추천 설정:

```yaml
model:
  base_model: "skt/kogpt2-base-v2"
  max_seq_length: 256
  max_new_tokens: 80

train:
  epochs: 5
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 5.0e-5
  val_ratio: 0.1
  seed: 42
```

Stage1은 `epochs: 1`이면 실행 확인용에 가깝습니다. 질문 품질을 확인하려면 먼저 5 epoch로 학습하고, 부족하면 10 epoch까지 비교합니다.

---

### 2.2 Stage3: 답변 분석 모델

Stage3는 사용자의 답변을 보고 두 가지 값을 예측합니다.

```text
1. intent_score   # 질문 의도 반영도
2. suitable_job   # 적합 직무 분류
```

모델 구조는 encoder 기반 multi-task 모델입니다. 하나의 encoder 출력에서 회귀 head는 `intent_score`, 분류 head는 `suitable_job`을 예측합니다.

실행:

```bash
cd stage3_multitask
python train.py
python evaluate.py
```

추천 설정:

```yaml
train:
  epochs: 10
  batch_size: 1
  learning_rate: 2.0e-5
  val_ratio: 0.1
  seed: 42
  freeze_encoder: false
  num_workers: 0
```

Stage3 학습 결과가 없으면 CLI는 실제 분석 모델 대신 대체값을 사용합니다.

```text
intent_score = 50.0
suitable_job = backend / ai_ml / product 중 랜덤
```

따라서 CLI에서 의도 반영도와 적합 직무를 제대로 보고 싶으면 Stage3 학습을 먼저 완료해야 합니다.

---

### 2.3 Stage4: Stage1 + Stage4 통합 생성 모델

Stage4는 현재 CLI에서 가장 직접적으로 사용되는 생성 모델입니다. Stage1의 최초 질문 생성 형식과 Stage4의 꼬리질문 생성 형식을 함께 학습합니다.

학습 형식:

```text
[TASK:generate_question]
competency → question

[TASK:tail_question]
question + answer + intent_score + suitable_job → tail_question
```

실행:

```bash
cd stage4_tail_question
python train.py
python evaluate.py
```

추천 설정:

```yaml
model:
  base_model: "skt/kogpt2-base-v2"
  max_seq_length: 256
  max_new_tokens: 80

train:
  epochs: 5
  batch_size: 1
  learning_rate: 5.0e-5
  val_ratio: 0.1
  seed: 42
  max_samples: 0
```

`max_samples: 30`은 빠른 테스트용입니다. 제대로 학습하려면 `max_samples: 0`으로 전체 데이터를 사용합니다. `max_seq_length: 128`은 답변이 조금만 길어져도 잘릴 수 있으므로, 꼬리질문 생성에서는 256을 권장합니다.

---

## 3. 권장 학습 순서

```bash
# 1. 최초 질문 생성 모델 학습
cd stage1_question_generation
python train.py
python evaluate.py

# 2. 답변 분석 모델 학습
cd ../stage3_multitask
python train.py
python evaluate.py

# 3. Stage1 + Stage4 통합 생성 모델 학습
cd ../stage4_tail_question
python train.py
python evaluate.py

# 4. CLI 실행
cd ../cli
python main.py
```

현재 CLI의 생성 모델 로딩은 `stage4_tail_question/models/best_cpu`를 우선 사용합니다. 따라서 실제 CLI 질문/꼬리질문 품질에 가장 직접적인 영향을 주는 것은 Stage4 학습 결과입니다.

---

## 4. CLI 실행 명령

새 세션 실행:

```bash
python cli/main.py
```

저장된 세션 기록 확인:

```bash
python cli/main.py --history
```

---

## 5. CLI 전체 실행 흐름

```text
1. 모델 로딩
   ├─ 생성 모델 로딩
   │  ├─ stage4_tail_question/models/best_cpu 있으면 사용
   │  └─ 없으면 skt/kogpt2-base-v2 사용
   │
   └─ 분석 모델 로딩
      ├─ stage3_multitask/models/best_cpu 있으면 사용
      ├─ 없으면 stage3_multitask/models/best 확인
      └─ 없으면 분석 모델 없이 대체값 사용

2. 세션 생성
   ├─ session_id 생성
   ├─ 날짜 저장
   └─ 역량 랜덤 선택

3. 최초 질문 생성
   └─ Stage4 통합 생성 모델 또는 base KoGPT2 사용

4. conversation txt 파일 생성
   └─ storage/drafts/{session_id}_conversation.txt

5. 사용자가 txt 파일에 최초 답변 작성

6. 최초 답변 분석
   ├─ intent_score 계산
   └─ suitable_job 예측

7. 최초 답변 일관성 검사
   └─ 첫 답변은 비교 대상이 없으므로 보통 양호 처리

8. 꼬리질문 반복
   ├─ 전체 conversation_context 생성
   ├─ 직전 일관성 검사 결과를 consistency_note로 전달
   ├─ 꼬리질문 생성
   ├─ 같은 txt 파일에 [꼬리질문 n] / [답변 n] 추가
   ├─ 사용자가 [답변 n] 아래에 답변 작성
   ├─ 현재 답변 분석
   ├─ 현재 답변 일관성 검사
   └─ 저장하고 종료할지 확인

9. 세션 종료
   ├─ 적합 직무 변화 추적 출력
   ├─ JSON 저장
   └─ 최종 TXT 저장
```

기본 꼬리질문 최대 횟수는 `MAX_TAIL_TURNS = 5`입니다. 각 턴이 끝난 뒤 `저장하고 종료? (y/n)`에서 `y`를 입력하면 중간 종료됩니다.

---

## 6. CLI에서 추가 구현한 핵심 기능

### 6.1 메모장 기반 답변 입력 통합

기존에는 최초 답변과 꼬리답변의 입력 방식이 다를 수 있었습니다. 현재 CLI는 최초 답변과 모든 꼬리답변을 동일하게 하나의 txt 파일에서 입력받습니다.

생성 파일:

```text
storage/drafts/{session_id}_conversation.txt
```

파일 구조:

```text
[역량]
문제해결

[원본 질문]
...

[최초 답변]
여기에 최초 답변 작성

----------------------------------------
[꼬리질문 1]
...

[답변 1]
여기에 꼬리질문 답변 작성
```

입력 규칙:

```text
- 기존 내용은 지우지 않는다.
- 새 답변은 새 [답변 n] 아래에 작성한다.
- #으로 시작하는 줄은 답변에서 제외된다.
- 최초 답변은 최소 30자 이상이어야 한다.
- 꼬리질문 답변은 최소 10자 이상이어야 한다.
```

파일을 열 때는 Windows에서는 `notepad`를 우선 사용합니다. 그 외 환경에서는 `$EDITOR`, `nano`, `vi` 순서로 시도합니다.

---

### 6.2 전체 대화 맥락 누적

꼬리질문 생성 시 단순히 직전 답변만 사용하는 것이 아니라, 최초 질문과 지금까지의 모든 꼬리질문/답변을 하나의 `conversation_context`로 묶어 전달합니다.

누적 형식:

```text
[원본 질문] ...
[최초 답변] ...
[꼬리질문 1] ...
[답변 1] ...
[꼬리질문 2] ...
[답변 2] ...
...
```

이 구조 덕분에 꼬리질문 모델은 다음 내용을 반영할 수 있습니다.

```text
- 사용자가 이미 답한 내용
- 아직 부족한 설명
- 앞뒤 답변에서 충돌 가능성이 있는 부분
- 직무 분석 결과
- 직전 일관성 검사 메모
```

---

### 6.3 pairwise + global 일관성 검사

일관성 검사는 현재 답변 하나를 전체 통합본 하나와만 비교하지 않습니다. 현재 답변을 이전 답변 각각과 비교하고, 추가로 전체 이전 답변 묶음과도 비교합니다.

Pairwise 검사:

```text
현재 답변 vs 최초 답변
현재 답변 vs 꼬리답변 1
현재 답변 vs 꼬리답변 2
...
```

Global 검사:

```text
현재 답변 vs 전체 이전 답변 묶음
```

이렇게 나눈 이유는 다음과 같습니다.

```text
- 직전 답변만 보면 오래전 답변과의 충돌을 놓칠 수 있음
- 전체 통합본만 보면 어떤 답변과 충돌했는지 알기 어려움
- pairwise 검사로 충돌 위치를 찾고, global 검사로 전체 흐름을 확인함
```

---

### 6.4 규칙 기반 + NLI + SBERT 조합 검사

현재 CLI는 생성 모델에게 “모순이 있는지 판단해줘”라고 맡기지 않습니다. 대신 세 가지 기준을 조합합니다.

```text
1. 규칙 기반 검사
2. NLI contradiction 검사
3. SBERT cosine similarity 검사
```

#### 규칙 기반 검사

자기소개서 답변에서 자주 나오는 충돌 패턴을 검사합니다.

```text
리더 / 팀장 / 주도 / 총괄
↔ 결정권 없음 / 보조 / 시키는 일 / 기여 적음

혼자 / 단독 / 개인
↔ 팀 / 협업 / 함께 / 조원
```

예시:

```text
이전 답변: 제가 팀장으로 프로젝트를 주도했습니다.
현재 답변: 저는 결정권이 없었고 시키는 일만 했습니다.
→ 역할/권한 표현 충돌 가능성
```

#### NLI contradiction 검사

이전 답변과 현재 답변을 NLI 모델에 넣고 contradiction 확률을 계산합니다.

기본 모델:

```text
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

기본 기준:

```text
contradiction >= 0.65 → 모순 가능성 있음
contradiction >= 0.80 → 모순 가능성 큼
```

NLI는 방향성이 있으므로 두 방향을 모두 검사합니다.

```text
이전 답변 → 현재 답변
현재 답변 → 이전 답변
```

둘 중 contradiction이 더 높은 값을 최종 contradiction 점수로 사용합니다.

CPU에서 NLI가 너무 느리면 끌 수 있습니다.

PowerShell:

```powershell
$env:CONSISTENCY_USE_NLI="0"
python cli/main.py
```

다시 켜기:

```powershell
$env:CONSISTENCY_USE_NLI="1"
python cli/main.py
```

다른 NLI 모델을 쓰고 싶으면:

```powershell
$env:CONSISTENCY_NLI_MODEL="모델명"
python cli/main.py
```

#### SBERT 유사도 검사

SBERT는 모순 자체를 판단하기보다는, 현재 답변이 이전 답변의 같은 경험을 이어서 설명하는지 확인하는 보조 지표입니다.

기본 모델:

```text
snunlp/KR-SBERT-V40K-klueNLI-augSTS
```

기준:

```text
similarity < 0.35 → 맥락 이탈 가능성
similarity >= 0.55 + 충돌 이슈 있음 → 같은 경험 내 표현 충돌 가능성
```

---

### 6.5 일관성 검사 결과 저장 구조

각 턴의 일관성 검사 결과는 `session.consistency_checks`에 저장됩니다.

저장 예시:

```json
{
  "turn": 2,
  "status": "주의",
  "issues": [
    "최초 답변과 현재 답변 사이의 NLI contradiction 점수가 높아 모순 가능성이 큽니다."
  ],
  "suggestions": [
    "최초 답변에서 말한 내용과 현재 답변의 역할·권한·기여도 표현이 함께 성립하는지 확인해 주세요."
  ],
  "method": "pairwise_rule_nli_sbert",
  "pair_results": [
    {
      "source": "최초 답변",
      "similarity": 0.7633,
      "nli": {
        "contradiction": 0.8009,
        "direction": "current_to_previous"
      },
      "issues": ["..."]
    }
  ],
  "global_result": {
    "source": "전체 이전 답변",
    "similarity": 0.7000,
    "nli": {"contradiction": 0.9940},
    "issues": ["..."]
  }
}
```

이 구조를 보면 단순히 `주의`라고만 저장되는 것이 아니라, 어떤 이전 답변과 충돌했는지, SBERT 유사도는 얼마였는지, NLI contradiction 점수는 얼마였는지까지 확인할 수 있습니다.

---

### 6.6 적합 직무 변화 추적

각 턴마다 Stage3 분석 결과를 `job_history`에 저장합니다.

저장 내용:

```json
{
  "turn": 1,
  "source": "tail_1",
  "intent_score": 14.0,
  "job": "product"
}
```

세션 종료 시에는 시작 직무와 최종 직무, 의도 반영도 변화량, 세션 내 많이 등장한 직무를 출력합니다.

출력 예시:

```text
적합 직무 변화 추적
Turn  0 | initial    | product  | intent 7.8/100
Turn  1 | tail_1     | product  | intent 14.0/100
Turn  2 | tail_2     | product  | intent 9.5/100

시작 직무 → 최종 직무 : product → product
의도 반영도 변화       : 7.8 → 9.5 (+1.7)
세션 내 TOP 직무       : product
```

---

### 6.7 세션 단일 파일 저장

답변 입력은 하나의 conversation txt에 누적되고, 종료 시 JSON과 최종 TXT가 저장됩니다.

```text
storage/drafts/{session_id}_conversation.txt      # 사용자가 작성하는 원본 대화 파일
storage/sessions/{session_id}.json                # 구조화된 세션 데이터
storage/sessions/{session_id}_final.txt           # 사람이 읽기 좋은 최종 요약 파일
```

`JSON`에는 모델 분석 결과, 꼬리질문 이력, 일관성 검사 결과, 직무 변화 추적이 모두 저장됩니다. `final.txt`에는 사용자가 작성한 대화 내용과 최종 요약이 함께 저장됩니다.

---

## 7. CLI 내부 모듈 역할

### 7.1 `main.py`

CLI의 전체 흐름을 담당합니다.

```text
- 모델 로딩 호출
- 세션 생성
- 역량 선택
- 최초 질문 출력
- conversation txt 생성 및 답변 입력 처리
- 꼬리질문 반복
- 일관성 검사 실행
- 분석 결과 출력
- 직무 변화 추적 출력
- JSON/TXT 저장
```

추가 구현의 중심 파일입니다. 특히 다음 기능이 `main.py`에 포함됩니다.

```text
- txt 파일 기반 답변 입력
- conversation_context 생성
- pairwise/global 일관성 검사
- 규칙 기반 충돌 검사
- NLI contradiction 검사
- SBERT 유사도 검사
- job_history 출력
- final txt 저장
```

---

### 7.2 `models.py`

모델 로딩과 추론 함수를 담당합니다.

```text
- 생성 모델 로딩
- 분석 모델 로딩
- 최초 질문 생성
- 꼬리질문 생성
- 답변 분석 모델 래퍼
```

생성 모델 로딩 우선순위:

```text
1. stage4_tail_question/models/best_cpu
2. skt/kogpt2-base-v2
```

분석 모델 로딩 우선순위:

```text
1. stage3_multitask/models/best_cpu
2. stage3_multitask/models/best
3. 없으면 None 반환
```

---

### 7.3 `session.py`

한 번의 CLI 실행에서 생기는 세션 데이터를 관리합니다.

주요 필드:

```text
session_id
competency
question
answer
conversation
final_answer
intent_score
suitable_jobs
job_history
consistency_checks
conversation_context
```

꼬리질문 답변은 `conversation` 리스트에 저장되고, 직무 변화는 `job_history`, 일관성 검사 결과는 `consistency_checks`에 저장됩니다.

---

### 7.4 `storage.py`

세션 저장과 기록 조회를 담당합니다.

```text
save_session()   # 세션 JSON 저장
load_all()       # 저장된 JSON 전체 로딩
print_history()  # --history 실행 시 세션 요약 출력
get_top_jobs()   # 전체 세션 기준 TOP 직무 계산
```

---

## 8. 저장되는 파일

세션 종료 시 두 종류의 결과 파일이 저장됩니다.

### 8.1 JSON 저장

```text
storage/sessions/{session_id}.json
```

저장 내용:

```json
{
  "session_id": "...",
  "date": "...",
  "competency": "...",
  "question": "...",
  "answer": "...",
  "intent_score": 0.0,
  "suitable_jobs": [],
  "conversation": [],
  "final_answer": "...",
  "job_history": [],
  "consistency_checks": [],
  "conversation_context": "..."
}
```

### 8.2 최종 TXT 저장

```text
storage/sessions/{session_id}_final.txt
```

최종 TXT에는 사용자가 작성한 전체 대화 내용과 최종 요약이 함께 저장됩니다.

```text
[최종 요약]
날짜:
역량:
최종 적합 직무:
최종 의도 반영도:

[적합 직무 변화 추적]
Turn 0 | initial | ...
Turn 1 | tail_1  | ...
...
```

---

## 9. 전체 프로세스 요약

```text
Stage1 학습
→ 최초 질문 생성 능력 학습

Stage3 학습
→ 답변의 intent_score와 suitable_job 예측 능력 학습

Stage4 학습
→ 최초 질문 생성 + 꼬리질문 생성 능력 통합 학습

CLI 실행
→ 질문 생성
→ txt 파일로 답변 입력
→ 답변 분석
→ 규칙 기반 + NLI + SBERT 일관성 검사
→ 전체 대화 맥락 기반 꼬리질문 생성
→ 답변/분석/검사를 반복
→ 직무 변화 추적
→ JSON 및 TXT 저장
```
