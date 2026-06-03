# 하루 1 문답 - 구현 가이드

> 지금까지 구현된 내용, 실행 순서, 모델 교체 방법 정리

---

## 디렉토리 구조

```
test3/
├── DEV_GUIDE.md                          ← 이 파일
├── PLAN.md                               ← 기술/모델 중심 계획
├── USER_FLOW_PLAN.md                     ← 유저 플로우 중심 계획
├── requirements.txt
│
├── data/
│   ├── raw/                              ← 원천 수집 데이터 보관
│   ├── stage1/
│   │   ├── train.csv                     ← question_id, competency, question
│   │   └── test.csv                      ← question_id, competency, question, generated_question, is_valid
│   ├── stage3/
│   │   ├── train.csv                     ← question, answer, intent_score, suitable_job
│   │   └── test.csv                      ← + generated_intent_score, generated_suitable_job
│   └── stage4/
│       ├── tail_train.csv                ← question, answer, intent_score, suitable_job, tail_question
│       └── tail_test.csv                 ← + generated_tail_question
│
├── stage1_question_gen/
│   ├── train.py                          ← Qwen3-8B LoRA 학습
│   ├── evaluate.py                       ← BLEU-4 / ROUGE-L / Cosine 평가
│   └── models/best/                      ← 학습 후 자동 생성
│       ├── adapter_model.safetensors
│       ├── best_lora.pt                  ← evaluate.py에서 직접 로드
│       └── train_config.json
│
├── stage3_multitask/
│   ├── train.py                          ← ModernBERT Multi-task 학습
│   ├── evaluate.py                       ← MAE/RMSE/Pearson r + Accuracy/F1 평가
│   └── models/best/
│       ├── best_model.pt
│       └── train_config.json
│
├── stage4_tail_question/
│   ├── train.py                          ← Qwen3-8B Stage1+4 통합 학습
│   ├── evaluate.py                       ← BLEU-4 / ROUGE-L / Cosine + 열린 질문 검사
│   └── models/best/
│       ├── best_lora.pt
│       └── train_config.json
│
├── cli/
│   ├── main.py                           ← 전체 플로우 진입점
│   ├── session.py                        ← 세션 상태 관리
│   └── storage.py                        ← 로컬 JSON 저장소
│
└── storage/
    └── sessions/                         ← 사용자 세션 JSON 파일들
```

---

## 환경 설치

```bash
# 1. PyTorch 먼저 설치 (CUDA 버전 확인 후)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Unsloth 설치
pip install unsloth
# 또는 최신 버전
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# 3. 나머지 패키지
pip install -r requirements.txt
```

---

## 실행 순서

### STEP 1. 데이터 수집 후 CSV 채우기

수집처: 잡코리아 / 원티드 / 블라인드 / 링크드인 IT·AI 직무 공고

| 파일 | 필수 컬럼 | 목표 수량 |
|------|-----------|-----------|
| `data/stage1/train.csv` | question_id, competency, question | 300~500개 |
| `data/stage1/test.csv` | question_id, competency, question | 75~100개 |
| `data/stage3/train.csv` | question, answer, intent_score, suitable_job | 60~90개 |
| `data/stage3/test.csv` | question, answer, intent_score, suitable_job | 20~30개 |
| `data/stage4/tail_train.csv` | question, answer, intent_score, suitable_job, tail_question | 300개 |
| `data/stage4/tail_test.csv` | question, answer, intent_score, suitable_job, tail_question | 75개 |

> **competency 종류**: 문제해결 / 협업 / 성장 / 실패경험 / 주도성  
> **suitable_job 종류**: backend / ai_ml / product  
> **intent_score 범위**: 0~100 (사람이 직접 채점)

---

### STEP 2. Stage 1 질문 생성 모델 학습

```bash
cd stage1_question_gen
python train.py
```

**완료 후 생성 파일**
```
models/best/
├── adapter_model.safetensors
├── best_lora.pt          ← evaluate.py에서 로드
└── train_config.json
```

---

### STEP 3. Stage 1 평가

```bash
cd stage1_question_gen
python evaluate.py
```

**평가 지표 기준값**

| 지표 | 최소 통과 | 목표 | 설명 |
|------|-----------|------|------|
| BLEU-4 | > 0.15 | > 0.35 | 보조 지표 (한국어 특성상 낮게 나올 수 있음) |
| ROUGE-L | > 0.30 | > 0.40 | LCS 기반 유사도 |
| Cosine 유사도 | > 0.75 | > 0.80 | **메인 지표** - 의미적 유사도 |

> 완료 후 `data/stage1/test.csv`의 `generated_question`, `is_valid` 컬럼이 채워짐

---

### STEP 4. Stage 3 Multi-task 모델 학습

```bash
cd stage3_multitask
python train.py
```

**완료 후 생성 파일**
```
models/best/
├── best_model.pt
└── train_config.json
```

---

### STEP 5. Stage 3 평가

```bash
cd stage3_multitask
python evaluate.py
```

**평가 지표 기준값**

| 태스크 | 지표 | 기준값 |
|--------|------|--------|
| intent_score (회귀) | MAE | ≤ 10 |
| intent_score (회귀) | RMSE | ≤ 15 |
| intent_score (회귀) | Pearson r | ≥ 0.80 |
| suitable_job (분류) | Accuracy | > 0.75 |
| suitable_job (분류) | F1-score (macro) | > 0.70 |
| suitable_job (분류) | Precision (macro) | > 0.70 |
| suitable_job (분류) | Recall (macro) | > 0.70 |

> 완료 후 `data/stage3/test.csv`의 `generated_intent_score`, `generated_suitable_job` 컬럼이 채워짐

---

### STEP 6. Stage 4 꼬리질문 모델 학습 (Stage 1+4 통합)

> **주의**: stage4 데이터 + stage1 데이터를 함께 학습. stage1의 best_lora.pt를 대체함.

```bash
cd stage4_tail_question
python train.py
```

**완료 후 생성 파일**
```
models/best/
├── best_lora.pt
└── train_config.json
```

---

### STEP 7. Stage 4 평가

```bash
cd stage4_tail_question
python evaluate.py
```

**평가 지표 기준값**

| 지표 | 기준값 | 설명 |
|------|--------|------|
| BLEU-4 | > 0.35 | |
| ROUGE-L | > 0.40 | |
| Cosine 유사도 | > 0.80 | |
| 열린 질문 비율 | > 80% | Yes/No로 답할 수 없는 질문 비율 |

---

### STEP 8. CLI 실행

```bash
# 하루 1 문답 시작
python cli/main.py

# 저장된 이력 조회
python cli/main.py --history
```

---

## 모델 교체 방법 (cli/main.py)

학습 완료 후 `cli/main.py` 상단의 두 플래그를 변경:

```python
# 기본값 (학습 전)
USE_STAGE14_FINETUNED = False   # base Qwen3-8B 사용
USE_STAGE3_FINETUNED  = False   # 분석 결과 stub 사용

# stage3 학습 완료 후
USE_STAGE3_FINETUNED  = True    # stage3_multitask/models/best 로드

# stage4 학습 완료 후 (stage1도 함께 대체됨)
USE_STAGE14_FINETUNED = True    # stage4_tail_question/models/best 로드
```

---

## CLI 전체 플로우

```
python cli/main.py
        │
        ▼
  models.load_generation_model()   Qwen3-8B LoRA
  models.load_analysis_model()     ModernBERT
        │
        ▼
[DEV-1] 질문 생성  generate_question(competency)
        │
        ▼
[DEV-2] 답변 입력  get_user_answer()  (멀티라인, 최소 30자)
        │
        ▼
[DEV-3] 분석       analysis.predict(q, a)
        │           의도 반영도 ██████░░░░ 63/100
        │           적합 직무   backend
        ▼
[DEV-4] 꼬리질문 루프 (최대 5회)
        │   generate_tail_question(...)  → 꼬리질문
        │   사용자 답변 입력
        │   analysis.predict(q, new_a)  → 재분석
        │   "저장하고 종료?" → y 입력 시 종료
        ▼
[DEV-5] 저장  storage.save_session()
              storage/sessions/{session_id}.json
```

---

## 세션 저장 형식 (storage/sessions/)

```json
{
  "session_id": "20240602_a1b2c3",
  "date": "2024-06-02",
  "competency": "문제해결",
  "question": "기술적 오류를 마주했을 때 어떻게 해결했는지 서술하세요.",
  "answer": "API 오류를 로그 분석으로...",
  "intent_score": 85.0,
  "suitable_jobs": ["backend", "ai_ml"],
  "conversation": [
    {
      "tail_question": "로그에서 원인을 찾기까지 어떤 순서로 접근했나요?",
      "response": "우선 에러 메시지를 확인하고..."
    }
  ],
  "final_answer": "우선 에러 메시지를 확인하고..."
}
```

---

## 주요 패키지 버전

| 패키지 | 버전 | 용도 |
|--------|------|------|
| unsloth | 최신 | Qwen3-8B LoRA 학습 가속 |
| trl | ≥ 0.8.6 | SFTTrainer |
| transformers | ≥ 4.47.0 | ModernBERT 지원 |
| sentence-transformers | ≥ 2.7.0 | Cosine 유사도 (KR-SBERT) |
| scipy | ≥ 1.13.0 | Pearson r 계산 |
| scikit-learn | ≥ 1.4.0 | 분류 지표 |

---

## CLI 파일 역할 분리

| 파일 | 역할 |
|------|------|
| `cli/main.py` | 전체 플로우 오케스트레이션만 담당 |
| `cli/models.py` | 모든 모델 로딩 및 추론 함수 |
| `cli/session.py` | 세션 상태 (dataclass) |
| `cli/storage.py` | JSON 저장/조회 |
