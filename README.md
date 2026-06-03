# 하루 1 문답 — 자소서 훈련기

> IT/AI 직무 취준생을 위한 자기소개서 작성 훈련 CLI  
> 매일 질문 하나를 받아 답변을 쓰고, AI 피드백으로 개선하며 나만의 자소서 소스를 쌓는 도구

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 대상 | 대학교 1학년 ~ 장기 취준생 |
| 범위 | IT / AI 관련 직무 한정 |
| 방식 | CLI 기반 대화형 자소서 훈련 |
| 기간 | 2025년 4월 ~ 6월 |
| 팀원 | 안진영 · 이진태 · 박동혁 |

### 핵심 플로우

```
1. 오늘의 질문 제공  (Stage 1 모델 — 질문 생성)
        ↓
2. 사용자 답변 입력  (txt 파일 → 에디터)
        ↓
3. AI 분석           (Stage 3 모델 — 의도 반영도 + 적합 직무)
        ↓
4. 꼬리질문 대화     (Stage 4 모델 — 꼬리질문 생성)
        ↓
5. 저장              (JSON → storage/sessions/)
```

---

## 디렉토리 구조

```
test3/
├── README.md
├── requirements.txt
├── .gitignore
│
├── crawler.py               # Stage 1 데이터 수집 크롤러 (질문)
├── crawler_stage3.py        # Stage 3 데이터 수집 크롤러 (질문+답변+직무)
├── merge.py                 # 수집 데이터 병합 스크립트
│
├── data/
│   ├── stage1/
│   │   ├── train.csv        # 질문 생성 학습 데이터 (question_id, competency, question)
│   │   └── test.csv         # 질문 생성 평가 데이터
│   ├── stage3/
│   │   ├── train.csv        # Multi-task 학습 데이터 (question, answer, intent_score, suitable_job)
│   │   └── test.csv
│   └── stage4/
│       ├── tail_train.csv   # 꼬리질문 학습 데이터
│       └── tail_test.csv
│
├── stage1_question_gen/     # 질문 생성 모델 (Llama-3.1 LoRA)
│   ├── config.yaml          # 모델/파라미터 설정
│   ├── train.py             # 학습
│   └── evaluate.py          # 평가 (--baseline 옵션 지원)
│
├── stage3_multitask/        # 분석 모델 (ModernBERT Multi-task)
│   ├── config.yaml
│   ├── train.py
│   └── evaluate.py          # 평가 (--baseline 옵션 지원)
│
├── stage4_tail_question/    # 꼬리질문 생성 모델 (Llama-3.1 LoRA, Stage1+4 통합)
│   ├── config.yaml
│   ├── train.py
│   └── evaluate.py          # 평가 (--baseline 옵션 지원)
│
├── cli/
│   ├── main.py              # 전체 플로우 진입점
│   ├── models.py            # 모델 로딩 및 추론
│   ├── session.py           # 세션 상태 관리
│   └── storage.py           # JSON 저장소
│
└── storage/
    ├── sessions/            # 사용자 세션 저장 (.gitignore)
    └── drafts/              # 답변 임시 파일 (.gitignore)
```

---

## 환경 설정

### 1. PyTorch 설치 (CUDA 버전에 맞게)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Unsloth 설치

```bash
pip install unsloth
# 또는 최신 버전
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

### 3. 나머지 패키지

```bash
pip install -r requirements.txt
```

### 4. 크롤러용 (선택)

```bash
pip install playwright selenium webdriver-manager
playwright install chromium
```

---

## 사용 모델

| Stage | 모델 | 방식 | VRAM |
|-------|------|------|------|
| Stage 1 (질문 생성) | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | LoRA Fine-tuning | ~9GB |
| Stage 3 (분석) | `answerdotai/ModernBERT-base` | Full Fine-tuning | ~4GB |
| Stage 4 (꼬리질문) | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | LoRA Fine-tuning | ~10GB |

> 모델 교체: 각 stage의 `config.yaml` 에서 `model.base_model` 또는 `model.name` 만 변경

---

## 실행 순서

### Step 1. 데이터 수집

```bash
# Stage 1용 질문 데이터 수집
python crawler.py --output-dir ./data/stage1

# Stage 3용 질문+답변 데이터 수집
python crawler_stage3.py --output-dir ./data/stage3 --label-method keyword

# 중단 후 재개
python crawler.py --start-id 34000 --output-dir ./data/stage1

# 수집 현황 확인
python crawler.py --status --output-dir ./data/stage1
```

### Step 2. Stage 1 — 질문 생성 모델 학습

```bash
cd stage1_question_gen

# 학습
python train.py

# 베이스라인 평가 (파인튜닝 전)
python evaluate.py --baseline

# 파인튜닝 후 평가
python evaluate.py
```

**평가 지표**

| 지표 | 최소 기준 | 목표 |
|------|----------|------|
| BLEU-4 | > 0.15 | > 0.35 |
| ROUGE-L | > 0.30 | > 0.40 |
| Cosine 유사도 | > 0.75 | > 0.80 |
| 통합 점수 (가중 평균) | > 0.60 | — |

### Step 3. Stage 3 — Multi-task 분석 모델 학습

```bash
cd stage3_multitask

python train.py

python evaluate.py --baseline   # 파인튜닝 전
python evaluate.py              # 파인튜닝 후
```

**평가 지표**

| 태스크 | 지표 | 기준 |
|--------|------|------|
| intent_score (회귀) | MAE | ≤ 10 |
| intent_score (회귀) | Pearson r | ≥ 0.80 |
| suitable_job (분류) | Accuracy | > 0.75 |
| suitable_job (분류) | F1-macro | > 0.70 |

### Step 4. Stage 4 — 꼬리질문 생성 모델 학습

```bash
cd stage4_tail_question

python train.py

python evaluate.py --baseline
python evaluate.py
```

### Step 5. CLI 실행

```bash
# 모델 교체 (cli/main.py 상단 플래그 변경)
# USE_STAGE14_FINETUNED = True   ← stage4 학습 완료 후
# USE_STAGE3_FINETUNED  = True   ← stage3 학습 완료 후

# 하루 1 문답 시작
python cli/main.py

# 저장 이력 조회
python cli/main.py --history
```

---

## config.yaml 설명

각 stage 폴더에 `config.yaml` 파일이 있으며, 이 파일만 수정하면 모델 교체 및 파라미터 변경이 가능합니다.

```yaml
model:
  base_model: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # ← 모델 교체 시 여기만 변경

lora:
  r: 16          # LoRA rank
  alpha: 16      # LoRA scaling

train:
  epochs: 3
  batch_size: 2
  learning_rate: 2.0e-4
```

---

## CSV 컬럼 구조

| 파일 | 컬럼 |
|------|------|
| `data/stage1/train.csv` | question_id, competency, question |
| `data/stage3/train.csv` | question, answer, intent_score, suitable_job, url |
| `data/stage4/tail_train.csv` | question, answer, intent_score, suitable_job, tail_question |

> `intent_score`: 학습 데이터 수집 단계에서 비워두고 별도 라벨링 작업으로 채움  
> `suitable_job`: backend / ai_ml / product 중 하나
