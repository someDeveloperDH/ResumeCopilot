# 🧭 하루 1문답 — AI Agentic 자기소개서 코칭 CLI

> IT/AI 직무 취업준비생을 위한 **Orchestrator-Subagent 기반** 자기소개서 셀프 트레이닝 CLI
> 매일 질문 1개 → AI 분석(intent_score/적합 직무) → STAR 코칭 → 꼬리질문 인터뷰 → 일관성 검사 → 세션 저장

---

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [팀원 및 역할](#팀원-및-역할)
- [서비스 아키텍처](#️-서비스-아키텍처)
- [폴더 구조](#-폴더-구조)
- [실행 방법](#️-실행-방법)
- [모델 평가](#-모델-평가)
- [참고 자료](#-참고-자료)

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 대상 | IT/AI 직무를 준비하는 대학교 1학년 ~ 장기 취준생 |
| 범위 | IT/AI 직무 한정 (`backend` / `ai_ml` / `product` 3개 트랙) |
| 방식 | CLI 기반 1인 세션, 로컬 실행 (CPU/GPU 자동 분기) |
| 핵심 아이디어 | 5개의 역할별 AI 서브에이전트(`AgentResult[T]` 공통 계약)가 질문 생성 → 분석 → STAR 코칭 → 꼬리질문 인터뷰 → 일관성 검사를 순서대로 수행 |

자세한 문제 배경 / 해결 방식 / 평가 설계는 [`represent_doc/PROJECT_REPORT.md`](./represent_doc/PROJECT_REPORT.md) 참고.

---

## 👥 팀원 및 역할

> ※ 아래 표는 초안이며, 팀원 본인이 직접 작성/검토 예정

| 구성원 | 역할 |
|---|---|
| A. 박동혁 | 아키텍처 설계 · 데이터 구축 · 베이스라인 구축 |
| B. 안진영 | 아이디어 설계 및 고도화 · CLI 구축 · 모델 설정 |
| C. 이진태 | CLI 구축 및 고도화 · 모델 설정 · 평가 설계 |

---

## 🏛️ 서비스 아키텍처

### 데이터 · 학습 파이프라인

```
[Linkareer 자기소개서 게시물]
    │  crawler.py / crawler_stage3.py
    ▼
[원본 CSV] ──merge.py──▶ stage1_train/test.csv ──▶ Stage1 학습 ──▶ stage1_question_gen/models/best(_cpu)
    │
    ├──▶ stage3_train/test.csv (intent_score 라벨링) ──▶ Stage3 학습 ──▶ stage3_multitask/models/best(_cpu)
    │
    └──▶ tail_train/test.csv (구간별 꼬리질문 라벨링) ──▶ Stage4 학습 ──▶ stage4_tail_question/models/best(_cpu)
```

### 런타임(CLI) — Orchestrator-Subagent

```
cli/main.py : run_session()  (Orchestrator, Session 상태 관리)
   │
   ├─ QuestionAgent      (light)              — 오늘의 질문 생성
   ├─ AnalysisAgent      (specialized→rule)   — intent_score / suitable_job 예측
   ├─ ConsistencyAgent   (specialized)        — rule + NLI + SBERT 일관성 검사
   ├─ CoachAgent         (smart→rule)         — STAR 추출(analyze) / 재작성(rewrite)
   └─ InterviewerAgent   (smart→light→rule)   — 꼬리질문 생성

   "smart" = EEVE-Korean-Instruct-10.8B (Ollama 로컬 서빙)
   모든 에이전트는 AgentResult(data, tier_used, fallback_used) 반환
   │
   ▼
storage/drafts/*_conversation.txt  (사용자 입력)
storage/sessions/*.json, *_final.txt  (결과 저장)
```

서브에이전트별 고정 모델 티어:

| 서브에이전트 | 책임 | 고정 티어 | 폴백 |
|---|---|---|---|
| `QuestionAgent` | 오늘의 질문 생성 | light (CPU: KoGPT2 `stage4_tail_question/models/best_cpu` / GPU: Llama-3.1-8B-4bit — Stage1+Stage4 통합 학습 체크포인트, `InterviewerAgent`와 공유) | 모델 로드 실패 시 → 샘플 질문 |
| `InterviewerAgent` | 꼬리질문 생성 | smart (EEVE-Korean-10.8B) | light → rule |
| `CoachAgent` | STAR 추출/재작성 | smart (EEVE-Korean-10.8B) | rule |
| `AnalysisAgent` | intent_score/적합 직무 | specialized (KoELECTRA 멀티태스크) | rule 보정 |
| `ConsistencyAgent` | 일관성 검사 | specialized (rule+NLI+SBERT) | (내부 단계적 폴백) |

---

## 📂 폴더 구조

```text
test4/
├── README.md
├── requirements.txt
│
├── crawler/                    # 데이터 수집/병합
│   ├── crawler.py              # Stage1: 질문 데이터 수집
│   ├── crawler_stage3.py       # Stage3: 질문+답변+직무 데이터 수집
│   └── merge.py                # 수집 데이터 병합
│
├── data/
│   ├── stage1/{train,test}.csv     # question_id, question, url, competency (500 / 100건)
│   ├── stage3/{train,test}.csv     # question, answer, intent_score, suitable_job, url (300 / 75건)
│   └── stage4/{tail_train,tail_test}.csv  # + tail_question (300 / 75건)
│
├── stage1_question_gen/        # 질문 생성 모델 (KoGPT2 CPU / Llama-3.1 LoRA GPU)
│   ├── config.yaml
│   ├── train.py
│   └── evaluate.py              # --baseline 옵션 지원
│
├── stage3_multitask/            # 분석 모델 (KoELECTRA / ModernBERT 멀티태스크)
│   ├── config.yaml
│   ├── train.py
│   └── evaluate.py
│
├── stage4_tail_question/        # 꼬리질문(+질문) 통합 생성 모델
│   ├── config.yaml
│   ├── train.py
│   └── evaluate.py
│
├── cli/
│   ├── main.py                  # Orchestrator: run_session()
│   ├── config.yaml               # agents: 섹션 — 역할별 모델 티어 고정 배정
│   ├── models.py                  # GPU/CPU 자동 분기 모델 로딩/추론
│   ├── coach.py / consistency.py / readiness.py / memory.py
│   ├── session.py / storage.py
│   └── agents/                    # 5개 서브에이전트 (AgentResult[T] 계약)
│       ├── base.py
│       ├── question_agent.py
│       ├── interviewer_agent.py
│       ├── coach_agent.py
│       ├── analysis_agent.py
│       └── consistency_agent.py
│
├── represent_doc/                # 발표용 정리 자료
│   ├── README.md
│   └── PROJECT_REPORT.md
│
└── storage/
    ├── sessions/                 # 세션 JSON / *_final.txt
    └── drafts/                   # 답변 작성용 conversation.txt
```

---

## ▶️ 실행 방법

### 1) 환경 설정

```bash
# conda 환경 생성
conda create -n jobfit python=3.11 -y
conda activate jobfit

# PyTorch (CUDA 버전에 맞게)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth (GPU LoRA 학습/추론용)
pip install unsloth

# 나머지 의존성
pip install -r requirements.txt

# (선택) 크롤러 의존성
pip install playwright selenium webdriver-manager
playwright install chromium
```

### 2) "smart" 코칭/인터뷰 모델 등록 (EEVE-Korean-10.8B, Ollama)

```bash
# Modelfile 예시: FROM /path/to/EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf
ollama create eeve-korean-10.8b -f Modelfile
ollama list   # eeve-korean-10.8b:latest 확인
```

`cli/config.yaml`의 `models.smart.model`이 위 이름과 일치해야 합니다.

### 3) 데이터 수집 & 병합

```bash
python crawler/crawler.py --output-dir ./data/stage1
python crawler/crawler_stage3.py --output-dir ./data/stage3 --label-method keyword
python crawler/merge.py
```

### 4) Stage1 / Stage3 / Stage4 학습 & 평가

```bash
# Stage1 — 질문 생성
cd stage1_question_gen && python train.py && python evaluate.py --baseline && python evaluate.py

# Stage3 — 분석 (intent_score + suitable_job)
cd ../stage3_multitask && python train.py && python evaluate.py --baseline && python evaluate.py

# Stage4 — 꼬리질문(+질문) 통합 생성
cd ../stage4_tail_question && python train.py && python evaluate.py --baseline && python evaluate.py
```

### 5) CLI 실행

```bash
cd ..   # test4/ 루트로 이동 (cli/config.yaml 상대경로 기준)

python cli/main.py            # 새 세션: 질문 → 답변 → AI 분석 → STAR 코칭 → 꼬리질문 인터뷰 → 저장
python cli/main.py --history   # 저장된 세션 기록 조회
```

> `cli/main.py` 상단 `USE_STAGE14_FINETUNED` / `USE_STAGE3_FINETUNED` 플래그가 `True`여야 학습된 체크포인트(`models/best`, `models/best_cpu`)가 사용됩니다.

---

## 📈 모델 평가

### Stage 1 — 질문 생성

- **방법**: `competency → question`. 동일 역량(competency)의 정답 질문들과 생성 질문 간 SBERT **Cosine** 유사도 최대값의 평균을 측정하고, `valid_threshold(0.70)` 이상인 비율을 **valid_ratio**로 측정. (BLEU-4/ROUGE-L/composite은 현재 `evaluate.py`에서 계산하지 않음)

| 모드 | Cosine | valid_ratio |
|---|---|---|
| baseline | 0.485 | 0.00 |
| finetuned | 0.671 | 0.54 |


### Stage 3 — 분석 (intent_score / suitable_job)

- **방법**: `(question, answer) → (intent_score 회귀, suitable_job 분류)`. 회귀는 MAE/RMSE, 분류는 Accuracy/F1-macro(+ Confusion Matrix, 콘솔 출력)로 평가. (Pearson r/Precision/Recall은 현재 `evaluate.py`에서 계산하지 않음)

| 모드 | MAE | RMSE | Accuracy | F1-macro |
|---|---|---|---|---|
| baseline | 3.11 | 3.82 | 0.267 | 0.140 |
| finetuned | 7.55 | 9.40 | 0.533 | 0.533 |



### Stage 4 — 꼬리질문(+질문) 통합 생성

- **방법**: `(question, answer, intent_score, suitable_job) → tail_question`. 정답 꼬리질문과의 SBERT **Cosine** 유사도, `open_question_ratio`(Yes/No 불가 비율)로 인터뷰 효과를 측정. (BLEU-4/ROUGE-L은 현재 `evaluate.py`에서 계산하지 않음)

| 모드 | Cosine | open_question_ratio |
|---|---|---|
| baseline | 0.247 | 0.373 |
| finetuned | 0.539 | 1.000 |


---

## 📄 참고 자료

- [`represent_doc/PROJECT_REPORT.md`](./represent_doc/PROJECT_REPORT.md) — 문제 배경, 해결 방식, 사용자 플로우, 데이터 수집/가공, 전체 아키텍처, 기술 스택, 단계별 모델 설계/평가, 핵심 기능, 시연 재현, 문제점/한계점/향후 계획 전체 정리
- [`MERGE_NOTES.md`](./MERGE_NOTES.md) — 3개 브랜치(test3/jintae_v2/jinyoung_v2) 통합 내역
