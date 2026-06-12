# test4 통합 노트 — 어디서 무엇을 가져왔는가

`test4`는 `test3`(베이스), `jintae_v2`(Ollama 코칭 강화판), `jinyoung_v2`(경량 한국어 모델 +
일관성 검사판) 세 브랜치를 "짜깁기"가 아니라 검증된 기능 단위로 골라 하나의 흐름으로
재구성한 통합본이다. 아래는 파일/기능 단위로 출처와 선택 이유를 정리한 표다.

## 1. 학습 파이프라인 (stage1/3/4, crawler, merge)

| 항목 | 출처 | 이유 |
|---|---|---|
| `stage1_question_gen/`, `stage3_multitask/`, `stage4_tail_question/`, `crawler*.py`, `merge.py` | **jinyoung_v2** 그대로 복사 | 세 브랜치의 학습 코드는 거의 동일하지만, jinyoung_v2 쪽이 `best_cpu` 체크포인트 디렉터리·CPU 친화적 `train_config.json` 옵션까지 포함하고 있어 GPU/CPU 양쪽 환경에 가장 범용적이었음 |

## 2. CLI 핵심 모듈

| 파일 | 출처 | 가져온 내용 / 이유 |
|---|---|---|
| `cli/coach.py` | **jintae_v2** 그대로 이식 | `CoachAnalysis` 데이터클래스, STAR 추출/재작성 프롬프트 빌더(`build_extraction_prompt`/`build_rewrite_prompt`), `rule_analyze`/`estimate_score`/`extract_tools`/`extract_metrics` 등 rule 기반 폴백 로직, `classify_input`/`target_chars_for`. jinyoung_v2에는 없는 "코칭 지능" 레이어 전체 |
| `cli/readiness.py` | **jintae_v2** 그대로 이식 | 5대 역량 준비도 계산(`calculate_readiness`), 약점 영역 미션 추천(`weakest_area`/`mission_for`), 대시보드 출력(`print_readiness`/`print_readiness_delta`) |
| `cli/memory.py` | **jintae_v2** 그대로 이식 | 과거 자소서 카드를 토큰 유사도로 추천하는 `recommend_cards`, 메모리 프롬프트 출력 |
| `cli/consistency.py` | **jinyoung_v2** `main.py`에서 **신규 분리** | rule 기반 키워드 충돌 검사 + NLI(mDeBERTa) + SBERT(KR-SBERT) pairwise 비교 로직을 그대로 가져오되, 가독성을 위해 `main.py`에서 별도 모듈로 추출. `check_consistency_pairwise`/`consistency_note_for_prompt`/`show_consistency`를 공개 API로 정리 |
| `cli/session.py` | **jintae_v2 + jinyoung_v2 합본** | 두 `Session` 데이터클래스의 필드를 합집합으로 병합. jinyoung_v2의 멀티턴 추적 필드(`job_history`/`consistency_checks`/`conversation_context`)와 jintae_v2의 코칭 채점 필드(`input_type`/`score_before`/`score_after`/`retry_used`/`answer_card`/`versions`)를 한 클래스에서 함께 관리 |
| `cli/storage.py` | **jinyoung_v2** 그대로 사용 | `STORAGE_DIR` 기반 JSON 저장/조회, `get_top_jobs`, `print_history`(직무 추세 표시 포함) — drafts/sessions/`_final.txt` 저장 구조를 그대로 지원 |
| `cli/models.py` | **신규 작성** (test3/jintae_v2 GPU 스택 + jinyoung_v2 CPU 스택 통합) | 아래 "3. 모델 계층" 참고 |
| `cli/config.yaml` | **신규 작성** (jintae_v2 구조 기반) | `runtime`(Ollama 연동)/`input`/`interviewer`/`rewrite`/`memory`/`dashboard` 섹션은 jintae_v2 구조를 그대로 채택. `consistency.py`는 jinyoung_v2 원본처럼 모듈 상수(임계값)를 사용하므로 dead config가 되는 `session:`/`consistency:` 섹션은 만들지 않음 |
| `cli/main.py` | **신규 작성** (jinyoung_v2 흐름 골격 + jintae_v2 코칭 단계 삽입) | 아래 "4. main.py 통합 흐름" 참고 |

## 3. 모델 계층 — GPU/CPU 자동 감지

`models.py`는 `torch.cuda.is_available()` + `unsloth` 설치 여부로 다음과 같이 분기한다.

| 환경 | 생성 모델 | 분석 모델 | 출처 |
|---|---|---|---|
| GPU 가능 | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit + LoRA r=16) | `ModernBERT-base` 기반 멀티태스크 모델 | test3 / jintae_v2 |
| GPU 불가 (CPU) | `skt/kogpt2-base-v2` | `monologg/koelectra-small-v3-discriminator` 기반 멀티태스크 모델 | jinyoung_v2 |

- 분석 모델의 `_MultiTaskModel`/`AnalysisModel` 클래스는 ModernBERT/KoELECTRA에서 아키텍처가
  동일하므로 코드를 공유하고, 체크포인트 경로(`models/best` vs `models/best_cpu`)와
  `train_config.json`의 `model_name`만 환경에 맞게 선택한다.
- 프롬프트 포맷도 분기한다: GPU 경로는 Llama-3 채팅 템플릿 특수 토큰
  (`<|begin_of_text|>`, `<|eot_id|>` 등, jintae_v2 방식)을 쓰고, CPU 경로는 후처리
  (`_clean_generated`)가 붙는 평문 한국어 프롬프트(jinyoung_v2 방식)를 쓴다. 두 경로 모두
  `generate_question`/`generate_tail_question`이라는 동일한 공개 API 뒤에 숨겨져 있다.
- `ollama_chat`은 jintae_v2에서 그대로 이식 — 로컬 LLM(EXAONE 3.5)으로 STAR 추출/재작성을
  수행하고, 실패 시 rule 기반으로 자동 폴백한다.

## 4. main.py 통합 흐름 (`run_session`)

jinyoung_v2의 "파일 편집기 기반 멀티턴 + 일관성 검사 + 직무 추적" 골격을 베이스로 삼고,
jintae_v2의 "STAR 코칭 지능" 단계를 그 위에 끼워 넣었다. 단계별 출처:

1. **준비도 대시보드 + 추천 미션** — `load_config`/`calculate_readiness`/`print_readiness`/`mission_for` *(jintae_v2)*
2. **역량 랜덤 선택 후 질문 생성** — `models.generate_question` *(공통, models.py로 통합)*
3. **과거 카드 추천** — `recommend_cards`/`print_recommendations` *(jintae_v2 memory.py)*
4. **파일 편집기 기반 최초 답변 입력** — `_get_answer_via_file`(`conversation.txt` 누적 템플릿) *(jinyoung_v2)*
5. **분석 모델 1차 예측 + 일관성 검사** — `analysis_model.predict`, `check_consistency_pairwise` *(jinyoung_v2)*
6. **STAR 추출 → 추가 질문 → 1차 재작성 → 재시도 코칭** — `_analyze_with_llm`(Ollama → `rule_analyze` 폴백),
   `_ask_one_followup`, `_rewrite_with_llm`(Ollama → `_rule_rewrite` 폴백), `_maybe_retry` *(jintae_v2)*
7. **꼬리질문 인터뷰 루프(최대 5턴)** — `build_conversation_context` + `consistency_note_for_prompt`를
   프롬프트에 주입해 `models.generate_tail_question` 호출, `_get_tail_reply_via_file`로 같은
   `conversation.txt`에 답변 누적, 매 턴마다 재분석·일관성 재검사·`job_history`/`suitable_jobs` 갱신
   *(jinyoung_v2 루프 구조 + jintae_v2 코칭 단계를 결합)*
8. **자소서 카드 빌드** — `_build_card`(STAR 구조/도구/수치/before-after 점수/개선 요약,
   원본→코칭본→인터뷰 최종본 3단계 버전 이력 포함) *(jintae_v2 카드 설계를 인터뷰 결과까지 반영하도록 확장)*
9. **저장** — `save_session`(JSON) + `_save_final_text`(대화 전문 + 카드 요약 + 직무 변화 추적이 담긴
   사람이 읽는 `_final.txt`) *(jinyoung_v2 저장 방식을 jintae_v2 카드 정보로 보강)*
10. **마무리 대시보드** — `print_readiness_delta` + `_show_job_trend` + `_show_summary` *(두 브랜치의 다이얼로그를 결합)*

## 5. test4에서 새로 설계한 부분

- **"코칭 후 인터뷰" 순서**: 두 원본 중 어느 쪽도 하지 않던 결합으로, 사용자의 최초 답변을
  STAR 코칭(추출→재작성→재시도)으로 먼저 다듬은 뒤, 그 다듬어진 답변을 기준으로 꼬리질문
  인터뷰를 진행한다. 이렇게 하면 인터뷰가 막연한 raw 답변이 아니라 이미 구조화된 답변을
  더 깊이 파고드는 방향으로 진행되고, 일관성 검사도 "원문 vs 코칭본 vs 인터뷰 답변" 세
  층위를 모두 비교할 수 있다.
- **3단계 버전 이력 카드**: 기존 jintae_v2 카드는 `original`/`final` 2버전만 저장했지만,
  test4에서는 STAR 코칭 직후 텍스트(`coached`)도 함께 보존해 "원본 → 코칭 → 인터뷰 후 최종"
  변화 과정을 그대로 추적할 수 있게 했다.
- **`requirements.txt` 병합**: 세 브랜치 내용이 거의 동일했으나 jinyoung_v2 쪽에
  `bitsandbytes`가 빠져 있어 GPU 4-bit 추론이 불가능했다. jintae_v2/test3 기준으로
  `bitsandbytes>=0.43.0`을 추가했다.
