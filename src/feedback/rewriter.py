# STEP 4 - 로컬 LLM 기반 STAR 재구성
# Ollama를 통해 로컬에서 EXAONE-3.5-7.8B 모델을 호출한다.
# GPU(RTX 5070 Ti, 16GB)에서 4-bit 양자화로 약 5GB 사용
#
# 모델 교체 시 MODEL 상수만 변경하면 됨:
#   exaone3.5:7.8b  → 한국어 특화 (기본값, 추천)
#   qwen2.5:14b     → 다국어 강세, 더 높은 추론 성능
#   gemma3:12b      → Google 최신 모델

import requests
from src.similarity.schema import ComparisonResult
from src.setup.ollama_manager import ensure

MODEL      = 'exaone3.5:7.8b'
OLLAMA_URL = 'http://localhost:11434/api/chat'

# 생성 파라미터
# temperature: 낮을수록 일관성 높음 (자소서 특성상 0.7 권장)
# num_predict: 최대 생성 토큰 수
OPTIONS = {
    'temperature': 0.7,
    'num_predict': 1024,
    'top_p': 0.9,
}

# 프로그램 시작 후 첫 rewrite() 호출 시 한 번만 자동 설정 실행
_ready = False


def _ensure_ready() -> None:
    """Ollama 설치·서버·모델을 한 번만 자동으로 준비한다."""
    global _ready
    if not _ready:
        ensure(MODEL)
        _ready = True


def rewrite(
    result: ComparisonResult,
    user_request: str = '',
    question: str | None = None,    # 문항 형식 시 원래 질문 텍스트
    entities: dict | None = None,   # 경력/학력/어학 조건
) -> str:
    """
    ComparisonResult를 바탕으로 자소서 문단을 STAR 구조로 재작성한다.

    Args:
        result:       STEP 3 비교 결과
        user_request: r 옵션으로 입력한 추가 요구사항 (없으면 빈 문자열)
    """
    _ensure_ready()

    prompt = _build_prompt(result, user_request, question, entities)

    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': OPTIONS,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError('모델 응답 시간 초과. GPU 메모리 부족이거나 모델 로딩 중일 수 있습니다.')

    return resp.json()['message']['content'].strip()


def _build_prompt(
    result: ComparisonResult,
    user_request: str,
    question: str | None = None,
    entities: dict | None = None,
) -> str:
    """
    재작성 프롬프트를 구성한다.
    v2: 문항 질문(jinyong1)과 엔티티 조건(jintea1)을 추가 컨텍스트로 포함.
    EXAONE은 한국어 지시문을 잘 따르므로 명확한 한국어 지시가 효과적이다.
    """
    dominant     = result.section_scores.dominant()
    keywords_str = ', '.join(result.priority_keywords) if result.priority_keywords else '없음'
    user_note    = f'\n추가 요구사항: {user_request}' if user_request else ''
    question_ctx = f'\n[자소서 문항]\n{question}' if question else ''
    entity_ctx   = ''
    if entities:
        parts = [f"{k}: {', '.join(v)}" for k, v in entities.items()]
        entity_ctx = f"\n[채용 조건]\n" + '\n'.join(parts)

    return f"""다음 자기소개서 문단을 채용공고에 맞게 STAR 구조로 재작성해주세요.{question_ctx}

[원문]
{result.paragraph_text}

[채용공고 핵심 정보]
- 가장 관련된 섹션: {dominant}
- 반드시 포함해야 할 키워드: {keywords_str}
- 현재 공고 부합도: {result.overall_score:.2f} / 1.0{entity_ctx}{user_note}

[재작성 규칙]
1. STAR 구조를 따를 것 (S:상황 → T:역할 → A:행동 → R:결과)
2. 위의 키워드를 자연스럽게 녹여 넣을 것
3. 원문의 실제 경험을 과장하지 말고 표현만 바꿀 것
4. 문항이 있는 경우 그 질문 의도에서 벗어나지 말 것
5. 3~5문장 분량으로 작성할 것
6. 한국어로 작성할 것

재작성된 문단만 출력하세요. 설명이나 부연은 포함하지 마세요."""
