# # STEP 4 - 로컬 LLM 기반 STAR 재구성
# # Ollama를 통해 로컬에서 EXAONE-3.5-7.8B 모델을 호출한다.
# # GPU(RTX 5070 Ti, 16GB)에서 4-bit 양자화로 약 5GB 사용
# #
# # 모델 교체 시 MODEL 상수만 변경하면 됨:
# #   exaone3.5:7.8b  → 한국어 특화 (기본값, 추천)
# #   qwen2.5:14b     → 다국어 강세, 더 높은 추론 성능
# #   gemma3:12b      → Google 최신 모델
#
# import requests
# from src.similarity.schema import ComparisonResult
# from src.setup.ollama_manager import ensure
#
# # MODEL      = 'exaone3.5:7.8b'
# # MODEL = 'gemma3:4b'
# MODEL = 'gemma3:1b'
# OLLAMA_URL = 'http://localhost:11434/api/chat'
#
# # 생성 파라미터
# # temperature: 낮을수록 일관성 높음 (자소서 특성상 0.7 권장)
# # num_predict: 최대 생성 토큰 수
# OPTIONS = {
#     'temperature': 0.7,
#     'num_predict': 512,
#     'top_p': 0.9,
# }
#
# # 프로그램 시작 후 첫 rewrite() 호출 시 한 번만 자동 설정 실행
# _ready = False
#
#
# def _ensure_ready() -> None:
#     """Ollama 설치·서버·모델을 한 번만 자동으로 준비한다."""
#     global _ready
#     if not _ready:
#         ensure(MODEL)
#         _ready = True
#
#
# def rewrite(
#     result: ComparisonResult,
#     user_request: str = '',
#     question: str | None = None,    # 문항 형식 시 원래 질문 텍스트
#     entities: dict | None = None,   # 경력/학력/어학 조건
# ) -> str:
#     """
#     ComparisonResult를 바탕으로 자소서 문단을 STAR 구조로 재작성한다.
#
#     Args:
#         result:       STEP 3 비교 결과
#         user_request: r 옵션으로 입력한 추가 요구사항 (없으면 빈 문자열)
#     """
#     _ensure_ready()
#
#     prompt = _build_prompt(result, user_request, question, entities)
#
#     payload = {
#         'model': MODEL,
#         'messages': [{'role': 'user', 'content': prompt}],
#         'stream': False,
#         'options': OPTIONS,
#     }
#
#     try:
#         resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
#         resp.raise_for_status()
#     except requests.Timeout:
#         raise RuntimeError('모델 응답 시간 초과. GPU 메모리 부족이거나 모델 로딩 중일 수 있습니다.')
#
#     return resp.json()['message']['content'].strip()
#
#
# def _build_prompt(
#     result: ComparisonResult,
#     user_request: str,
#     question: str | None = None,
#     entities: dict | None = None,
# ) -> str:
#     """
#     재작성 프롬프트를 구성한다.
#     v2: 문항 질문(jinyong1)과 엔티티 조건(jintea1)을 추가 컨텍스트로 포함.
#     EXAONE은 한국어 지시문을 잘 따르므로 명확한 한국어 지시가 효과적이다.
#     """
#     dominant     = result.section_scores.dominant()
#     keywords_str = ', '.join(result.priority_keywords) if result.priority_keywords else '없음'
#     user_note    = f'\n추가 요구사항: {user_request}' if user_request else ''
#     question_ctx = f'\n[자소서 문항]\n{question}' if question else ''
#     entity_ctx   = ''
#     if entities:
#         parts = [f"{k}: {', '.join(v)}" for k, v in entities.items()]
#         entity_ctx = f"\n[채용 조건]\n" + '\n'.join(parts)
#
#     return f"""다음 자기소개서 문단을 채용공고에 맞게 STAR 구조로 재작성해주세요.{question_ctx}
#
# [원문]
# {result.paragraph_text}
#
# [채용공고 핵심 정보]
# - 가장 관련된 섹션: {dominant}
# - 반드시 포함해야 할 키워드: {keywords_str}
# - 현재 공고 부합도: {result.overall_score:.2f} / 1.0{entity_ctx}{user_note}
#
# [재작성 규칙]
# 1. 자기소개서 본문만 출력할 것.
# 2. 설명, 해설, 제목, 서론, 안내문을 절대 출력하지 말 것.
# 3. "다음은", "재작성 결과", "STAR 구조", "수정본" 등의 문구를 절대 출력하지 말 것.
# 4. S:, T:, A:, R: 와 같은 표기 사용 금지.
# 5. 요약 문장 1문장 + 본문 3~5문장 분량으로 작성할 것.
# 6. 첫 문장은 답변 전체를 요약하는 의미 있는 한 문장으로 시작할 것.
#    - 예: "가장 단순한 해결 방법을 통해 효율적인 결과를 도출하다."
#    - 단순 제목만 쓰지 말고, 답변 핵심이 드러나는 완성형 문장으로 작성할 것.
# 7. 위의 키워드를 자연스럽게 녹여 넣을 것.
# 8. 이후 내용은 STAR 흐름이 자연스럽게 녹아 있어야 하며 명시적으로 구분하지 말 것.
# 9. 결과는 1개의 완성된 자기소개서 문단으로 작성할 것.
# 10. 원문에 없는 경험이나 수치는 추가하지 말 것.
# 11. 한국어로 작성할 것.
#
#
# 출력은 자기소개서 본문만 작성한다.
# 설명문, 제목, 마크다운, 목록, STAR 표기 금지."""


# rewriter.py
# STEP 4/5/6 - Ollama 기반 인터랙티브 체크리스트 및 템플릿 엔진

import json
import requests
from src.similarity.schema import ComparisonResult
from src.setup.ollama_manager import ensure

MODEL = 'gemma3:1b'  # 사용자 지정 로컬 최적화 모델 적용
OLLAMA_URL = 'http://localhost:11434/api/chat'
OPTIONS = {'temperature': 0.5, 'num_predict': 600, 'top_p': 0.85}

_ready = False


def _ensure_ready() -> None:
    global _ready
    if not _ready:
        ensure(MODEL)
        _ready = True


def _call_llm(prompt: str) -> str:
    _ensure_ready()
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': OPTIONS,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()['message']['content'].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama 통신 실패: {e}")


def evaluate_checklist(text: str, checklist: list[str], question: str | None = None) -> list[str]:
    """
    들어온 답변이 10개 체크사항에 해당되는지 분석하여 미흡한 항목만 리스트로 반환 (JSON 파싱 유도).
    """
    checklist_str = "\n".join(checklist)
    question_ctx = f"[자소서 문항 질문]: {question}\n" if question else ""

    prompt = f"""
당신은 자소서 전문 인사평가관입니다. 다음 [자기소개서 답변]을 [10대 핵심 체크사항]과 대조하여, '내용이 누락되었거나 보완이 반드시 필요한 항목'만 선별해 주세요.

{question_ctx}
[자기소개서 답변]
{text}

[10대 핵심 체크사항]
{checklist_str}

[출력 규칙]
1. 미흡하거나 보완이 필요한 항목의 문장 전체를 JSON string array 형식으로만 반환하세요.
2. 예: ["2. 당시에 직면했던 가장 큰 핵심 문제(또는 목표)가 명확히 기술되었는가?", "5. 정량적 수치나 객관적 지표를 통한 구체적인 성과(Result)가 제시되었는가?"]
3. 만약 모든 항목이 잘 충족되었다면 빈 배열 [] 만 출력하세요.
4. 설명이나 마크다운 래퍼(```json) 없이 순수 JSON 배열만 출력하세요.
"""
    response = _call_llm(prompt)
    try:
        # 마크다운 블록이 붙어 나올 경우를 대비한 가벼운 정제
        if "[" in response and "]" in response:
            cleaned = response[response.find("["):response.rfind("]") + 1]
            return json.loads(cleaned)
        return []
    except Exception:
        # 파싱 에러 방어 코드: 수동 파싱 시도 또는 일부 항목 추출
        return [item for item in checklist if item[:2] in response]


def generate_guiding_question(text: str, check_item: str, question: str | None = None) -> str:
    """미흡한 체크사항을 바탕으로 사용자 경험을 이끌어낼 부드럽고 구체적인 유도질문 생성"""
    question_ctx = f"문항 질문: {question}\n" if question else ""
    prompt = f"""
다음 자기소개서 내용과 미흡 항목을 바탕으로, 지원자에게 추가 정보를 자연스럽게 얻어낼 수 있는 '친절한 유도 질문'을 1개만 생성해 주세요.

{question_ctx}
[현재 자소서 본문]
{text}

[미흡한 체크사항]
{check_item}

[작성 규칙]
- 딱딱한 질문 지양, "당시 상황에서 ~는 구체적으로 어떤 상황이었나요?" 처럼 구체적 경험을 유도할 것.
- 오직 질문 문장 1개만 출력할 것 (설명문 금지).
"""
    return _call_llm(prompt)


# src/feedback/rewriter.py 내부 함수 정의 수정

def internal_rewrite(
        text: str,
        guiding_question: str,
        answer: str,
        check_item: str,
        question: str | None = None,  # 문항 질문 컨텍스트 이름 통일
        entities: dict | None = None
) -> str:
    """[사용자에게 미출력] 유도질문과 유도질문 답변을 기존 본문에 자연스럽게 결합하여 누적 업데이트"""
    question_ctx = f"[자소서 문항 질문]: {question}\n" if question else ""

    prompt = f"""
기존 자기소개서 본문에, 유도 질문을 통해 획득한 지원자의 추가 답변 내용을 논리적으로 결합하여 살을 붙인 고도화된 자소서 본문을 작성해 주세요.

{question_ctx}
[기존 자소서 본문]
{text}

[체크 보완 항목]
{check_item}

[AI 유도 질문]
{guiding_question}

[지원자의 추가 답변]
{answer}

[작성 규칙]
- 기존 내용과 추가된 답변 정보가 모순 없이 유기적으로 이어지도록 문맥을 조정하세요.
- 최종 템플릿 형태로 가공하기 전 단계이므로 문단 구조나 서론/결론 장식 없이 '풍부해진 본문 내용 전체'를 완성된 글 형태로 반환하세요.
- 설명이나 주석은 절대 금지합니다.
"""
    return _call_llm(prompt)


def generate_final_template(text: str, result: ComparisonResult, user_request: str = '', question: str | None = None,
                            entities: dict | None = None) -> str:
    """인터뷰가 완료된 텍스트를 최종 템플릿 규칙(임팩트 요약문 + 자연스러운 STAR 흐름)에 맞춤 빌드"""
    dominant = result.section_scores.dominant()
    keywords_str = ', '.join(result.priority_keywords) if result.priority_keywords else '없음'
    user_note = f'\n[사용자 추가 요구사항]: {user_request}' if user_request else ''
    question_ctx = f'\n[자소서 채용 문항]\n{question}' if question else ''

    entity_ctx = ''
    if entities:
        parts = [f"{k}: {', '.join(v)}" for k, v in entities.items()]
        entity_ctx = f"\n[지원자 스펙/컨텍스트]\n" + '\n'.join(parts)

    prompt = f"""
지금까지 보완된 자소서 텍스트를 바탕으로, 제시된 엄격한 [최종 템플릿 규칙]을 준수하여 완벽한 하나의 자기소개서 문단으로 빌드해 주세요.

{question_ctx}
[보완 완료된 원천 텍스트]
{text}

[채용공고 매칭 가이드]
- 추천 타겟 섹션: {dominant}
- 필수 융합 키워드: {keywords_str}{entity_ctx}{user_note}

[최종 템플릿 규칙]
1. 첫 문장은 답변 전체를 관통하는 '임팩트 있는 완성형 한 줄 요약문'으로 시작할 것.
   - 예시: "철저한 데이터 분석과 최적화 설계를 통해 시스템 병목 현상을 40% 해결하다."
   - 단순 명사형 제목(예: "협업의 중요성")은 절대 금지합니다.
2. 두 번째 문장부터는 STAR(Situation, Task, Action, Result) 구조의 흐름이 자연스럽게 녹아들도록 전개할 것.
3. 본문 내에 "S:", "T:", "STAR 구조", "다음은 수정본입니다" 와 같은 메타 표기나 안내 문구는 절대 포함하지 마십시오.
4. 요약문 1문장 + 본문 4~6문장 이내의 유기적인 단일 문단으로 구성할 것.
5. 원천 텍스트에 없는 허위 사실이나 수치는 가공해 넣지 마십시오.

출력은 오직 완성된 자기소개서 본문 한 문단만 출력합니다.
"""
    return _call_llm(prompt)