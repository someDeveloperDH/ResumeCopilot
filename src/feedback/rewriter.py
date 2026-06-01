# rewriter.py
# STEP 4/5/6 - Ollama 기반 인터랙티브 체크리스트 및 템플릿 엔진

import json
import requests
from src.similarity.schema import ComparisonResult
from src.setup.ollama_manager import ensure

MODEL = 'gemma3:4b'  # 사용자 지정 로컬 최적화 모델 적용
OLLAMA_URL = 'http://localhost:11434/api/chat'
OPTIONS = {'temperature': 0.5, 'num_predict': 300, 'top_p': 0.85}

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
    # try:
    #     resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    #     resp.raise_for_status()
    #     return resp.json()['message']['content'].strip()
    # except Exception as e:
    #     raise RuntimeError(f"Ollama 통신 실패: {e}")
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)

        if not resp.ok:
            print("STATUS:", resp.status_code)
            print("BODY:", resp.text)

        resp.raise_for_status()

        return resp.json()['message']['content'].strip()

    except Exception as e:
        raise RuntimeError(f"Ollama 통신 실패: {e}")


def generate_guiding_questions(text: str, question: str | None = None) -> list[str]:
    """
    분석 후 '정말 치명적으로 보완이 필요한 경우'에만 제한적으로 질문을 생성합니다.
    본문이 이미 어느 정도 완성도가 있다면 빈 배열을 반환하여 유도 질문을 패스합니다.
    """
    question_ctx = f"[자기소개서 해당 문항]: {question}\n" if question else "[자기소개서 문항 정보 없음]\n"

    prompt = f"""
당신은 자기소개서 전문 컨설턴트입니다. 지원자가 작성한 [현재 답변 본문]이 [자기소개서 해당 문항]의 의도에 부합하는지 확인하고, 논리적 흐름상 핵심 내용(구체적 행동이나 성과)이 심각하게 누락되어 '반드시 추가 피드백을 받아야 하는 핵심 질문'이 있다면 도출해 주세요.

{question_ctx}
[현재 답변 본문]
{text}

[출력 제약 조건]
1. 내용이 충분하거나 굳이 질문할 필요가 없다면 반드시 빈 배열 [] 만 반환하세요. (억지로 질문을 만들지 마십시오)
2. 보완이 꼭 필요한 경우에만 최소 1개 ~ 최대 3개의 질문을 JSON string array 형식으로 반환하세요.
3. 설명이나 마크다운 래퍼 없이 오직 순수 JSON 배열만 출력하세요.
"""
    response = _call_llm(prompt)
    try:
        if "[" in response and "]" in response:
            cleaned = response[response.find("["):response.rfind("]") + 1]
            return json.loads(cleaned)[:3]
        return []
    except Exception:
        return []


def generate_guiding_question_v2(text: str, missing_point: str, question: str | None = None) -> str:
    """단일 미흡점 보완을 위한 백업/가공용 유도질문 생성기"""
    question_ctx = f"문항 질문: {question}\n" if question else ""
    prompt = f"""
다음 자기소개서 내용과 보완점을 바탕으로, 지원자에게 추가 정보를 자연스럽게 얻어낼 수 있는 친절한 유도 질문을 1개만 생성해 주세요.

{question_ctx}
[현재 자소서 본문]
{text}

[보완 필요 지점]
{missing_point}

오직 질문 문장 1개만 출력할 것 (설명문 금지).
"""
    return _call_llm(prompt)


def internal_rewrite(
        text: str,
        guiding_question: str,
        answer: str,
        check_item: str,
        question: str | None = None,
        entities: dict | None = None
) -> str:
    """유도질문과 유도질문 답변을 기존 본문에 자연스럽게 결합하여 누적 업데이트"""
    question_ctx = f"[자소서 문항 질문]: {question}\n" if question else ""

    prompt = f"""
기존 자기소개서 내용에 지원자의 추가 답변을 결합하여, 문맥이 끊기지 않고 서사가 풍부해진 고도화된 자소서 본문을 작성해 주세요. 
단순 문장 나열이 아니라 당시 상황과 행동이 입체적으로 드러나도록 내용을 살을 붙여 풍분하게 전개해야 합니다.

{question_ctx}
[기존 자소서 본문]
{text}

[AI 유도 질문과 지원자 답변]
질문: {guiding_question}
답변: {answer}

오직 풍부해진 완성형 본문 글만 반환하세요. 설명문이나 주석은 절대 금지합니다.
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
    당신은 최고의 웅변가이자 대기업 인사담당자입니다. 제시된 정보를 바탕으로 보완 완료된 원천 텍스트를 매끄럽고 설득력 있는 완성형 자기소개서로 작성해 주세요. 
    가벼운 어조나 요약 형식을 지양하고, 실제 제출 가능한 수준의 묵직하고 전문적인 비즈니스 문체로 분량을 풍부하게 채워야 합니다.

    {question_ctx}
    [보완 완료된 원천 텍스트]
    {text}

    [채용공고 매칭 가이드]
    - 추천 타겟 섹션: {dominant}
    - 필수 융합 키워드: {keywords_str}{entity_ctx}{user_note}

    [엄격한 작성 규칙]
    1. 첫 문장은 답변의 성과와 가치를 한눈에 보여주는 '임팩트 있는 완성형 요약문'으로 명확히 시작하세요.
       - 예시: "철저한 요구사항 분석과 최적화 설계를 바탕으로 시스템의 병목 현상을 해결하고 안정성을 크게 향상시켰습니다."
    2. 두 번째 문장부터는 STAR(상황-역할-행동-성과) 구조가 한 호흡의 이야기처럼 자연스럽게 이어지도록 서사를 풍부하게 전개하세요.
    3. 분량 조건: 최소 5문장 이상, 전체 단일 문단으로 풍성하게 기술하세요. 줄글이 너무 짧거나 생략되면 안 됩니다.
    4. 문맥에 맞게 [필수 융합 키워드]를 자연스럽고 깊이 있게 녹여내세요.
    5. "S:", "T:", "STAR 구조", "수정본입니다" 같은 가이드용 메타 표기나 안내 문구는 절대 포함하지 마십시오.

    출력은 오직 최종 완성된 자기소개서 본문 한 문단(텍스트)만 출력합니다.
    """
    return _call_llm(prompt)