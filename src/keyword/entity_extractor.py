# jintea1/part1/entity_extractor.py 기반
# 채용공고에서 경력/학력/어학 조건을 정량적으로 추출한다
# test1에는 없던 기능 — 자소서 재구성 시 LLM 프롬프트에 활용

from __future__ import annotations

import re

ENTITY_PATTERNS = {
    '경력': re.compile(r'(?:경력\s*)?\d+\s*년\s*(?:이상|이하|차|내외)?'),
    '어학': re.compile(
        r'(?:TOEIC|토익|TOEFL|토플|OPIc|오픽|IELTS)\s*[A-Za-z0-9+\-]*\s*\d*\s*'
        r'(?:점|급|이상|IH|IM|AL)?',
        re.IGNORECASE,
    ),
    '학력': re.compile(r'(?:학사|석사|박사|전문학사)\s*(?:이상|졸업|학위)?'),
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    채용공고 텍스트에서 정량 조건을 추출한다.
    반환값은 rewriter 프롬프트에 추가 컨텍스트로 전달된다.

    예시:
        {'경력': ['3년 이상'], '학력': ['학사 이상'], '어학': ['TOEIC 800점 이상']}
    """
    entities: dict[str, list[str]] = {}
    for label, pattern in ENTITY_PATTERNS.items():
        values = []
        for match in pattern.finditer(text):
            value = ' '.join(match.group(0).split())
            if value and value not in values:
                values.append(value)
        if values:
            entities[label] = values
    return entities
