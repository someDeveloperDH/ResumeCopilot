# STEP 0 - 채용공고 URL 스크래핑
# 사람인, 잡코리아 등 주요 채용 사이트의 구조가 달라
# 특정 태그 대신 텍스트 밀도 기반으로 본문을 추출함

import re
import requests
from bs4 import BeautifulSoup

# 채용공고 섹션 헤더 패턴 (사이트마다 표현이 다양해서 정규식으로 처리)
SECTION_PATTERNS = {
    '주요업무': r'주요\s*업무|담당\s*업무|업무\s*내용|하는\s*일',
    '자격요건': r'자격\s*요건|지원\s*자격|필수\s*조건|요구\s*사항',
    '우대사항': r'우대\s*사항|우대\s*조건|이런\s*분이면',
}

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}


def fetch(url: str, timeout: int = 10) -> dict:
    """
    URL에서 채용공고 텍스트를 수집한다.

    Returns:
        {
            'raw_text': str,           전체 본문 텍스트
            'sections': dict,          섹션별 텍스트 {주요업무, 자격요건, 우대사항}
            'eval': dict               STEP 0 평가 지표
        }
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        return _fail_result(str(e))

    soup = BeautifulSoup(resp.text, 'html.parser')

    # JS 렌더링 결과가 없는 정적 HTML만 처리 가능
    # script·style 태그는 본문 추출 방해 요소
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    raw_text = soup.get_text(separator='\n')
    raw_text = _normalize(raw_text)

    sections = _extract_sections(raw_text)

    eval_result = {
        'fetch_success': True,
        'text_length': len(raw_text),
        'section_detected': {k: bool(v) for k, v in sections.items()},
    }

    return {
        'raw_text': raw_text,
        'sections': sections,
        'eval': eval_result,
    }


def _extract_sections(text: str) -> dict:
    """텍스트에서 채용 섹션을 분리한다."""
    sections = {key: '' for key in SECTION_PATTERNS}
    lines = text.split('\n')

    current_section = None
    buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 섹션 헤더 탐지
        matched = _match_section(line)
        if matched:
            # 이전 섹션 저장
            if current_section:
                sections[current_section] = ' '.join(buffer)
            current_section = matched
            buffer = []
        elif current_section:
            buffer.append(line)

    # 마지막 섹션 저장
    if current_section and buffer:
        sections[current_section] = ' '.join(buffer)

    return sections


def _match_section(line: str) -> str | None:
    """해당 줄이 섹션 헤더인지 확인하고 섹션명을 반환한다."""
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, line):
            return section
    return None


def _normalize(text: str) -> str:
    """연속 공백과 빈 줄을 정리한다."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _fail_result(reason: str) -> dict:
    return {
        'raw_text': '',
        'sections': {k: '' for k in SECTION_PATTERNS},
        'eval': {
            'fetch_success': False,
            'text_length': 0,
            'section_detected': {k: False for k in SECTION_PATTERNS},
            'error': reason,
        },
    }
