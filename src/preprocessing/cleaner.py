# jintea1/part1/cleaner.py 기반으로 test1 cleaner.py를 강화
# 추가된 것: HTML 태그 제거, 이모지 제거, 기술명 정규화 (Python/파이썬 통일 등)

import re
from src.keyword.constants import TECH_NORMALIZE

HTML_TAG_RE        = re.compile(r'<[^>]+>')
EMOJI_RE           = re.compile(r'[\U0001F000-\U0001FFFF☀-⟿]+')
STANDALONE_SYM_RE  = re.compile(r'(?<![A-Za-z0-9가-힣])[^\w\s+#./-]+(?![A-Za-z0-9가-힣])')
SPACES_RE          = re.compile(r'\s+')
LIST_BULLET_RE     = re.compile(r'[•·※◎○●▶▷■◆]')
LINE_DASH_RE       = re.compile(r'^\s*[-–—]\s*', flags=re.MULTILINE)


def _normalize_key_pattern(key: str) -> re.Pattern[str]:
    escaped = re.escape(key)
    if re.search(r'[가-힣]', key):
        return re.compile(rf'(?<![가-힣A-Za-z0-9]){escaped}(?![가-힣A-Za-z0-9])', re.IGNORECASE)
    return re.compile(rf'(?<![A-Za-z0-9+#.]){escaped}(?![A-Za-z0-9+#.])', re.IGNORECASE)


def normalize_tech_names(text: str) -> str:
    """한글/약어 기술명을 표준 표기로 통일한다 (파이썬→Python 등)."""
    for source in sorted(TECH_NORMALIZE, key=len, reverse=True):
        text = _normalize_key_pattern(source).sub(TECH_NORMALIZE[source], text)
    return text


def clean_job_posting(text: str) -> str:
    """
    채용공고 전용 전처리.
    HTML, 이모지, 리스트 기호 제거 후 기술명 정규화까지 수행.
    """
    if not text:
        return ''
    text = HTML_TAG_RE.sub(' ', text)
    text = EMOJI_RE.sub(' ', text)
    text = LIST_BULLET_RE.sub(' ', text)
    text = LINE_DASH_RE.sub('', text)
    text = STANDALONE_SYM_RE.sub(' ', text)
    text = normalize_tech_names(text)
    text = SPACES_RE.sub(' ', text)
    return text.strip()


def clean_cover_letter(text: str) -> str:
    """
    자소서 전용 전처리.
    문단/문항 구조(빈 줄, ### 헤더)를 보존하면서 노이즈만 제거한다.
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if not line.strip():
            cleaned.append('')
        elif line.strip().startswith('###'):
            # 문항 헤더는 그대로 보존
            cleaned.append(line)
        else:
            line = HTML_TAG_RE.sub(' ', line)
            line = EMOJI_RE.sub(' ', line)
            line = SPACES_RE.sub(' ', line).strip()
            cleaned.append(line)
    return '\n'.join(cleaned)


def clean_text(text: str) -> str:
    """범용 경량 전처리 (자소서 단순 파싱 등에 사용)."""
    text = SPACES_RE.sub(' ', text)
    return text.strip()
