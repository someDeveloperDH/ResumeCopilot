# STEP 0 - 파일 파싱 (txt / pdf)
# jinyong1/parser.py의 문항(### 1. 질문) 파싱을 통합
# 자소서가 문항 형식이면 문항 리스트로, 아니면 일반 텍스트로 반환

import re
from pathlib import Path


def parse(file_path: str) -> dict:
    """
    파일을 읽어 텍스트와 평가 지표를 반환한다.

    Returns:
        {
            'text': str,            원본 전체 텍스트
            'sections': list[dict] | None,  문항 구조 감지 시 [{number, question, answer}]
            'mode': 'section' | 'paragraph',
            'eval': dict
        }
    """
    path = Path(file_path)
    if not path.exists():
        return _fail_result(f"파일 없음: {file_path}")

    suffix = path.suffix.lower()
    if suffix == '.txt':
        raw = _read_txt(path)
    elif suffix == '.pdf':
        raw = _read_pdf(path)
    else:
        return _fail_result(f"지원하지 않는 형식: {suffix}")

    if raw is None:
        return _fail_result(f"파일 읽기 실패: {file_path}")

    # 문항 형식(### 1. 질문) 감지 시도
    sections = _parse_sections(raw)
    mode = 'section' if sections else 'paragraph'

    text = _normalize(raw)
    return {
        'text': text,
        'sections': sections,
        'mode': mode,
        'eval': {
            'fetch_success': True,
            'source': str(path),
            'text_length': len(text),
            'likely_valid': len(text) >= 200,
            'mode': mode,
            'n_sections': len(sections) if sections else 0,
        },
    }


def _parse_sections(text: str) -> list[dict] | None:
    """
    자기소개서 문항 파싱 로직.

    지원 형식:
    1) ### 1. 질문
       답변
    2) 1. 질문
       답변

    다음 번호 문항이 나오기 전까지의 모든 하위 문단을 해당 문항의 답변으로 묶는다.
    """
    patterns = [
        # 기존 형식: ### 1. 질문
        r'^\s*###\s*(\d+)\.\s*(.+?)\s*\n(.*?)(?=^\s*###\s*\d+\.\s*|\Z)',
        # 일반 형식: 1. 질문
        r'^\s*(\d+)\.\s+(.+?)\s*\n(.*?)(?=^\s*\d+\.\s+|\Z)',
    ]

    matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)
        if matches:
            break

    if not matches:
        return None

    sections = []
    for number, question, answer in matches:
        question = _normalize_question(question)
        answer = _normalize(answer)
        if question and answer:
            sections.append({
                'number': int(number),
                'question': question,
                'answer': answer,
            })
    return sections if sections else None


def _normalize_question(question: str) -> str:
    """질문 텍스트 내부 개행/공백을 한 줄로 정리한다."""
    question = re.sub(r'\s+', ' ', question)
    return question.strip()

def _read_txt(path: Path) -> str | None:
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding='cp949')
        except Exception:
            return None


def _read_pdf(path: Path) -> str | None:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        raise RuntimeError("pdfminer.six 미설치: pip install pdfminer.six")
    except Exception:
        return None


def _normalize(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _fail_result(reason: str) -> dict:
    return {
        'text': '',
        'sections': None,
        'mode': 'paragraph',
        'eval': {
            'fetch_success': False,
            'error': reason,
            'text_length': 0,
            'likely_valid': False,
            'mode': 'paragraph',
            'n_sections': 0,
        },
    }
