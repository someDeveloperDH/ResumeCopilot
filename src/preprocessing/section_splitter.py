# jintea1/part1/section_splitter.py 기반
# test1의 fetcher.py가 단순 regex를 쓰던 것을 이 모듈로 대체
# 다양한 헤더 표현(■, ◆, ▶, [ ] 등)을 모두 처리

from __future__ import annotations

import re
from src.keyword.constants import SECTION_NAMES, SECTION_PATTERNS

HEADER_GROUP_TO_SECTION: dict[str, str] = {}


def _header_regex() -> re.Pattern[str]:
    alternatives = []
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            group_name = f"s{len(HEADER_GROUP_TO_SECTION)}"
            HEADER_GROUP_TO_SECTION[group_name] = section
            alternatives.append(f"(?P<{group_name}>{pattern})")

    regex = (
        r"(?im)(?:^|\n|\s)"
        r"[\[\]()<>{}■◆▶\-*ㆍ·\s]*"
        r"(" + "|".join(alternatives) + r")"
        r"[\]\)>\s:：\-]*"
    )
    return re.compile(regex)


HEADER_RE = _header_regex()


def _section_for_match(match: re.Match[str]) -> str:
    for group_name, section in HEADER_GROUP_TO_SECTION.items():
        if match.group(group_name):
            return section
    return "자격요건"


def split_sections(text: str) -> dict[str, str]:
    """
    채용공고 텍스트를 주요업무/자격요건/우대사항 섹션으로 분리한다.

    헤더가 없으면 전체 텍스트를 자격요건으로 처리.
    동일 섹션이 여러 번 등장하면 내용을 이어붙인다.
    """
    sections = {section: "" for section in SECTION_NAMES}
    if not text:
        return sections

    matches = list(HEADER_RE.finditer(text))
    if not matches:
        # 헤더 없음 → 전체를 자격요건으로 처리
        sections["자격요건"] = text.strip()
        return sections

    for index, match in enumerate(matches):
        section = _section_for_match(match)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip(" :：-")
        if body:
            sections[section] = f"{sections[section]} {body}".strip()

    # 첫 번째 헤더 이전 텍스트는 자격요건으로 포함
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections["자격요건"] = f"{prefix} {sections['자격요건']}".strip()

    return sections
