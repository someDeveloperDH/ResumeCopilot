# jintea1/part1/rule_extractor.py 기반
# Mecab 우선, 없으면 kiwipiepy로 자동 폴백 — 어느 환경에서나 동작
# Hard/Soft/Action 키워드 타입 분류 추가 (test1에는 없던 기능)

from __future__ import annotations

import re
from dataclasses import dataclass

from src.keyword.constants import (
    ACTION_VERBS, HARD_SKILLS, SOFT_SKILLS,
    STOPWORDS, USER_DICT_WORDS, WEIGHTS,
)

TOKEN_RE            = re.compile(r'[A-Za-z][A-Za-z0-9+#./-]*|[가-힣]{2,}|\d+(?:년|점|급)?')
KOREAN_PARTICLE_RE  = re.compile(r'(은|는|이|가|을|를|과|와|도|만|부터|까지|에서|으로|로|에게|의)$')
MECAB_KEEP_POS      = {'NNG', 'NNP', 'VV', 'VA', 'SL'}
MIN_WORD_LEN        = 2


@dataclass(frozen=True)
class TaggedToken:
    token: str
    pos: str


# ─── 형태소 분석기 (Mecab → kiwipiepy 순으로 폴백) ────────────────────────────

def _try_kiwipiepy(text: str) -> list[TaggedToken] | None:
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        tokens = kiwi.tokenize(text)
        return [TaggedToken(t.form, t.tag) for t in tokens if t.tag in MECAB_KEEP_POS]
    except Exception:
        return None


def _try_konlpy_mecab(text: str) -> list[TaggedToken] | None:
    try:
        from konlpy.tag import Mecab  # type: ignore
        tagged = []
        for token, pos in Mecab().pos(text):
            p = pos.split('+', 1)[0].split(',', 1)[0]
            if p in MECAB_KEEP_POS:
                tagged.append(TaggedToken(token, p))
        return tagged
    except Exception:
        return None


def _fallback_pos(text: str) -> list[TaggedToken]:
    """어떤 형태소 분석기도 없을 때 사용하는 regex 기반 폴백."""
    tagged = []
    for token in TOKEN_RE.findall(text):
        if token in ACTION_VERBS:
            pos = 'VV'
        elif re.match(r'^[A-Za-z]', token):
            pos = 'SL'
        else:
            pos = 'NNG'
        tagged.append(TaggedToken(token, pos))
    return tagged


def _tag(text: str) -> list[TaggedToken]:
    """kiwipiepy → KoNLPy Mecab → regex fallback 순서로 시도."""
    return _try_kiwipiepy(text) or _try_konlpy_mecab(text) or _fallback_pos(text)


# ─── 키워드 타입 분류 ──────────────────────────────────────────────────────────

def _classify_type(token: str) -> str:
    if token in HARD_SKILLS:
        return 'Hard'
    if token in SOFT_SKILLS:
        return 'Soft'
    if token in ACTION_VERBS:
        return 'Action'
    return 'General'


# ─── 사용자 사전 복합어 처리 ──────────────────────────────────────────────────

def _extract_user_dict_terms(text: str) -> list[str]:
    return [term for term in sorted(USER_DICT_WORDS, key=len, reverse=True) if term in text]


def _mask(text: str, terms: list[str]) -> str:
    masked = text
    for term in terms:
        masked = masked.replace(term, '_' * len(term))
    return masked


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def extract_keywords(sections: dict[str, str]) -> dict[str, list[dict]]:
    """
    섹션별로 Rule-based 키워드를 추출하고 Hard/Soft/Action 타입을 분류한다.
    결과는 hybrid_extractor와 evaluator의 입력 형식과 호환된다.
    """
    result = {}
    for section, text in sections.items():
        if not text:
            result[section] = []
            continue

        weight = WEIGHTS.get(section, 1.0)
        user_terms = _extract_user_dict_terms(text)
        masked_text = _mask(text, user_terms)
        tagged = _tag(masked_text)

        seen: set[str] = set()
        items: list[dict] = []

        # 사용자 사전 복합어 우선 추가
        for term in user_terms:
            if term not in seen:
                seen.add(term)
                items.append({
                    'keyword': term,
                    'pos': 'NNG',
                    'type': _classify_type(term),
                    'weight': weight,
                    'score': 0.0,
                    'hybrid_score': 0.0,
                    'source': 'UserDict',
                })

        for tt in tagged:
            token = KOREAN_PARTICLE_RE.sub('', tt.token).strip()
            if (
                token
                and len(token) >= MIN_WORD_LEN
                and token not in STOPWORDS
                and token not in seen
            ):
                seen.add(token)
                items.append({
                    'keyword': token,
                    'pos': tt.pos,
                    'type': _classify_type(token),
                    'weight': weight,
                    'score': 0.0,
                    'hybrid_score': 0.0,
                    'source': 'Rule',
                })

        result[section] = items
    return result


def evaluate(section_keywords: dict) -> dict:
    """STEP 1 Rule-based 평가 지표."""
    all_kws = [item['keyword'] for items in section_keywords.values() for item in items]
    type_counts = {'Hard': 0, 'Soft': 0, 'Action': 0, 'General': 0}
    for items in section_keywords.values():
        for item in items:
            type_counts[item.get('type', 'General')] = type_counts.get(item.get('type', 'General'), 0) + 1
    return {
        'method': 'rule_based',
        'n_keywords_per_section': {s: len(v) for s, v in section_keywords.items()},
        'total_keywords': len(all_kws),
        'type_breakdown': type_counts,
        'valid': len(all_kws) >= 5,
    }
