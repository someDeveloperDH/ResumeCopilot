"""로컬 LLM 기반 자소서 코칭 보조 로직."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict


DENIAL_WORDS = {"n", "no", "없음", "없어요", "모름", "잘 모르겠음", "해당없음", "경험없음"}
TECH_WORDS = {
    "python", "java", "javascript", "typescript", "react", "node", "node.js",
    "sql", "mysql", "postgresql", "mongodb", "redis", "kafka", "rabbitmq",
    "pandas", "numpy", "openpyxl", "excel", "aws", "gcp", "azure", "docker",
    "kubernetes", "fastapi", "spring", "django", "flask", "jwt", "api",
}


@dataclass
class CoachAnalysis:
    input_type: str = "normal_answer"
    core_experience: str = ""
    situation: str = ""
    task: str = ""
    action: str = ""
    result: str = ""
    tools: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    followup_question: str = ""
    improvement_summary: list[str] = field(default_factory=list)
    reuse_angles: list[str] = field(default_factory=list)
    source_cards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str = "cli/config.yaml") -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_denial(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.strip().lower())
    return normalized in DENIAL_WORDS


def classify_input(text: str, cfg: dict) -> str:
    length = len(text.strip())
    if is_denial(text):
        return "no_experience"
    if length < cfg["input"]["seed_threshold_chars"]:
        return "seed"
    if length < cfg["input"]["short_threshold_chars"]:
        return "short_dump"
    if length >= cfg["input"]["long_threshold_chars"]:
        return "long_answer"
    return "normal_answer"


def validate_user_dump(text: str, cfg: dict) -> tuple[bool, str]:
    if is_denial(text):
        return True, ""
    if len(text.strip()) < cfg["input"]["min_dump_chars"]:
        return (
            False,
            "[안내] 답변이 조금 짧아 자소서 카드로 만들기 어렵습니다. "
            "사용한 툴, 문제 상황, 결과 중 한 가지만 더 적어주세요.",
        )
    return True, ""


def extract_tools(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for word in sorted(TECH_WORDS):
        if word in lower:
            found.append(_canonical_tool(word))
    return sorted(set(found))


def extract_metrics(text: str) -> list[str]:
    patterns = [
        r"\d+\s*(?:%|퍼센트|명|건|회|시간|분|초|일|개월|년)",
        r"\d+\s*(?:에서|->|→)\s*\d+\s*(?:%|퍼센트|명|건|회|시간|분|초|일)?",
        r"(?:절감|감소|단축|증가|개선|향상|자동화|축소|확대)",
    ]
    metrics = []
    for pattern in patterns:
        metrics.extend(re.findall(pattern, text))
    return [m.strip() for m in metrics if m.strip()]


def rule_analyze(question: str, answer: str, cfg: dict) -> CoachAnalysis:
    input_type = classify_input(answer, cfg)
    tools = extract_tools(answer)
    metrics = extract_metrics(answer)

    missing = []
    if len(answer.strip()) < cfg["input"]["short_threshold_chars"]:
        missing.append("상황/맥락")
    if not tools:
        missing.append("도구/기술")
    if not metrics:
        missing.append("정량 성과")

    followup = _build_followup(missing)
    core = _guess_core_experience(answer)
    action = ", ".join(tools) + " 활용" if tools else ""
    result = ", ".join(metrics) if metrics else ""

    return CoachAnalysis(
        input_type=input_type,
        core_experience=core,
        situation=answer.strip() if input_type in {"seed", "short_dump"} else "",
        task="",
        action=action,
        result=result,
        tools=tools,
        metrics=metrics,
        missing_fields=missing,
        followup_question=followup,
        improvement_summary=[
            "문제 상황과 본인 역할이 드러나도록 구조를 정리합니다.",
            "사용 기술과 행동을 한 문장에 묶습니다.",
            "제공된 수치가 있으면 결과 문장에 배치합니다.",
        ],
        reuse_angles=_guess_reuse_angles(question, answer),
    )


def parse_analysis_json(raw: str, fallback: CoachAnalysis) -> CoachAnalysis:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return fallback
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return fallback

    def list_value(key: str) -> list[str]:
        value = data.get(key, [])
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    star = data.get("star", {}) if isinstance(data.get("star"), dict) else {}
    return CoachAnalysis(
        input_type=str(data.get("input_type") or fallback.input_type),
        core_experience=str(data.get("core_experience") or fallback.core_experience),
        situation=str(star.get("situation") or data.get("situation") or fallback.situation),
        task=str(star.get("task") or data.get("task") or fallback.task),
        action=str(star.get("action") or data.get("action") or fallback.action),
        result=str(star.get("result") or data.get("result") or fallback.result),
        tools=list_value("tools") or fallback.tools,
        metrics=list_value("metrics") or fallback.metrics,
        missing_fields=list_value("missing_fields") or fallback.missing_fields,
        followup_question=str(data.get("followup_question") or fallback.followup_question),
        improvement_summary=list_value("improvement_summary") or fallback.improvement_summary,
        reuse_angles=list_value("reuse_angles") or fallback.reuse_angles,
        source_cards=list_value("source_cards"),
    )


def build_extraction_prompt(question: str, competency: str, answer: str, input_type: str) -> str:
    return f"""
당신은 한국어 IT/AI 자기소개서 코치입니다.
사용자가 제공한 사실만 구조화하세요. 없는 도구, 수치, 성과를 만들지 마세요.
응답은 반드시 JSON 하나만 출력하세요.

JSON 스키마:
{{
  "input_type": "seed|short_dump|normal_answer|long_answer|no_experience",
  "core_experience": "핵심 경험 한 줄",
  "star": {{
    "situation": "상황",
    "task": "과제/역할",
    "action": "행동",
    "result": "결과"
  }},
  "tools": ["사용자가 언급한 도구만"],
  "metrics": ["사용자가 언급한 수치만"],
  "missing_fields": ["부족한 항목"],
  "followup_question": "부족한 핵심 하나만 묻는 한국어 질문",
  "improvement_summary": ["개선 요약 1", "개선 요약 2", "개선 요약 3"],
  "reuse_angles": ["문제해결", "협업", "주도성"]
}}

역량: {competency}
질문: {question}
입력 타입: {input_type}
사용자 입력:
{answer}
""".strip()


def build_rewrite_prompt(
    question: str,
    competency: str,
    original: str,
    analysis: CoachAnalysis,
    followup_answer: str,
    target_chars: int,
) -> str:
    if analysis.input_type == "no_experience":
        track = "Track B: 없는 사실, 도구, 수치를 추가하지 말고 문장만 정리."
    else:
        track = "Track A: 사용자가 제공한 사실만 STAR 흐름으로 통합."

    return f"""
한국어 IT/AI 자기소개서 코치입니다.
{track}
환각 금지: 사용자가 말하지 않은 회사명, 도구, 수치, 성과를 만들지 마세요.
{target_chars}자 이내로 작성하세요.

질문: {question}
역량: {competency}
원문:
{original}
추출 정보:
S={analysis.situation}
T={analysis.task}
A={analysis.action}
R={analysis.result}
도구={", ".join(analysis.tools)}
수치={", ".join(analysis.metrics)}
보충={followup_answer}

출력:
최종문단: ...
개선요약:
- ...
- ...
- ...
""".strip()


def split_rewrite_response(text: str, fallback_summary: list[str]) -> tuple[str, list[str]]:
    final = text.strip()
    summary = fallback_summary[:]

    normalized = text.replace("**", "")
    match = re.search(r"개선\s*요약\s*:?", normalized)
    if match:
        before = normalized[: match.start()]
        after = normalized[match.end() :]
        final = before
        final = re.sub(r"^\s*(?:최종\s*문단|최종문단)\s*:?", "", final).strip()
        final = final.replace("1)", "").strip()
        bullets = [line.strip(" -\t") for line in after.splitlines() if line.strip().startswith("-")]
        if bullets:
            summary = bullets[:3]
    final = re.sub(r"^\s*(?:최종\s*문단|최종문단)\s*:?", "", final).strip()
    return final, summary[:3]


def estimate_score(question: str, text: str, analysis: CoachAnalysis | None = None) -> float:
    score = 35.0
    if len(text.strip()) >= 80:
        score += 15
    if len(text.strip()) >= 300:
        score += 10
    if extract_tools(text) or (analysis and analysis.tools):
        score += 15
    if extract_metrics(text) or (analysis and analysis.metrics):
        score += 20
    if any(word in text for word in ["문제", "역할", "해결", "개선", "결과", "성과"]):
        score += 5
    if any(word in text for word in re.findall(r"[가-힣A-Za-z0-9]+", question)[:8]):
        score += 5
    return max(0.0, min(100.0, score))


def target_chars_for(input_type: str, cfg: dict) -> int:
    rewrite = cfg["rewrite"]
    if input_type in {"seed", "short_dump"}:
        return int(rewrite["short_target_chars"])
    if input_type == "long_answer":
        return int(rewrite["long_target_chars"])
    return int(rewrite["default_target_chars"])


def _build_followup(missing: list[str]) -> str:
    if "정량 성과" in missing:
        return "결과를 숫자로 표현할 수 있나요? 예: 2시간에서 10분, 오류 50% 감소, 사용자 500명"
    if "도구/기술" in missing:
        return "이 경험에서 사용한 구체적인 기술, 도구, 라이브러리는 무엇인가요?"
    return "당시 본인이 맡은 역할이나 해결해야 했던 문제를 한 문장으로 더 설명해 주세요."


def _guess_core_experience(text: str) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return "소재 미정"
    return clean[:40] + ("..." if len(clean) > 40 else "")


def _guess_reuse_angles(question: str, answer: str) -> list[str]:
    joined = question + " " + answer
    angles = []
    if any(word in joined for word in ["문제", "오류", "개선", "해결", "자동화"]):
        angles.append("문제해결")
    if any(word in joined for word in ["팀", "협업", "조율", "의견"]):
        angles.append("협업")
    if any(word in joined for word in ["주도", "제안", "직접", "먼저"]):
        angles.append("주도성")
    if any(word in joined for word in ["실패", "실수", "재발", "고친"]):
        angles.append("실패경험")
    if any(word in joined for word in ["처음", "배운", "성장", "학습"]):
        angles.append("성장")
    return angles or ["문제해결"]


def _canonical_tool(word: str) -> str:
    mapping = {
        "node": "Node.js",
        "node.js": "Node.js",
        "sql": "SQL",
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "openpyxl": "Openpyxl",
        "excel": "Excel",
        "api": "API",
        "jwt": "JWT",
    }
    return mapping.get(word, word.capitalize())
