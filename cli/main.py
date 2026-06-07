"""
하루 1 문답 CLI

실행: python cli/main.py           # 새 세션 시작
      python cli/main.py --history  # 저장 이력 조회

main2의 학습 파이프라인은 유지하고, CLI 실행은 로컬 Ollama/EXAONE 기반
자소서 카드 저장 흐름으로 확장한다.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import models
from coach import (
    build_extraction_prompt,
    build_rewrite_prompt,
    classify_input,
    estimate_score,
    is_denial,
    load_config,
    parse_analysis_json,
    rule_analyze,
    split_rewrite_response,
    target_chars_for,
    validate_user_dump,
)
from memory import print_memory_prompts, print_recommendations, recommend_cards
from readiness import (
    calculate_readiness,
    mission_for,
    print_readiness,
    print_readiness_delta,
    weakest_area,
)
from session import Session
from storage import load_all, print_history, save_session


USE_STAGE14_FINETUNED = False
USE_STAGE3_FINETUNED = False
COMPETENCIES = ["문제해결", "협업", "성장", "실패경험", "주도성"]


def _show_question(competency: str, question: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"오늘의 3분 미션 [{competency}]")
    print(f"{'=' * 60}")
    print(question)
    print("\n이 질문이 보고 싶은 것")
    for item in _question_intents(competency):
        print(f"  - {item}")


def _question_intents(competency: str) -> list[str]:
    mapping = {
        "문제해결": ["문제를 정의한 방식", "본인이 취한 구체적 행동", "개선 결과나 수치"],
        "협업": ["갈등이나 조율 상황", "본인의 커뮤니케이션 방식", "팀 결과에 준 영향"],
        "성장": ["처음 부족했던 지점", "학습/시도 과정", "이후 달라진 결과"],
        "실패경험": ["실패 원인", "재발 방지 행동", "실패 이후 바뀐 방식"],
        "주도성": ["스스로 발견한 문제", "먼저 제안/실행한 행동", "주변이나 결과에 준 변화"],
    }
    return mapping.get(competency, ["상황", "본인의 역할", "행동과 결과"])


def _read_free_dump(cfg: dict) -> str:
    print("\n기억나는 대로 편하게 적어주세요.")
    print("문장 완성 안 해도 됩니다. 키워드만 적어도 됩니다.")
    print("오늘은 완성 답변 1개 또는 소재 씨앗 1개만 저장해도 성공입니다.")
    print("입력 종료: 빈 줄 Enter\n")

    while True:
        lines = []
        while True:
            line = input("> ")
            if not line:
                break
            lines.append(line)
        answer = "\n".join(lines).strip()

        if is_denial(answer):
            print_memory_prompts()
            seed = input("> ").strip()
            if seed:
                return seed
            continue

        ok, message = validate_user_dump(answer, cfg)
        if ok:
            return answer
        print(message)


def _analyze_with_llm(question: str, competency: str, answer: str, cfg: dict):
    fallback = rule_analyze(question, answer, cfg)
    prompt = build_extraction_prompt(question, competency, answer, classify_input(answer, cfg))
    runtime = cfg["runtime"]
    if runtime.get("provider") != "ollama":
        return fallback

    try:
        raw = models.ollama_chat(prompt, runtime, json_format=True)
    except RuntimeError as exc:
        if runtime.get("allow_rule_fallback", True):
            print(f"\n[안내] 로컬 LLM 분석 실패 → rule 기반으로 진행합니다. ({exc})")
            return fallback
        raise
    return parse_analysis_json(raw, fallback)


def _ask_one_followup(analysis) -> str:
    if not analysis.followup_question:
        return ""
    print("\n부족한 핵심 정보 1개만 보강해 주세요.")
    print(analysis.followup_question)
    return input("> ").strip()


def _rewrite_with_llm(question: str, competency: str, original: str, analysis, followup: str, cfg: dict) -> tuple[str, list[str]]:
    runtime = cfg["runtime"]
    target_chars = target_chars_for(analysis.input_type, cfg)
    prompt = build_rewrite_prompt(question, competency, original, analysis, followup, target_chars)

    if runtime.get("provider") != "ollama":
        return _rule_rewrite(original, analysis, followup), analysis.improvement_summary

    try:
        raw = models.ollama_chat(prompt, runtime, json_format=False)
    except RuntimeError as exc:
        if runtime.get("allow_rule_fallback", True):
            print(f"\n[안내] 로컬 LLM 재작성 실패 → rule 기반 문단으로 진행합니다. ({exc})")
            return _rule_rewrite(original, analysis, followup), analysis.improvement_summary
        raise
    return split_rewrite_response(raw, analysis.improvement_summary)


def _rule_rewrite(original: str, analysis, followup: str) -> str:
    if analysis.input_type == "seed":
        return f"{analysis.core_experience} 경험을 자소서 소재로 저장했습니다. 추후 상황, 행동, 결과를 보강해 완성 문단으로 확장할 수 있습니다."

    parts = []
    if analysis.situation:
        parts.append(analysis.situation)
    if analysis.task:
        parts.append(f"이 과정에서 저는 {analysis.task}")
    if analysis.action:
        parts.append(f"{analysis.action}을 중심으로 문제를 해결했습니다.")
    if followup:
        parts.append(f"추가로 {followup}")
    if analysis.result:
        parts.append(f"그 결과 {analysis.result}라는 성과를 만들었습니다.")
    if not parts:
        parts.append(original)
    return " ".join(parts)


def _maybe_retry(question: str, competency: str, original: str, analysis, rewritten: str, score_before: float, score_after: float, cfg: dict):
    threshold = cfg["interviewer"]["retry_delta_threshold"]
    if score_after - score_before >= threshold * 100:
        return rewritten, score_after, False

    print(f"\n재평가 결과: 부합도 {score_before:.0f} -> {score_after:.0f} (상승폭 미미)")
    print("입력하신 답변에 역량을 증명할 구체적인 도구나 정량적 수치가 부족할 수 있습니다.")
    print("[y] 그냥 수락  [r] 한 번만 보충  [s] 원문 저장")
    choice = input("선택 (y/r/s): ").strip().lower() or "y"
    if choice == "s":
        return original, score_before, False
    if choice != "r":
        return rewritten, score_after, False

    extra = _ask_one_followup(analysis)
    retry_text, summary = _rewrite_with_llm(question, competency, original, analysis, extra, cfg)
    retry_score = estimate_score(question, retry_text, analysis)
    analysis.improvement_summary = summary
    return retry_text, retry_score, True


def _build_card(session: Session, original: str, rewritten: str, analysis, score_before: float, score_after: float, summary: list[str]) -> dict:
    return {
        "question": session.question,
        "core_experience": analysis.core_experience,
        "input_type": analysis.input_type,
        "star": {
            "situation": analysis.situation,
            "task": analysis.task,
            "action": analysis.action,
            "result": analysis.result,
        },
        "tools": analysis.tools,
        "metrics": analysis.metrics,
        "competencies": sorted(set([session.competency] + analysis.reuse_angles)),
        "before_score": round(score_before, 1),
        "after_score": round(score_after, 1),
        "final_answer": rewritten,
        "improvement_summary": summary[:3],
        "versions": [
            {"type": "original", "text": original},
            {"type": "rewritten", "text": rewritten},
        ],
        "reuse_angles": analysis.reuse_angles,
        "source_cards": analysis.source_cards,
    }


def _print_card_result(card: dict) -> None:
    print("\n자소서 카드 저장 내용")
    print(f"  핵심 경험: {card['core_experience']}")
    print(f"  입력 타입: {card['input_type']}")
    if card["tools"]:
        print(f"  기술/도구: {', '.join(card['tools'])}")
    if card["metrics"]:
        print(f"  수치/성과: {', '.join(card['metrics'])}")
    print(f"  점수 변화: {card['before_score']:.0f} -> {card['after_score']:.0f}")
    print("\n개선 요약")
    for item in card.get("improvement_summary", [])[:3]:
        print(f"  - {item}")


def _show_summary(session: Session, path: str) -> None:
    print(f"\n{'=' * 60}")
    print("저장 완료")
    print(f"날짜      : {session.date}")
    print(f"역량      : {session.competency}")
    print(f"저장 위치 : {path}")
    print(f"{'=' * 60}\n")


def run_session(gen_model, gen_tokenizer, analysis_model: models.AnalysisModel | None):
    cfg = load_config()
    stored = load_all()
    readiness_before = calculate_readiness(stored)

    if cfg["dashboard"].get("show_readiness_map", True):
        print_readiness("현재 자소서 준비도", readiness_before)
        area = weakest_area(readiness_before)
        print(f"\n오늘 추천 미션: {mission_for(area)}")

    session = Session()
    competency = random.choice(COMPETENCIES)
    question = models.generate_question(gen_model, gen_tokenizer, competency)
    session.competency = competency
    session.question = question

    _show_question(competency, question)
    if cfg["memory"].get("enable_semantic_search", True):
        cards = recommend_cards(question, stored, cfg["memory"].get("max_recommendations", 3))
        print_recommendations(cards)

    original = _read_free_dump(cfg)
    session.answer = original

    coach_analysis = _analyze_with_llm(question, competency, original, cfg)
    session.input_type = coach_analysis.input_type

    score_before = estimate_score(question, original, coach_analysis)
    followup = ""
    if coach_analysis.input_type != "seed":
        followup = _ask_one_followup(coach_analysis)

    rewritten, summary = _rewrite_with_llm(question, competency, original, coach_analysis, followup, cfg)
    score_after = estimate_score(question, rewritten, coach_analysis)
    rewritten, score_after, retry_used = _maybe_retry(
        question, competency, original, coach_analysis, rewritten, score_before, score_after, cfg
    )

    if analysis_model:
        score_after, job = analysis_model.predict(question, rewritten)
    else:
        job = _guess_job(coach_analysis)

    session.score_before = round(score_before, 1)
    session.score_after = round(score_after, 1)
    session.intent_score = round(score_after, 1)
    session.retry_used = retry_used
    session.suitable_jobs = [job]
    session.final_answer = rewritten
    session.add_turn(coach_analysis.followup_question, followup)

    card = _build_card(session, original, rewritten, coach_analysis, score_before, score_after, summary)
    session.set_answer_card(card)
    _print_card_result(card)

    path = save_session(session.to_dict())
    readiness_after = calculate_readiness(load_all())
    if cfg["dashboard"].get("show_readiness_map", True):
        print_readiness_delta(readiness_before, readiness_after)
    _show_summary(session, path)


def _guess_job(analysis) -> str:
    tools = {tool.lower() for tool in analysis.tools}
    if tools & {"sql", "pandas", "numpy", "python"}:
        return "ai_ml"
    if tools & {"node.js", "java", "spring", "api", "docker", "mysql"}:
        return "backend"
    return "product"


def main():
    if "--history" in sys.argv:
        print_history()
        return

    print("\n모델 로딩 중...")
    gen_model, gen_tokenizer = models.load_generation_model(USE_STAGE14_FINETUNED)
    analysis = models.load_analysis_model(USE_STAGE3_FINETUNED)
    run_session(gen_model, gen_tokenizer, analysis)


if __name__ == "__main__":
    main()
