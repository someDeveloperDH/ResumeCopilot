"""
하루 1 문답 CLI (test4 통합본)

실행:
    python cli/main.py            # 새 세션 시작
    python cli/main.py --history  # 저장 이력 조회

세 브랜치(test3 / jintae_v2 / jinyoung_v2)의 장점을 한 흐름으로 결합한다.
세부 출처는 MERGE_NOTES.md 참고. 핵심 결합 지점만 요약하면:

  1) 준비도 대시보드 + 추천 미션                          (jintae_v2 readiness.py)
  2) 과거 카드 의미 검색 추천                              (jintae_v2 memory.py)
  3) 파일 편집기 기반 멀티턴 입력 (conversation.txt 누적)   (jinyoung_v2)
  4) STAR 추출 → 1차 재작성 → 재시도 코칭 루프              (jintae_v2 coach.py + Ollama)
  5) 꼬리질문 인터뷰 루프(최대 5턴, 전체 대화 맥락 주입)     (jinyoung_v2)
  6) 매 턴 일관성 검사(rule + NLI + SBERT) 및 메모 주입     (jinyoung_v2 → consistency.py)
  7) 분석 모델(intent_score/직무) 예측 + 직무 변화 추적     (test3/jintae_v2 + jinyoung_v2)
  8) 자소서 카드(JSON) + 사람이 읽는 최종 txt 동시 저장     (jintae_v2 카드 + jinyoung_v2 txt)
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import models
from agents import (
    AnalysisAgent,
    CoachAgent,
    ConsistencyAgent,
    InterviewContext,
    InterviewerAgent,
    QuestionAgent,
)
from coach import estimate_score, load_config
from consistency import consistency_note_for_prompt, show_consistency
from memory import print_recommendations, recommend_cards
from readiness import calculate_readiness, mission_for, print_readiness, print_readiness_delta, weakest_area
from session import Session
from storage import STORAGE_DIR, load_all, print_history, save_session


USE_STAGE14_FINETUNED = True
USE_STAGE3_FINETUNED = True

MAX_TAIL_TURNS = 5
MIN_ANSWER_LEN = 30
COMPETENCIES = ["문제해결", "협업", "성장", "실패경험", "주도성"]
DRAFTS_DIR = Path(__file__).parent.parent / "storage" / "drafts"


# ── 질문 표시 ─────────────────────────────────────────────────────────
def _question_intents(competency: str) -> list[str]:
    mapping = {
        "문제해결": ["문제를 정의한 방식", "본인이 취한 구체적 행동", "개선 결과나 수치"],
        "협업": ["갈등이나 조율 상황", "본인의 커뮤니케이션 방식", "팀 결과에 준 영향"],
        "성장": ["처음 부족했던 지점", "학습/시도 과정", "이후 달라진 결과"],
        "실패경험": ["실패 원인", "재발 방지 행동", "실패 이후 바뀐 방식"],
        "주도성": ["스스로 발견한 문제", "먼저 제안/실행한 행동", "주변이나 결과에 준 변화"],
    }
    return mapping.get(competency, ["상황", "본인의 역할", "행동과 결과"])


def _show_question(competency: str, question: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  오늘의 질문  [{competency}]")
    print(f"{'=' * 60}")
    print(f"  {question}")
    print("\n  이 질문이 보고 싶은 것")
    for item in _question_intents(competency):
        print(f"    - {item}")
    print(f"{'=' * 60}\n")


# ── 파일 편집기 기반 멀티턴 입력 (jinyoung_v2) ─────────────────────────
def _open_editor(filepath: Path):
    """Windows 우선: notepad → $EDITOR → nano → vi → 수동 안내"""
    editor = os.environ.get("EDITOR", "")
    candidates = ["notepad", editor, "nano", "vi"] if os.name == "nt" else [editor, "nano", "vi"]

    for cmd in [c for c in candidates if c]:
        try:
            subprocess.run([cmd, str(filepath)], check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print("\n  에디터를 자동으로 열 수 없습니다.")
    print("  아래 파일을 직접 열어 답변을 작성하고 저장한 뒤 Enter를 눌러주세요.")
    print(f"  {filepath}\n")
    input("  [완료 후 Enter] ")


def _conversation_file_path(session_id: str) -> Path:
    """세션 전체 답변을 하나의 txt 파일에 누적 저장한다."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    return DRAFTS_DIR / f"{session_id}_conversation.txt"


def _extract_section_answer(raw: str, marker: str) -> str:
    """marker 다음부터 다음 섹션 시작 전까지 답변만 추출한다."""
    if marker not in raw:
        return ""

    text = raw.split(marker, 1)[1]
    next_headers = ["\n[꼬리질문 ", "\n[답변 ", "\n[최종", "\n---"]
    cut_positions = [text.find(h) for h in next_headers if text.find(h) != -1]
    if cut_positions:
        text = text[:min(cut_positions)]

    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    return "\n".join(lines).strip()


def _open_and_read_section(*, path: Path, marker: str, min_len: int, empty_message: str = "입력이 비어 있습니다.") -> str:
    """같은 conversation txt 파일을 열고, marker 아래 답변만 읽는다."""
    print("\n  입력 파일이 열립니다. 작성 후 저장하고 닫아주세요.")
    print(f"  파일 위치: {path}\n")

    _open_editor(path)

    raw = path.read_text(encoding="utf-8")
    answer = _extract_section_answer(raw, marker)

    if not answer:
        print(f"\n  {empty_message} 파일을 다시 열어 작성해주세요.\n")
        return _open_and_read_section(path=path, marker=marker, min_len=min_len, empty_message=empty_message)

    if len(answer) < min_len:
        print(f"\n  너무 짧습니다 (현재 {len(answer)}자, 최소 {min_len}자).")
        print("  같은 파일을 다시 열어 보완해주세요.\n")
        return _open_and_read_section(path=path, marker=marker, min_len=min_len, empty_message=empty_message)

    print(f"  입력 확인 ({len(answer)}자)\n")
    return answer


def _init_conversation_file(question: str, session_id: str, competency: str) -> Path:
    """최초 질문/답변용 conversation txt 파일을 1개만 만든다."""
    path = _conversation_file_path(session_id)
    template = (
        "# ───────────────────────────────────────────────────────\n"
        "# 하루 1 문답 - 전체 대화 작성 파일\n"
        "# ───────────────────────────────────────────────────────\n"
        "# 안내:\n"
        "#   - 이 파일 하나에 최초 답변과 모든 꼬리질문 답변을 이어서 작성합니다.\n"
        "#   - 꼬리질문이 생기면 이 파일 맨 아래에 자동으로 추가됩니다.\n"
        "#   - 기존 내용은 지우지 말고, 새 [답변 n] 아래에만 작성하세요.\n"
        "#   - '#' 으로 시작하는 줄은 답변에서 제외됩니다.\n"
        "# ───────────────────────────────────────────────────────\n\n"
        f"[역량]\n{competency}\n\n"
        f"[원본 질문]\n{question}\n\n"
        "[최초 답변]\n"
    )
    path.write_text(template, encoding="utf-8")
    return path


def _get_answer_via_file(question: str, session_id: str, competency: str) -> tuple[str, Path]:
    path = _init_conversation_file(question, session_id, competency)
    answer = _open_and_read_section(
        path=path, marker="[최초 답변]", min_len=MIN_ANSWER_LEN, empty_message="최초 답변이 비어 있습니다."
    )
    return answer, path


def _append_tail_question(path: Path, tail_question: str, turn: int) -> str:
    marker = f"[답변 {turn}]"
    current = path.read_text(encoding="utf-8")
    if f"[꼬리질문 {turn}]" not in current:
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n\n----------------------------------------\n"
                f"[꼬리질문 {turn}]\n{tail_question}\n\n{marker}\n"
            )
    return marker


def _get_tail_reply_via_file(tail_question: str, conversation_path: Path, turn: int) -> str:
    marker = _append_tail_question(conversation_path, tail_question, turn)
    return _open_and_read_section(
        path=conversation_path, marker=marker, min_len=10, empty_message=f"꼬리질문 {turn} 답변이 비어 있습니다."
    )


def build_conversation_context(session: Session, current_answer: str | None = None) -> str:
    lines = [
        f"[원본 질문] {session.question}",
        f"[최초 답변] {session.answer}",
    ]
    for i, turn in enumerate(session.conversation, start=1):
        lines.append(f"[꼬리질문 {i}] {turn['tail_question']}")
        lines.append(f"[답변 {i}] {turn['response']}")
    if current_answer and (not session.conversation or session.conversation[-1].get("response") != current_answer):
        lines.append(f"[현재 답변] {current_answer}")
    return "\n".join(lines)


# ── STAR 코칭 (CoachAgent 호출 + 사용자 인터랙션) ───────────────────────
def _ask_one_followup(analysis) -> str:
    if not analysis.followup_question:
        return ""
    print("\n  부족한 핵심 정보 1개만 보강해 주세요.")
    print(f"  {analysis.followup_question}")
    return input("  > ").strip()


def _maybe_retry(question, competency, original, analysis, rewritten, score_before, score_after, cfg, coach_agent):
    threshold = cfg["interviewer"]["retry_delta_threshold"]
    if score_after - score_before >= threshold * 100:
        return rewritten, score_after, False

    print(f"\n  재평가 결과: 부합도 {score_before:.0f} -> {score_after:.0f} (상승폭 미미)")
    print("  입력하신 답변에 역량을 증명할 구체적인 도구나 정량적 수치가 부족할 수 있습니다.")
    print("  [y] 그냥 수락  [r] 한 번만 보충  [s] 원문 저장")
    choice = input("  선택 (y/r/s): ").strip().lower() or "y"
    if choice == "s":
        return original, score_before, False
    if choice != "r":
        return rewritten, score_after, False

    extra = _ask_one_followup(analysis)
    retry = coach_agent.rewrite(question, competency, original, analysis, extra).data
    retry_score = estimate_score(question, retry.text, analysis)
    analysis.improvement_summary = retry.improvement_summary
    return retry.text, retry_score, True


def _build_card(session, original, final_text, analysis, score_before, score_after, summary, coached: str | None = None):
    versions = [{"type": "original", "text": original}]
    if coached and coached != original and coached != final_text:
        versions.append({"type": "coached", "text": coached})
    versions.append({"type": "final", "text": final_text})

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
        "final_answer": final_text,
        "improvement_summary": summary[:3],
        "versions": versions,
        "reuse_angles": analysis.reuse_angles,
        "source_cards": analysis.source_cards,
    }


# ── 분석/일관성/추세 표시 ────────────────────────────────────────────
def _show_analysis(intent_score: float, job: str):
    score = max(0, min(100, int(intent_score)))
    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
    print("\n  분석 결과")
    print(f"  의도 반영도 [{bar}] {score}/100")
    print(f"  적합 직무   {job}\n")


def _print_card_result(card: dict) -> None:
    print("\n  자소서 카드 저장 내용")
    print(f"    핵심 경험 : {card['core_experience']}")
    print(f"    입력 타입 : {card['input_type']}")
    if card["tools"]:
        print(f"    기술/도구 : {', '.join(card['tools'])}")
    if card["metrics"]:
        print(f"    수치/성과 : {', '.join(card['metrics'])}")
    print(f"    STAR 부합도 변화 : {card['before_score']:.0f} -> {card['after_score']:.0f}")
    print("\n  개선 요약")
    for item in card.get("improvement_summary", [])[:3]:
        print(f"    - {item}")


def _show_job_trend(session: Session):
    history = session.job_history
    print(f"\n{'=' * 55}")
    print("  적합 직무 변화 추적")
    print(f"{'=' * 55}")

    if not history:
        print("  기록된 직무 예측이 없습니다.")
        print(f"{'=' * 55}\n")
        return

    for h in history:
        print(f"  Turn {h['turn']:>2} | {h['source']:<10} | {h['job']:<8} | intent {h['intent_score']:>5.1f}/100")

    first, last = history[0], history[-1]
    delta = last["intent_score"] - first["intent_score"]
    sign = "+" if delta >= 0 else ""
    counter = Counter(h["job"] for h in history)
    top_jobs = [job for job, _ in counter.most_common(3)]

    print("-" * 55)
    print(f"  시작 직무 → 최종 직무 : {first['job']} → {last['job']}")
    print(f"  의도 반영도 변화       : {first['intent_score']:.1f} → {last['intent_score']:.1f} ({sign}{delta:.1f})")
    print(f"  세션 내 TOP 직무       : {' > '.join(top_jobs)}")

    if first["job"] != last["job"]:
        print("  해석                  : 답변 보완 과정에서 강조되는 직무 방향이 바뀌었습니다.")
    elif abs(delta) >= 10:
        print("  해석                  : 직무 방향은 유지됐고, 의도 반영도 변화가 크게 나타났습니다.")
    else:
        print("  해석                  : 직무 방향과 점수가 비교적 안정적으로 유지됐습니다.")
    print(f"{'=' * 55}\n")


# ── 저장 (JSON + 사람이 읽는 최종 txt) ───────────────────────────────
def _save_final_text(session: Session, conversation_path: Path, card: dict) -> str:
    """JSON과 별도로 사람이 읽기 좋은 최종 txt 파일을 저장한다 (대화 전문 + 카드 요약 + 직무 변화)."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    final_path = Path(STORAGE_DIR) / f"{session.session_id}_final.txt"
    conversation_text = conversation_path.read_text(encoding="utf-8") if conversation_path.exists() else session.conversation_context

    job_history_lines = [
        f"Turn {h['turn']} | {h['source']} | {h['job']} | intent {h['intent_score']}/100"
        for h in session.job_history
    ]
    star = card.get("star", {})

    final_text = (
        f"{conversation_text.rstrip()}\n\n"
        f"========================================\n"
        f"[최종 요약]\n"
        f"날짜: {session.date}\n"
        f"역량: {session.competency}\n"
        f"입력 타입: {session.input_type}\n"
        f"STAR 부합도 변화: {session.score_before:.1f} -> {session.score_after:.1f}"
        f"{' (재시도 1회)' if session.retry_used else ''}\n"
        f"최종 적합 직무: {' > '.join(session.suitable_jobs)}\n"
        f"최종 의도 반영도: {session.intent_score:.1f}/100\n\n"
        f"[STAR 구조]\n"
        f"S(상황): {star.get('situation') or '-'}\n"
        f"T(과제): {star.get('task') or '-'}\n"
        f"A(행동): {star.get('action') or '-'}\n"
        f"R(결과): {star.get('result') or '-'}\n"
        f"기술/도구: {', '.join(card.get('tools', [])) or '-'}\n"
        f"수치/성과: {', '.join(card.get('metrics', [])) or '-'}\n\n"
        f"[다듬어진 최종 답변]\n{card.get('final_answer', '')}\n\n"
        f"[적합 직무 변화 추적]\n"
        f"{chr(10).join(job_history_lines) if job_history_lines else '기록 없음'}\n"
    )
    final_path.write_text(final_text, encoding="utf-8")
    return str(final_path)


def _show_summary(session: Session, json_path: str, final_txt_path: str) -> None:
    print(f"\n{'=' * 55}")
    print("  저장 완료!")
    print(f"  날짜        : {session.date}")
    print(f"  역량        : {session.competency}")
    print(f"  STAR 부합도 : {session.score_before:.0f} -> {session.score_after:.0f}")
    print(f"  적합 직무   : {' > '.join(session.suitable_jobs)}")
    print(f"  의도 반영도 : {session.intent_score:.0f}/100")
    print(f"  JSON 저장   : {json_path}")
    print(f"  최종 TXT    : {final_txt_path}")
    print(f"{'=' * 55}\n")


# ── 메인 세션 흐름 ────────────────────────────────────────────────────
def run_session(gen_model, gen_tokenizer, analysis_model: models.AnalysisModel | None):
    cfg = load_config()
    smart_cfg = cfg["models"]["smart"]

    # 서브에이전트 생성 (역할별 고정 모델 티어 — MERGE_NOTES.md 및 plan 참고)
    question_agent = QuestionAgent(gen_model, gen_tokenizer)
    interviewer_agent = InterviewerAgent(smart_cfg, gen_model, gen_tokenizer)
    coach_agent = CoachAgent(smart_cfg, cfg)
    analysis_agent = AnalysisAgent(analysis_model)
    consistency_agent = ConsistencyAgent()

    stored = load_all()
    readiness_before = calculate_readiness(stored)

    if cfg["dashboard"].get("show_readiness_map", True):
        print_readiness("현재 자소서 준비도", readiness_before)
        area = weakest_area(readiness_before)
        print(f"\n  오늘 추천 미션: {mission_for(area)}")

    session = Session()
    competency = random.choice(COMPETENCIES)
    question = question_agent.ask(competency).data
    session.competency = competency
    session.question = question
    _show_question(competency, question)

    if cfg["memory"].get("enable_semantic_search", True):
        cards = recommend_cards(question, stored, cfg["memory"].get("max_recommendations", 3))
        print_recommendations(cards)

    # 1) 파일 편집기에서 최초 답변 받기 (jinyoung_v2 멀티턴 입력 방식)
    answer, conversation_path = _get_answer_via_file(question, session.session_id, competency)
    session.answer = answer
    session.final_answer = answer

    # 2) 분석 모델로 의도 반영도 / 적합 직무 1차 예측
    analysis_output = analysis_agent.predict(question, answer).data
    intent_score, job = analysis_output.intent_score, analysis_output.job
    session.intent_score = intent_score
    session.suitable_jobs = [job]
    session.add_job_history(0, "initial", intent_score, job)
    _show_analysis(intent_score, job)

    last_consistency = consistency_agent.check(session, answer).data
    session.add_consistency_check(0, last_consistency)
    show_consistency(last_consistency)

    # 3) STAR 추출 → 1차 재작성 → 재시도 코칭 (CoachAgent: smart 모델 우선, rule 폴백)
    coach_analysis = coach_agent.analyze(question, competency, answer).data
    session.input_type = coach_analysis.input_type

    score_before = estimate_score(question, answer, coach_analysis)
    followup = ""
    if coach_analysis.input_type != "seed":
        followup = _ask_one_followup(coach_analysis)

    rewrite_result = coach_agent.rewrite(question, competency, answer, coach_analysis, followup).data
    coached, summary = rewrite_result.text, rewrite_result.improvement_summary
    score_after = estimate_score(question, coached, coach_analysis)
    coached, score_after, retry_used = _maybe_retry(
        question, competency, answer, coach_analysis, coached, score_before, score_after, cfg, coach_agent
    )
    session.score_before = round(score_before, 1)
    session.score_after = round(score_after, 1)
    session.retry_used = retry_used

    current = coached
    session.final_answer = current

    # 4) 꼬리질문 인터뷰 루프 - 다듬어진 답변을 바탕으로 더 깊이 파고든다 (jinyoung_v2 구조)
    for turn in range(1, MAX_TAIL_TURNS + 1):
        conversation_context = build_conversation_context(session, current)
        tail_ctx = InterviewContext(
            question=question, answer=current,
            intent_score=intent_score, job=job,
            conversation_context=conversation_context,
            consistency_note=consistency_note_for_prompt(last_consistency),
        )
        tail_q = interviewer_agent.ask_followup(tail_ctx).data
        print(f"\n  꼬리질문 [{turn}/{MAX_TAIL_TURNS}]: {tail_q}")
        reply = _get_tail_reply_via_file(tail_q, conversation_path, turn)
        if not reply:
            break

        session.add_turn(tail_q, reply)
        current = reply
        session.final_answer = current

        last_consistency = consistency_agent.check(session, current).data
        session.add_consistency_check(turn, last_consistency)
        show_consistency(last_consistency)

        analysis_output = analysis_agent.predict(question, current).data
        intent_score, job = analysis_output.intent_score, analysis_output.job
        if job not in session.suitable_jobs:
            session.suitable_jobs.append(job)
        session.intent_score = intent_score
        session.add_job_history(turn, f"tail_{turn}", intent_score, job)
        _show_analysis(intent_score, job)

        if input("  저장하고 종료? (y/n): ").strip().lower() == "y":
            break

    session.conversation_context = build_conversation_context(session, current)

    # 5) 최종 자소서 카드 빌드 (원문 → STAR 코칭본 → 인터뷰 최종본)
    card = _build_card(session, answer, current, coach_analysis, session.score_before, session.score_after, summary, coached=coached)
    session.set_answer_card(card)
    _print_card_result(card)
    _show_job_trend(session)

    json_path = save_session(session.to_dict())
    final_txt_path = _save_final_text(session, conversation_path, card)

    readiness_after = calculate_readiness(load_all())
    if cfg["dashboard"].get("show_readiness_map", True):
        print_readiness_delta(readiness_before, readiness_after)
    _show_summary(session, json_path, final_txt_path)


def main():
    if "--history" in sys.argv:
        print_history()
        return

    print(f"\n모델 로딩 중... (device={models.DEVICE})")
    gen_model, gen_tokenizer = models.load_generation_model(USE_STAGE14_FINETUNED)
    analysis = models.load_analysis_model(USE_STAGE3_FINETUNED)
    run_session(gen_model, gen_tokenizer, analysis)


if __name__ == "__main__":
    main()
