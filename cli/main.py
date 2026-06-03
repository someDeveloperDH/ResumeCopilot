"""
하루 1 문답 CLI - CPU 버전 + 대화 맥락/일관성 검사/직무 변화 추적

실행:
    python cli/main.py
    python cli/main.py --history

추가 기능:
- 꼬리질문 생성 시 최초 답변 + 모든 꼬리질문/답변을 conversation_context로 사용
- 꼬리질문 답변도 최초 답변과 동일하게 메모장/txt 파일로 입력
- 생성 모델을 Judge처럼 사용해 conversation_context 기반 일관성 검사
- 세션 종료 시 적합 직무와 intent_score 변화 추적 출력
"""

import os
import sys
import random
import subprocess
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import models
from session import Session
from storage import save_session, print_history, STORAGE_DIR

USE_STAGE14_FINETUNED = True
USE_STAGE3_FINETUNED  = True

MAX_TAIL_TURNS = 5
MIN_ANSWER_LEN = 30
COMPETENCIES   = ["문제해결", "협업", "성장", "실패경험", "주도성"]
DRAFTS_DIR     = Path(__file__).parent.parent / "storage" / "drafts"

# 단순 휴리스틱 기반 일관성 검사 키워드
LEADER_WORDS = ["리더", "팀장", "주도", "총괄", "PM", "책임", "이끌"]
LOW_AUTHORITY_WORDS = ["결정권", "권한", "시키는", "보조", "따라", "참여만", "맡은 게 없", "기여가 적"]
SOLO_WORDS = ["혼자", "단독", "개인", "스스로"]
TEAM_WORDS = ["팀", "협업", "함께", "동료", "조원"]
VAGUE_WORDS = ["열심히", "노력", "잘", "많이", "최선을", "좋은 결과", "성공적"]


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

    print(f"\n  에디터를 자동으로 열 수 없습니다.")
    print(f"  아래 파일을 직접 열어 답변을 작성하고 저장한 뒤 Enter를 눌러주세요.")
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

    # 다음 질문/답변 섹션이 생긴 경우 그 전까지만 읽는다.
    next_headers = [
        "\n[꼬리질문 ",
        "\n[답변 ",
        "\n[최종",
        "\n---",
    ]
    cut_positions = [text.find(h) for h in next_headers if text.find(h) != -1]
    if cut_positions:
        text = text[:min(cut_positions)]

    # 안내 주석은 무시한다.
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    return "\n".join(lines).strip()


def _open_and_read_section(
    *,
    path: Path,
    marker: str,
    min_len: int,
    empty_message: str = "입력이 비어 있습니다.",
) -> str:
    """같은 conversation txt 파일을 열고, marker 아래 답변만 읽는다."""
    print(f"\n  입력 파일이 열립니다. 작성 후 저장하고 닫아주세요.")
    print(f"  파일 위치: {path}\n")

    _open_editor(path)

    raw = path.read_text(encoding="utf-8")
    answer = _extract_section_answer(raw, marker)

    if not answer:
        print(f"\n  {empty_message} 파일을 다시 열어 작성해주세요.\n")
        return _open_and_read_section(
            path=path,
            marker=marker,
            min_len=min_len,
            empty_message=empty_message,
        )

    if len(answer) < min_len:
        print(f"\n  너무 짧습니다 (현재 {len(answer)}자, 최소 {min_len}자).")
        print("  같은 파일을 다시 열어 보완해주세요.\n")
        return _open_and_read_section(
            path=path,
            marker=marker,
            min_len=min_len,
            empty_message=empty_message,
        )

    print(f"  입력 확인 ({len(answer)}자)\n")
    return answer


def _init_conversation_file(question: str, session_id: str, competency: str) -> Path:
    """최초 질문/답변용 conversation txt 파일을 1개만 만든다."""
    path = _conversation_file_path(session_id)
    template = (
        f"# ───────────────────────────────────────────────────────\n"
        f"# 하루 1 문답 - 전체 대화 작성 파일\n"
        f"# ───────────────────────────────────────────────────────\n"
        f"# 안내:\n"
        f"#   - 이 파일 하나에 최초 답변과 모든 꼬리질문 답변을 이어서 작성합니다.\n"
        f"#   - 꼬리질문이 생기면 이 파일 맨 아래에 자동으로 추가됩니다.\n"
        f"#   - 기존 내용은 지우지 말고, 새 [답변 n] 아래에만 작성하세요.\n"
        f"#   - '#' 으로 시작하는 줄은 답변에서 제외됩니다.\n"
        f"# ───────────────────────────────────────────────────────\n\n"
        f"[역량]\n{competency}\n\n"
        f"[원본 질문]\n{question}\n\n"
        f"[최초 답변]\n"
    )
    path.write_text(template, encoding="utf-8")
    return path


def _get_answer_via_file(question: str, session_id: str, competency: str) -> tuple[str, Path]:
    """최초 답변도 conversation txt 하나에서 받는다."""
    path = _init_conversation_file(question, session_id, competency)
    answer = _open_and_read_section(
        path=path,
        marker="[최초 답변]",
        min_len=MIN_ANSWER_LEN,
        empty_message="최초 답변이 비어 있습니다.",
    )
    return answer, path


def _append_tail_question(path: Path, tail_question: str, turn: int) -> str:
    """기존 conversation txt 맨 아래에 꼬리질문과 답변 칸을 추가한다."""
    marker = f"[답변 {turn}]"
    current = path.read_text(encoding="utf-8")

    # 같은 turn을 실수로 중복 append하지 않도록 방지
    if f"[꼬리질문 {turn}]" not in current:
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n\n----------------------------------------\n"
                f"[꼬리질문 {turn}]\n"
                f"{tail_question}\n\n"
                f"{marker}\n"
            )
    return marker


def _get_tail_reply_via_file(tail_question: str, conversation_path: Path, turn: int) -> str:
    """꼬리질문 답변도 같은 conversation txt 파일에서 이어서 받는다."""
    marker = _append_tail_question(conversation_path, tail_question, turn)
    return _open_and_read_section(
        path=conversation_path,
        marker=marker,
        min_len=10,
        empty_message=f"꼬리질문 {turn} 답변이 비어 있습니다.",
    )


def _save_final_text(session: Session, conversation_path: Path) -> str:
    """JSON과 별도로 사람이 읽기 좋은 최종 txt 파일도 하나로 저장한다."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    final_path = Path(STORAGE_DIR) / f"{session.session_id}_final.txt"
    conversation_text = conversation_path.read_text(encoding="utf-8") if conversation_path.exists() else session.conversation_context

    job_history_lines = []
    for h in session.job_history:
        job_history_lines.append(
            f"Turn {h['turn']} | {h['source']} | {h['job']} | intent {h['intent_score']}/100"
        )

    final_text = (
        f"{conversation_text.rstrip()}\n\n"
        f"========================================\n"
        f"[최종 요약]\n"
        f"날짜: {session.date}\n"
        f"역량: {session.competency}\n"
        f"최종 적합 직무: {' > '.join(session.suitable_jobs)}\n"
        f"최종 의도 반영도: {session.intent_score:.1f}/100\n\n"
        f"[적합 직무 변화 추적]\n"
        f"{chr(10).join(job_history_lines) if job_history_lines else '기록 없음'}\n"
    )
    final_path.write_text(final_text, encoding="utf-8")
    return str(final_path)

def _show_question(competency: str, question: str):
    print(f"\n{'='*55}")
    print(f"  오늘의 질문  [{competency}]")
    print(f"{'='*55}")
    print(f"  {question}")
    print(f"{'='*55}\n")


def _show_analysis(intent_score: float, job: str):
    score = max(0, min(100, int(intent_score)))
    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
    print(f"\n  분석 결과")
    print(f"  의도 반영도 [{bar}] {score}/100")
    print(f"  적합 직무   {job}\n")


def _fallback_analysis() -> tuple[float, str]:
    print("  [분석 대체값] stage3 CPU 모델 파일이 없어 임시값 사용")
    return 50.0, random.choice(["backend", "ai_ml", "product"])


def _contains_any(text: str, words: list[str]) -> bool:
    return any(w.lower() in text.lower() for w in words)


def build_conversation_context(session: Session, current_answer: str | None = None) -> str:
    """최초 답변과 모든 꼬리질문/답변을 하나의 맥락 문자열로 합침."""
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


def check_consistency_with_model(
    gen_model,
    gen_tokenizer,
    session: Session,
    latest_answer: str,
) -> dict:
    """대화 전체를 생성 모델에 넣어 일관성 검사 결과를 받는다."""
    conversation_context = build_conversation_context(session, latest_answer)
    try:
        return models.assess_consistency(
            gen_model,
            gen_tokenizer,
            conversation_context=conversation_context,
            latest_answer=latest_answer,
        )
    except Exception as e:
        return {
            "status": "주의",
            "issues": [f"일관성 검사 실행 중 오류가 발생했습니다: {e}"],
            "suggestions": ["저장된 대화 맥락을 수동으로 확인해 주세요."],
            "method": "llm_judge_error",
        }


def _show_consistency(result: dict):
    print("\n  일관성 검사")
    if result["status"] == "양호":
        print("  ✓ 현재까지 큰 모순은 감지되지 않았습니다.\n")
        return
    print("  ⚠ 확인 필요")
    for issue in result["issues"]:
        print(f"  - {issue}")
    if result["suggestions"]:
        print("  보완 방향")
        for sug in result["suggestions"]:
            print(f"  → {sug}")
    print()


def _consistency_note_for_prompt(result: dict) -> str:
    if result["status"] == "양호":
        return "현재까지 큰 모순은 없지만, 이전 답변을 반복하지 말고 더 깊은 정보를 물어볼 것."
    parts = []
    parts.extend(result.get("issues", []))
    parts.extend(result.get("suggestions", []))
    return " / ".join(parts)


def _show_job_trend(session: Session):
    history = session.job_history
    print(f"\n{'='*55}")
    print("  적합 직무 변화 추적")
    print(f"{'='*55}")

    if not history:
        print("  기록된 직무 예측이 없습니다.")
        print(f"{'='*55}\n")
        return

    for h in history:
        print(f"  Turn {h['turn']:>2} | {h['source']:<10} | {h['job']:<8} | intent {h['intent_score']:>5.1f}/100")

    first = history[0]
    last = history[-1]
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
    print(f"{'='*55}\n")


def _show_summary(session: Session, json_path: str, final_txt_path: str):
    print(f"\n{'='*55}")
    print(f"  저장 완료!")
    print(f"  날짜        : {session.date}")
    print(f"  역량        : {session.competency}")
    print(f"  적합 직무   : {' > '.join(session.suitable_jobs)}")
    print(f"  의도 반영도 : {session.intent_score:.0f}/100")
    print(f"  JSON 저장   : {json_path}")
    print(f"  최종 TXT    : {final_txt_path}")
    print(f"{'='*55}\n")


def run_session(gen_model, gen_tokenizer, analysis: models.AnalysisModel | None):
    session    = Session()
    competency = random.choice(COMPETENCIES)

    question           = models.generate_question(gen_model, gen_tokenizer, competency)
    session.competency = competency
    session.question   = question
    _show_question(competency, question)

    answer, conversation_path = _get_answer_via_file(question, session.session_id, competency)
    session.answer = answer
    session.final_answer = answer

    if analysis:
        intent_score, job = analysis.predict(question, answer)
    else:
        intent_score, job = _fallback_analysis()

    session.intent_score = intent_score
    session.suitable_jobs = [job]
    session.add_job_history(0, "initial", intent_score, job)
    _show_analysis(intent_score, job)

    initial_check = check_consistency_with_model(gen_model, gen_tokenizer, session, answer)
    session.add_consistency_check(0, initial_check)
    _show_consistency(initial_check)

    current = answer
    last_consistency = initial_check

    for turn in range(1, MAX_TAIL_TURNS + 1):
        conversation_context = build_conversation_context(session, current)
        tail_q = models.generate_tail_question(
            gen_model,
            gen_tokenizer,
            question=question,
            answer=current,
            intent_score=intent_score,
            job=job,
            conversation_context=conversation_context,
            consistency_note=_consistency_note_for_prompt(last_consistency),
        )
        print(f"꼬리질문 [{turn}/{MAX_TAIL_TURNS}]: {tail_q}")
        reply = _get_tail_reply_via_file(tail_q, conversation_path, turn)
        if not reply:
            break

        session.add_turn(tail_q, reply)
        current = reply
        session.final_answer = current

        last_consistency = check_consistency_with_model(gen_model, gen_tokenizer, session, current)
        session.add_consistency_check(turn, last_consistency)
        _show_consistency(last_consistency)

        # 분석은 현재 답변 기준으로 수행한다. 꼬리질문 생성은 전체 conversation_context를 사용한다.
        if analysis:
            intent_score, job = analysis.predict(question, current)
        else:
            intent_score, job = _fallback_analysis()

        if job not in session.suitable_jobs:
            session.suitable_jobs.append(job)
        session.intent_score = intent_score
        session.add_job_history(turn, f"tail_{turn}", intent_score, job)
        _show_analysis(intent_score, job)

        if input("저장하고 종료? (y/n): ").strip().lower() == "y":
            break

    session.conversation_context = build_conversation_context(session, current)
    _show_job_trend(session)
    json_path = save_session(session.to_dict())
    final_txt_path = _save_final_text(session, conversation_path)
    _show_summary(session, json_path, final_txt_path)


def main():
    if "--history" in sys.argv:
        print_history()
        return

    print("\n모델 로딩 중... (CPU)")
    gen_model, gen_tokenizer = models.load_generation_model(USE_STAGE14_FINETUNED)
    analysis                 = models.load_analysis_model(USE_STAGE3_FINETUNED)

    run_session(gen_model, gen_tokenizer, analysis)


if __name__ == "__main__":
    main()
