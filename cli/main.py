"""
하루 1 문답 CLI

실행: python cli/main.py           # 새 세션 시작
      python cli/main.py --history  # 저장 이력 조회

플로우: 질문 생성 → 답변 입력(txt 파일) → 분석 → 꼬리질문 루프 → 저장
모델 교체: USE_STAGE14_FINETUNED / USE_STAGE3_FINETUNED 플래그 변경
"""

import os
import sys
import random
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import models
from session import Session
from storage import save_session, print_history

# ── 설정 ──────────────────────────────────────────────────────────────
USE_STAGE14_FINETUNED = False   # stage4 학습 완료 후 True
USE_STAGE3_FINETUNED  = False   # stage3 학습 완료 후 True

MAX_TAIL_TURNS = 5
MIN_ANSWER_LEN = 30
COMPETENCIES   = ["문제해결", "협업", "성장", "실패경험", "주도성"]
DRAFTS_DIR     = Path(__file__).parent.parent / "storage" / "drafts"


# ── DEV-2: 답변 입력 (txt 파일 방식) ─────────────────────────────────
def _open_editor(filepath: Path):
    """우선순위: $EDITOR 환경변수 → nano → vi → 수동 안내"""
    editor = os.environ.get("EDITOR", "")
    candidates = [c for c in [editor, "nano", "vi"] if c]

    for cmd in candidates:
        try:
            subprocess.run([cmd, str(filepath)], check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    # 모든 에디터 실패 시 수동 안내
    print(f"\n  에디터를 자동으로 열 수 없습니다.")
    print(f"  아래 파일을 직접 열어 답변을 작성하고 저장한 뒤 Enter를 눌러주세요.")
    print(f"  {filepath}\n")
    input("  [완료 후 Enter] ")


def _get_answer_via_file(question: str, session_id: str) -> str:
    """
    txt 파일을 생성하고 에디터로 열어 답변을 받음.
    에디터를 닫으면 파일을 읽어 답변을 반환.
    '#' 으로 시작하는 줄은 안내 주석으로 무시됨.
    """
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    draft_path = DRAFTS_DIR / f"{session_id}_answer.txt"

    template = (
        f"# ───────────────────────────────────────────────────────\n"
        f"# 하루 1 문답 - 답변 작성\n"
        f"# ───────────────────────────────────────────────────────\n"
        f"# 질문: {question}\n"
        f"#\n"
        f"# 안내:\n"
        f"#   - '#' 으로 시작하는 줄은 무시됩니다.\n"
        f"#   - 아래에 자유롭게 답변을 작성하세요.\n"
        f"#   - 최소 {MIN_ANSWER_LEN}자 이상 작성해야 합니다.\n"
        f"#   - 작성 완료 후 저장하고 닫으면 다음 단계로 넘어갑니다.\n"
        f"# ───────────────────────────────────────────────────────\n"
        f"\n"
    )
    draft_path.write_text(template, encoding="utf-8")

    print(f"\n  답변 파일이 열립니다. 작성 후 저장하고 닫아주세요.")
    print(f"  파일 위치: {draft_path}\n")

    _open_editor(draft_path)

    # 파일 읽기 — '#' 주석 줄 제거
    raw = draft_path.read_text(encoding="utf-8")
    lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
    answer = "\n".join(lines).strip()

    if not answer:
        print("\n  답변이 비어 있습니다. 파일을 다시 열어 작성해주세요.\n")
        return _get_answer_via_file(question, session_id)

    if len(answer) < MIN_ANSWER_LEN:
        print(f"\n  너무 짧습니다 (현재 {len(answer)}자, 최소 {MIN_ANSWER_LEN}자).")
        print("  파일을 다시 열어 보완해주세요.\n")
        return _get_answer_via_file(question, session_id)

    print(f"  답변 확인 ({len(answer)}자)\n")
    return answer


def _show_question(competency: str, question: str):
    print(f"\n{'='*55}")
    print(f"  오늘의 질문  [{competency}]")
    print(f"{'='*55}")
    print(f"  {question}")
    print(f"{'='*55}\n")


def _show_analysis(intent_score: float, job: str):
    bar = "█" * int(intent_score / 10) + "░" * (10 - int(intent_score / 10))
    print(f"\n  분석 결과")
    print(f"  의도 반영도 [{bar}] {intent_score:.0f}/100")
    print(f"  적합 직무   {job}\n")


def _show_summary(session: Session, path: str):
    print(f"\n{'='*55}")
    print(f"  저장 완료!")
    print(f"  날짜        : {session.date}")
    print(f"  역량        : {session.competency}")
    print(f"  적합 직무   : {' > '.join(session.suitable_jobs)}")
    print(f"  의도 반영도 : {session.intent_score:.0f}/100")
    print(f"  저장 위치   : {path}")
    print(f"{'='*55}\n")


def _fallback_analysis() -> tuple[float, str]:
    """stage3 미준비 시 대체값 반환"""
    print("  [분석 건너뜀] stage3 모델 학습 후 USE_STAGE3_FINETUNED = True 로 교체")
    return 50.0, random.choice(["backend", "ai_ml", "product"])


# ── 세션 플로우 ───────────────────────────────────────────────────────
def run_session(gen_model, gen_tokenizer, analysis: models.AnalysisModel | None):
    session    = Session()
    competency = random.choice(COMPETENCIES)

    # DEV-1: 질문 생성
    question           = models.generate_question(gen_model, gen_tokenizer, competency)
    session.competency = competency
    session.question   = question
    _show_question(competency, question)

    # DEV-2: 답변 입력 (txt 파일 → 에디터 → 저장 후 닫으면 진행)
    answer         = _get_answer_via_file(question, session.session_id)
    session.answer = answer

    # DEV-3: 초기 분석
    if analysis:
        intent_score, job = analysis.predict(question, answer)
    else:
        intent_score, job = _fallback_analysis()

    session.intent_score  = intent_score
    session.suitable_jobs = [job]
    _show_analysis(intent_score, job)

    # DEV-4: 꼬리질문 루프
    current = answer
    for turn in range(1, MAX_TAIL_TURNS + 1):
        tail_q = models.generate_tail_question(
            gen_model, gen_tokenizer,
            question, current, intent_score, job,
        )
        print(f"꼬리질문 [{turn}/{MAX_TAIL_TURNS}]: {tail_q}\n> ", end="")
        reply = input().strip()
        if not reply:
            break

        session.add_turn(tail_q, reply)
        current = reply

        if analysis:
            intent_score, job = analysis.predict(question, current)
        else:
            intent_score, job = _fallback_analysis()

        if job not in session.suitable_jobs:
            session.suitable_jobs.append(job)
        session.intent_score = intent_score
        _show_analysis(intent_score, job)

        if input("저장하고 종료? (y/n): ").strip().lower() == "y":
            break

    # DEV-5: 저장
    session.final_answer = current
    path = save_session(session.to_dict())
    _show_summary(session, path)


# ── 진입점 ────────────────────────────────────────────────────────────
def main():
    if "--history" in sys.argv:
        print_history()
        return

    print("\n모델 로딩 중...")
    gen_model, gen_tokenizer = models.load_generation_model(USE_STAGE14_FINETUNED)
    analysis                 = models.load_analysis_model(USE_STAGE3_FINETUNED)

    run_session(gen_model, gen_tokenizer, analysis)


if __name__ == "__main__":
    main()
