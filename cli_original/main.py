"""
하루 1 문답 CLI - CPU 버전

실행:
    python cli/main.py
    python cli/main.py --history

CPU용 변경:
- 생성 모델: stage4_tail_question/models/best_cpu 우선 사용
- 분석 모델: stage3_multitask/models/best_cpu 우선 사용
- CUDA/Unsloth 없이 HuggingFace Transformers + PyTorch CPU로 실행
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
# CPU 학습 모델을 우선 사용. 파일이 없으면 models.py에서 base/fallback으로 자동 처리.
USE_STAGE14_FINETUNED = True
USE_STAGE3_FINETUNED  = True

MAX_TAIL_TURNS = 5
MIN_ANSWER_LEN = 30
COMPETENCIES   = ["문제해결", "협업", "성장", "실패경험", "주도성"]
DRAFTS_DIR     = Path(__file__).parent.parent / "storage" / "drafts"


# ── DEV-2: 답변 입력 (txt 파일 방식) ─────────────────────────────────
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


def _get_answer_via_file(question: str, session_id: str) -> str:
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
        f"# ───────────────────────────────────────────────────────\n\n"
    )
    draft_path.write_text(template, encoding="utf-8")

    print(f"\n  답변 파일이 열립니다. 작성 후 저장하고 닫아주세요.")
    print(f"  파일 위치: {draft_path}\n")

    _open_editor(draft_path)

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
    score = max(0, min(100, int(intent_score)))
    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
    print(f"\n  분석 결과")
    print(f"  의도 반영도 [{bar}] {score}/100")
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
    print("  [분석 대체값] stage3 CPU 모델 파일이 없어 임시값 사용")
    return 50.0, random.choice(["backend", "ai_ml", "product"])


# ── 세션 플로우 ───────────────────────────────────────────────────────
def run_session(gen_model, gen_tokenizer, analysis: models.AnalysisModel | None):
    session    = Session()
    competency = random.choice(COMPETENCIES)

    question           = models.generate_question(gen_model, gen_tokenizer, competency)
    session.competency = competency
    session.question   = question
    _show_question(competency, question)

    answer         = _get_answer_via_file(question, session.session_id)
    session.answer = answer

    if analysis:
        intent_score, job = analysis.predict(question, answer)
    else:
        intent_score, job = _fallback_analysis()

    session.intent_score  = intent_score
    session.suitable_jobs = [job]
    _show_analysis(intent_score, job)

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

    session.final_answer = current
    path = save_session(session.to_dict())
    _show_summary(session, path)


# ── 진입점 ────────────────────────────────────────────────────────────
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
