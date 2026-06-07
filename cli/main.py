# """
# 하루 1 문답 CLI - CPU 버전 + 대화 맥락/일관성 검사/직무 변화 추적
#
# 실행:
#     python cli/main.py
#     python cli/main.py --history
#
# 추가 기능:
# - 꼬리질문 생성 시 최초 답변 + 모든 꼬리질문/답변을 conversation_context로 사용
# - 꼬리질문 답변도 최초 답변과 동일하게 메모장/txt 파일로 입력
# - 생성 모델을 Judge처럼 사용해 conversation_context 기반 일관성 검사
# - 세션 종료 시 적합 직무와 intent_score 변화 추적 출력
# """
#
# import os
# import sys
# import random
# import subprocess
# from pathlib import Path
# from collections import Counter
#
# sys.path.insert(0, str(Path(__file__).parent))
# import models
# from session import Session
# from storage import save_session, print_history, STORAGE_DIR
#
# USE_STAGE14_FINETUNED = True
# USE_STAGE3_FINETUNED  = True
#
# MAX_TAIL_TURNS = 5
# MIN_ANSWER_LEN = 30
# COMPETENCIES   = ["문제해결", "협업", "성장", "실패경험", "주도성"]
# DRAFTS_DIR     = Path(__file__).parent.parent / "storage" / "drafts"
#
# # 단순 휴리스틱 기반 일관성 검사 키워드
# LEADER_WORDS = ["리더", "팀장", "주도", "총괄", "PM", "책임", "이끌"]
# LOW_AUTHORITY_WORDS = ["결정권", "권한", "시키는", "보조", "따라", "참여만", "맡은 게 없", "기여가 적"]
# SOLO_WORDS = ["혼자", "단독", "개인", "스스로"]
# TEAM_WORDS = ["팀", "협업", "함께", "동료", "조원"]
# VAGUE_WORDS = ["열심히", "노력", "잘", "많이", "최선을", "좋은 결과", "성공적"]
#
#
# def _open_editor(filepath: Path):
#     """Windows 우선: notepad → $EDITOR → nano → vi → 수동 안내"""
#     editor = os.environ.get("EDITOR", "")
#     candidates = ["notepad", editor, "nano", "vi"] if os.name == "nt" else [editor, "nano", "vi"]
#
#     for cmd in [c for c in candidates if c]:
#         try:
#             subprocess.run([cmd, str(filepath)], check=True)
#             return
#         except (FileNotFoundError, subprocess.CalledProcessError):
#             continue
#
#     print(f"\n  에디터를 자동으로 열 수 없습니다.")
#     print(f"  아래 파일을 직접 열어 답변을 작성하고 저장한 뒤 Enter를 눌러주세요.")
#     print(f"  {filepath}\n")
#     input("  [완료 후 Enter] ")
#
#
# def _conversation_file_path(session_id: str) -> Path:
#     """세션 전체 답변을 하나의 txt 파일에 누적 저장한다."""
#     DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
#     return DRAFTS_DIR / f"{session_id}_conversation.txt"
#
#
# def _extract_section_answer(raw: str, marker: str) -> str:
#     """marker 다음부터 다음 섹션 시작 전까지 답변만 추출한다."""
#     if marker not in raw:
#         return ""
#
#     text = raw.split(marker, 1)[1]
#
#     # 다음 질문/답변 섹션이 생긴 경우 그 전까지만 읽는다.
#     next_headers = [
#         "\n[꼬리질문 ",
#         "\n[답변 ",
#         "\n[최종",
#         "\n---",
#     ]
#     cut_positions = [text.find(h) for h in next_headers if text.find(h) != -1]
#     if cut_positions:
#         text = text[:min(cut_positions)]
#
#     # 안내 주석은 무시한다.
#     lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
#     return "\n".join(lines).strip()
#
#
# def _open_and_read_section(
#     *,
#     path: Path,
#     marker: str,
#     min_len: int,
#     empty_message: str = "입력이 비어 있습니다.",
# ) -> str:
#     """같은 conversation txt 파일을 열고, marker 아래 답변만 읽는다."""
#     print(f"\n  입력 파일이 열립니다. 작성 후 저장하고 닫아주세요.")
#     print(f"  파일 위치: {path}\n")
#
#     _open_editor(path)
#
#     raw = path.read_text(encoding="utf-8")
#     answer = _extract_section_answer(raw, marker)
#
#     if not answer:
#         print(f"\n  {empty_message} 파일을 다시 열어 작성해주세요.\n")
#         return _open_and_read_section(
#             path=path,
#             marker=marker,
#             min_len=min_len,
#             empty_message=empty_message,
#         )
#
#     if len(answer) < min_len:
#         print(f"\n  너무 짧습니다 (현재 {len(answer)}자, 최소 {min_len}자).")
#         print("  같은 파일을 다시 열어 보완해주세요.\n")
#         return _open_and_read_section(
#             path=path,
#             marker=marker,
#             min_len=min_len,
#             empty_message=empty_message,
#         )
#
#     print(f"  입력 확인 ({len(answer)}자)\n")
#     return answer
#
#
# def _init_conversation_file(question: str, session_id: str, competency: str) -> Path:
#     """최초 질문/답변용 conversation txt 파일을 1개만 만든다."""
#     path = _conversation_file_path(session_id)
#     template = (
#         f"# ───────────────────────────────────────────────────────\n"
#         f"# 하루 1 문답 - 전체 대화 작성 파일\n"
#         f"# ───────────────────────────────────────────────────────\n"
#         f"# 안내:\n"
#         f"#   - 이 파일 하나에 최초 답변과 모든 꼬리질문 답변을 이어서 작성합니다.\n"
#         f"#   - 꼬리질문이 생기면 이 파일 맨 아래에 자동으로 추가됩니다.\n"
#         f"#   - 기존 내용은 지우지 말고, 새 [답변 n] 아래에만 작성하세요.\n"
#         f"#   - '#' 으로 시작하는 줄은 답변에서 제외됩니다.\n"
#         f"# ───────────────────────────────────────────────────────\n\n"
#         f"[역량]\n{competency}\n\n"
#         f"[원본 질문]\n{question}\n\n"
#         f"[최초 답변]\n"
#     )
#     path.write_text(template, encoding="utf-8")
#     return path
#
#
# def _get_answer_via_file(question: str, session_id: str, competency: str) -> tuple[str, Path]:
#     """최초 답변도 conversation txt 하나에서 받는다."""
#     path = _init_conversation_file(question, session_id, competency)
#     answer = _open_and_read_section(
#         path=path,
#         marker="[최초 답변]",
#         min_len=MIN_ANSWER_LEN,
#         empty_message="최초 답변이 비어 있습니다.",
#     )
#     return answer, path
#
#
# def _append_tail_question(path: Path, tail_question: str, turn: int) -> str:
#     """기존 conversation txt 맨 아래에 꼬리질문과 답변 칸을 추가한다."""
#     marker = f"[답변 {turn}]"
#     current = path.read_text(encoding="utf-8")
#
#     # 같은 turn을 실수로 중복 append하지 않도록 방지
#     if f"[꼬리질문 {turn}]" not in current:
#         with path.open("a", encoding="utf-8") as f:
#             f.write(
#                 f"\n\n----------------------------------------\n"
#                 f"[꼬리질문 {turn}]\n"
#                 f"{tail_question}\n\n"
#                 f"{marker}\n"
#             )
#     return marker
#
#
# def _get_tail_reply_via_file(tail_question: str, conversation_path: Path, turn: int) -> str:
#     """꼬리질문 답변도 같은 conversation txt 파일에서 이어서 받는다."""
#     marker = _append_tail_question(conversation_path, tail_question, turn)
#     return _open_and_read_section(
#         path=conversation_path,
#         marker=marker,
#         min_len=10,
#         empty_message=f"꼬리질문 {turn} 답변이 비어 있습니다.",
#     )
#
#
# def _save_final_text(session: Session, conversation_path: Path) -> str:
#     """JSON과 별도로 사람이 읽기 좋은 최종 txt 파일도 하나로 저장한다."""
#     os.makedirs(STORAGE_DIR, exist_ok=True)
#     final_path = Path(STORAGE_DIR) / f"{session.session_id}_final.txt"
#     conversation_text = conversation_path.read_text(encoding="utf-8") if conversation_path.exists() else session.conversation_context
#
#     job_history_lines = []
#     for h in session.job_history:
#         job_history_lines.append(
#             f"Turn {h['turn']} | {h['source']} | {h['job']} | intent {h['intent_score']}/100"
#         )
#
#     final_text = (
#         f"{conversation_text.rstrip()}\n\n"
#         f"========================================\n"
#         f"[최종 요약]\n"
#         f"날짜: {session.date}\n"
#         f"역량: {session.competency}\n"
#         f"최종 적합 직무: {' > '.join(session.suitable_jobs)}\n"
#         f"최종 의도 반영도: {session.intent_score:.1f}/100\n\n"
#         f"[적합 직무 변화 추적]\n"
#         f"{chr(10).join(job_history_lines) if job_history_lines else '기록 없음'}\n"
#     )
#     final_path.write_text(final_text, encoding="utf-8")
#     return str(final_path)
#
# def _show_question(competency: str, question: str):
#     print(f"\n{'='*55}")
#     print(f"  오늘의 질문  [{competency}]")
#     print(f"{'='*55}")
#     print(f"  {question}")
#     print(f"{'='*55}\n")
#
#
# def _show_analysis(intent_score: float, job: str):
#     score = max(0, min(100, int(intent_score)))
#     bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
#     print(f"\n  분석 결과")
#     print(f"  의도 반영도 [{bar}] {score}/100")
#     print(f"  적합 직무   {job}\n")
#
#
# def _fallback_analysis() -> tuple[float, str]:
#     print("  [분석 대체값] stage3 CPU 모델 파일이 없어 임시값 사용")
#     return 50.0, random.choice(["backend", "ai_ml", "product"])
#
#
# def _contains_any(text: str, words: list[str]) -> bool:
#     return any(w.lower() in text.lower() for w in words)
#
#
# def build_conversation_context(session: Session, current_answer: str | None = None) -> str:
#     """최초 답변과 모든 꼬리질문/답변을 하나의 맥락 문자열로 합침."""
#     lines = [
#         f"[원본 질문] {session.question}",
#         f"[최초 답변] {session.answer}",
#     ]
#     for i, turn in enumerate(session.conversation, start=1):
#         lines.append(f"[꼬리질문 {i}] {turn['tail_question']}")
#         lines.append(f"[답변 {i}] {turn['response']}")
#     if current_answer and (not session.conversation or session.conversation[-1].get("response") != current_answer):
#         lines.append(f"[현재 답변] {current_answer}")
#     return "\n".join(lines)
#
#
# def check_consistency_with_model(
#     gen_model,
#     gen_tokenizer,
#     session: Session,
#     latest_answer: str,
# ) -> dict:
#     """대화 전체를 생성 모델에 넣어 일관성 검사 결과를 받는다."""
#     conversation_context = build_conversation_context(session, latest_answer)
#     try:
#         return models.assess_consistency(
#             gen_model,
#             gen_tokenizer,
#             conversation_context=conversation_context,
#             latest_answer=latest_answer,
#         )
#     except Exception as e:
#         return {
#             "status": "주의",
#             "issues": [f"일관성 검사 실행 중 오류가 발생했습니다: {e}"],
#             "suggestions": ["저장된 대화 맥락을 수동으로 확인해 주세요."],
#             "method": "llm_judge_error",
#         }
#
#
# def _show_consistency(result: dict):
#     print("\n  일관성 검사")
#     if result["status"] == "양호":
#         print("  ✓ 현재까지 큰 모순은 감지되지 않았습니다.\n")
#         return
#     print("  ⚠ 확인 필요")
#     for issue in result["issues"]:
#         print(f"  - {issue}")
#     if result["suggestions"]:
#         print("  보완 방향")
#         for sug in result["suggestions"]:
#             print(f"  → {sug}")
#     print()
#
#
# def _consistency_note_for_prompt(result: dict) -> str:
#     if result["status"] == "양호":
#         return "현재까지 큰 모순은 없지만, 이전 답변을 반복하지 말고 더 깊은 정보를 물어볼 것."
#     parts = []
#     parts.extend(result.get("issues", []))
#     parts.extend(result.get("suggestions", []))
#     return " / ".join(parts)
#
#
# def _show_job_trend(session: Session):
#     history = session.job_history
#     print(f"\n{'='*55}")
#     print("  적합 직무 변화 추적")
#     print(f"{'='*55}")
#
#     if not history:
#         print("  기록된 직무 예측이 없습니다.")
#         print(f"{'='*55}\n")
#         return
#
#     for h in history:
#         print(f"  Turn {h['turn']:>2} | {h['source']:<10} | {h['job']:<8} | intent {h['intent_score']:>5.1f}/100")
#
#     first = history[0]
#     last = history[-1]
#     delta = last["intent_score"] - first["intent_score"]
#     sign = "+" if delta >= 0 else ""
#     counter = Counter(h["job"] for h in history)
#     top_jobs = [job for job, _ in counter.most_common(3)]
#
#     print("-" * 55)
#     print(f"  시작 직무 → 최종 직무 : {first['job']} → {last['job']}")
#     print(f"  의도 반영도 변화       : {first['intent_score']:.1f} → {last['intent_score']:.1f} ({sign}{delta:.1f})")
#     print(f"  세션 내 TOP 직무       : {' > '.join(top_jobs)}")
#
#     if first["job"] != last["job"]:
#         print("  해석                  : 답변 보완 과정에서 강조되는 직무 방향이 바뀌었습니다.")
#     elif abs(delta) >= 10:
#         print("  해석                  : 직무 방향은 유지됐고, 의도 반영도 변화가 크게 나타났습니다.")
#     else:
#         print("  해석                  : 직무 방향과 점수가 비교적 안정적으로 유지됐습니다.")
#     print(f"{'='*55}\n")
#
#
# def _show_summary(session: Session, json_path: str, final_txt_path: str):
#     print(f"\n{'='*55}")
#     print(f"  저장 완료!")
#     print(f"  날짜        : {session.date}")
#     print(f"  역량        : {session.competency}")
#     print(f"  적합 직무   : {' > '.join(session.suitable_jobs)}")
#     print(f"  의도 반영도 : {session.intent_score:.0f}/100")
#     print(f"  JSON 저장   : {json_path}")
#     print(f"  최종 TXT    : {final_txt_path}")
#     print(f"{'='*55}\n")
#
#
# def run_session(gen_model, gen_tokenizer, analysis: models.AnalysisModel | None):
#     session    = Session()
#     competency = random.choice(COMPETENCIES)
#
#     question           = models.generate_question(gen_model, gen_tokenizer, competency)
#     session.competency = competency
#     session.question   = question
#     _show_question(competency, question)
#
#     answer, conversation_path = _get_answer_via_file(question, session.session_id, competency)
#     session.answer = answer
#     session.final_answer = answer
#
#     if analysis:
#         intent_score, job = analysis.predict(question, answer)
#     else:
#         intent_score, job = _fallback_analysis()
#
#     session.intent_score = intent_score
#     session.suitable_jobs = [job]
#     session.add_job_history(0, "initial", intent_score, job)
#     _show_analysis(intent_score, job)
#
#     initial_check = check_consistency_with_model(gen_model, gen_tokenizer, session, answer)
#     session.add_consistency_check(0, initial_check)
#     _show_consistency(initial_check)
#
#     current = answer
#     last_consistency = initial_check
#
#     for turn in range(1, MAX_TAIL_TURNS + 1):
#         conversation_context = build_conversation_context(session, current)
#         tail_q = models.generate_tail_question(
#             gen_model,
#             gen_tokenizer,
#             question=question,
#             answer=current,
#             intent_score=intent_score,
#             job=job,
#             conversation_context=conversation_context,
#             consistency_note=_consistency_note_for_prompt(last_consistency),
#         )
#         print(f"꼬리질문 [{turn}/{MAX_TAIL_TURNS}]: {tail_q}")
#         reply = _get_tail_reply_via_file(tail_q, conversation_path, turn)
#         if not reply:
#             break
#
#         session.add_turn(tail_q, reply)
#         current = reply
#         session.final_answer = current
#
#         last_consistency = check_consistency_with_model(gen_model, gen_tokenizer, session, current)
#         session.add_consistency_check(turn, last_consistency)
#         _show_consistency(last_consistency)
#
#         # 분석은 현재 답변 기준으로 수행한다. 꼬리질문 생성은 전체 conversation_context를 사용한다.
#         if analysis:
#             intent_score, job = analysis.predict(question, current)
#         else:
#             intent_score, job = _fallback_analysis()
#
#         if job not in session.suitable_jobs:
#             session.suitable_jobs.append(job)
#         session.intent_score = intent_score
#         session.add_job_history(turn, f"tail_{turn}", intent_score, job)
#         _show_analysis(intent_score, job)
#
#         if input("저장하고 종료? (y/n): ").strip().lower() == "y":
#             break
#
#     session.conversation_context = build_conversation_context(session, current)
#     _show_job_trend(session)
#     json_path = save_session(session.to_dict())
#     final_txt_path = _save_final_text(session, conversation_path)
#     _show_summary(session, json_path, final_txt_path)
#
#
# def main():
#     if "--history" in sys.argv:
#         print_history()
#         return
#
#     print("\n모델 로딩 중... (CPU)")
#     gen_model, gen_tokenizer = models.load_generation_model(USE_STAGE14_FINETUNED)
#     analysis = models.load_analysis_model(USE_STAGE3_FINETUNED)
#
#     run_session(gen_model, gen_tokenizer, analysis)
#
#
# if __name__ == "__main__":
#     main()


"""
하루 1 문답 CLI - CPU 버전 + 대화 맥락/일관성 검사/직무 변화 추적

실행:
    python cli/main.py
    python cli/main.py --history

추가 기능:
- 꼬리질문 생성 시 최초 답변 + 모든 꼬리질문/답변을 conversation_context로 사용
- 꼬리질문 답변도 최초 답변과 동일하게 메모장/txt 파일로 입력
- 이전 답변 각각 + 전체 누적 맥락을 현재 답변과 비교해 규칙 기반 + NLI + SBERT 일관성 검사
- 세션 종료 시 적합 직무와 intent_score 변화 추적 출력
"""

import os
import sys
import random
import subprocess
from pathlib import Path
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

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

# SBERT는 맥락 유사도 검사에 사용한다. 없으면 해당 검사만 건너뛴다.
SBERT_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
SBERT_MODEL = None
CONTEXT_LOW_SIM_THRESHOLD = 0.35
CONTEXT_RELATED_THRESHOLD = 0.55

# NLI는 실제 contradiction 보조 판정에 사용한다.
# 한국어 포함 XNLI를 지원하는 multilingual 모델을 기본값으로 둔다.
# CPU가 느리면 환경변수 CONSISTENCY_USE_NLI=0 으로 끌 수 있다.
NLI_MODEL_NAME = os.environ.get("CONSISTENCY_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
USE_NLI = os.environ.get("CONSISTENCY_USE_NLI", "1") != "0"
NLI_TOKENIZER = None
NLI_MODEL = None
NLI_DEVICE = "cpu"
NLI_CONTRADICTION_THRESHOLD = 0.65
NLI_STRONG_CONTRADICTION_THRESHOLD = 0.80



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


def _get_previous_answers(session: Session, latest_answer: str | None = None) -> list[dict]:
    """현재 답변을 제외한 이전 답변들을 비교 가능한 단위로 반환한다."""
    answers = []
    latest_text = latest_answer.strip() if latest_answer else ""

    # session.add_turn() 직후에는 현재 답변이 conversation 마지막에 들어가 있다.
    latest_is_tail = bool(
        session.conversation
        and latest_text
        and str(session.conversation[-1].get("response", "")).strip() == latest_text
    )

    initial_answer = session.answer.strip() if session.answer else ""
    # 최초 답변 검사 시에는 최초 답변이 곧 현재 답변이므로 비교 대상에서 제외한다.
    # 꼬리답변 검사 시에는 최초 답변이 중요한 이전 맥락이므로 포함한다.
    if initial_answer and (latest_is_tail or initial_answer != latest_text):
        answers.append({
            "source": "최초 답변",
            "text": initial_answer,
        })

    turns_to_compare = session.conversation[:-1] if latest_is_tail else session.conversation
    for idx, turn in enumerate(turns_to_compare, start=1):
        response = str(turn.get("response", "")).strip()
        if not response:
            continue

        answers.append({
            "source": f"꼬리답변 {idx}",
            "text": response,
        })

    return answers


def _get_sbert_model():
    """SBERT 모델을 한 번만 로딩한다. 설치되어 있지 않으면 None을 반환한다."""
    global SBERT_MODEL

    if SentenceTransformer is None or util is None:
        return None

    if SBERT_MODEL is None:
        try:
            SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
        except Exception as e:
            print(f"  [일관성 검사] SBERT 로딩 실패 → 유사도 검사는 건너뜀: {e}")
            SBERT_MODEL = False

    return None if SBERT_MODEL is False else SBERT_MODEL


def _calc_similarity(text1: str, text2: str) -> float | None:
    """SBERT cosine similarity. SBERT 사용 불가 시 None."""
    model = _get_sbert_model()
    if model is None:
        return None

    emb = model.encode([text1, text2], convert_to_tensor=True)
    return round(float(util.cos_sim(emb[0], emb[1]).item()), 4)


def _get_nli_components():
    """NLI tokenizer/model을 한 번만 로딩한다. 사용 불가 시 None을 반환한다."""
    global NLI_TOKENIZER, NLI_MODEL

    if not USE_NLI:
        return None

    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None

    if NLI_TOKENIZER is None or NLI_MODEL is None:
        try:
            NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
            NLI_MODEL.to(NLI_DEVICE)
            NLI_MODEL.eval()
        except Exception as e:
            print(f"  [일관성 검사] NLI 로딩 실패 → contradiction 검사는 건너뜀: {e}")
            NLI_TOKENIZER = False
            NLI_MODEL = False

    if NLI_TOKENIZER is False or NLI_MODEL is False:
        return None
    return NLI_TOKENIZER, NLI_MODEL


def _resolve_nli_label_map(model) -> dict:
    """모델 config의 id2label을 contradiction/entailment/neutral로 정규화한다."""
    id2label = getattr(model.config, "id2label", {}) or {}
    normalized = {}
    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if "contradiction" in label_lower or "contradict" in label_lower:
            normalized["contradiction"] = int(idx)
        elif "entailment" in label_lower or "entail" in label_lower:
            normalized["entailment"] = int(idx)
        elif "neutral" in label_lower:
            normalized["neutral"] = int(idx)

    # 일부 MNLI 계열 모델은 보통 0=contradiction, 1=neutral, 2=entailment를 쓴다.
    if "contradiction" not in normalized and getattr(model.config, "num_labels", 0) == 3:
        normalized["contradiction"] = 0
    if "neutral" not in normalized and getattr(model.config, "num_labels", 0) == 3:
        normalized["neutral"] = 1
    if "entailment" not in normalized and getattr(model.config, "num_labels", 0) == 3:
        normalized["entailment"] = 2

    return normalized


def _calc_nli_scores(premise: str, hypothesis: str) -> dict | None:
    """premise → hypothesis 관계의 NLI 확률을 계산한다. 사용 불가 시 None."""
    components = _get_nli_components()
    if components is None:
        return None

    tokenizer, model = components
    try:
        enc = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(NLI_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        label_map = _resolve_nli_label_map(model)
        return {
            "contradiction": round(float(probs[label_map.get("contradiction", 0)]), 4),
            "neutral": round(float(probs[label_map.get("neutral", 1)]), 4) if len(probs) > 1 else None,
            "entailment": round(float(probs[label_map.get("entailment", 2)]), 4) if len(probs) > 2 else None,
        }
    except Exception as e:
        print(f"  [일관성 검사] NLI 추론 실패 → 해당 pair는 건너뜀: {e}")
        return None


def _calc_bidirectional_nli(prev_text: str, latest_text: str) -> dict | None:
    """NLI는 방향성이 있으므로 이전→현재, 현재→이전을 모두 계산하고 contradiction 최댓값을 사용한다."""
    forward = _calc_nli_scores(prev_text, latest_text)
    backward = _calc_nli_scores(latest_text, prev_text)

    if forward is None and backward is None:
        return None

    forward_contra = forward.get("contradiction", 0.0) if forward else 0.0
    backward_contra = backward.get("contradiction", 0.0) if backward else 0.0
    max_contra = max(forward_contra, backward_contra)
    direction = "previous_to_current" if forward_contra >= backward_contra else "current_to_previous"

    return {
        "forward": forward,
        "backward": backward,
        "contradiction": round(float(max_contra), 4),
        "direction": direction,
    }


def _check_rule_conflicts(previous_text: str, latest_text: str, source: str = "이전 답변") -> tuple[list[str], list[str]]:
    """자소서 답변에서 자주 발생하는 역할/권한/협업 충돌을 검사한다."""
    issues = []
    suggestions = []

    if _contains_any(previous_text, LEADER_WORDS) and _contains_any(latest_text, LOW_AUTHORITY_WORDS):
        issues.append(f"{source}에서는 주도적 역할을 강조했지만, 현재 답변에서는 권한이나 기여도가 낮았다고 표현했습니다.")
        suggestions.append("실제로 맡은 역할, 결정권 범위, 본인 기여도를 구분해서 설명해 주세요.")

    if _contains_any(previous_text, LOW_AUTHORITY_WORDS) and _contains_any(latest_text, LEADER_WORDS):
        issues.append(f"{source}에서는 보조적 역할에 가까웠지만, 현재 답변에서는 주도적 역할로 표현이 바뀌었습니다.")
        suggestions.append("처음부터 주도한 것인지, 특정 단계에서만 주도했는지 구체적으로 정리해 주세요.")

    if _contains_any(previous_text, SOLO_WORDS) and _contains_any(latest_text, TEAM_WORDS):
        issues.append(f"{source}에서는 혼자 수행한 경험처럼 보였지만, 현재 답변에서는 팀 활동으로 표현되었습니다.")
        suggestions.append("개인 작업과 팀 협업의 범위를 나눠서 설명해 주세요.")

    if _contains_any(previous_text, TEAM_WORDS) and _contains_any(latest_text, SOLO_WORDS):
        issues.append(f"{source}에서는 팀/협업 경험으로 보였지만, 현재 답변에서는 혼자 수행한 것처럼 표현되었습니다.")
        suggestions.append("팀 전체 역할과 본인이 단독으로 맡은 부분을 구분해 주세요.")

    return issues, suggestions


def check_consistency_pairwise(session: Session, latest_answer: str) -> dict:
    """
    이전 답변 각각과 현재 답변을 pairwise로 비교하고,
    전체 이전 답변을 합친 누적 맥락과도 비교한다.
    """
    latest_text = latest_answer.strip()
    previous_answers = _get_previous_answers(session, latest_answer=latest_text)

    issues = []
    suggestions = []
    pair_results = []

    if len(latest_text) < 30:
        issues.append("현재 답변이 너무 짧아 이전 답변과의 일관성을 충분히 판단하기 어렵습니다.")
        suggestions.append("본인의 역할, 행동, 결과를 최소 2~3문장으로 작성해 주세요.")

    vague_count = sum(1 for word in VAGUE_WORDS if word in latest_text)
    if vague_count >= 2 and len(latest_text) < 120:
        issues.append("현재 답변에 추상적인 표현이 많고 구체적인 행동이나 결과가 부족합니다.")
        suggestions.append("상황, 본인의 행동, 결과 수치나 변화가 드러나도록 보완해 주세요.")

    # 최초 답변만 있는 시점에는 비교 대상이 없으므로 현재 답변 자체의 품질만 검사한다.
    if not previous_answers:
        return {
            "status": "주의" if issues else "양호",
            "issues": list(dict.fromkeys(issues)),
            "suggestions": list(dict.fromkeys(suggestions)),
            "method": "pairwise_rule_nli_sbert",
            "pair_results": [],
            "global_result": None,
        }

    for prev in previous_answers:
        source = prev["source"]
        prev_text = prev["text"]
        sim = _calc_similarity(prev_text, latest_text)
        nli = _calc_bidirectional_nli(prev_text, latest_text)
        pair_issues, pair_suggestions = _check_rule_conflicts(prev_text, latest_text, source)

        if nli is not None and nli["contradiction"] >= NLI_CONTRADICTION_THRESHOLD:
            if nli["contradiction"] >= NLI_STRONG_CONTRADICTION_THRESHOLD:
                pair_issues.append(f"{source}과 현재 답변 사이의 NLI contradiction 점수가 높아 모순 가능성이 큽니다. (score={nli['contradiction']})")
            else:
                pair_issues.append(f"{source}과 현재 답변 사이에 NLI 기준 모순 가능성이 있습니다. (score={nli['contradiction']})")
            pair_suggestions.append(f"{source}에서 말한 내용과 현재 답변의 역할·권한·기여도 표현이 함께 성립하는지 확인해 주세요.")

        if sim is not None and sim < CONTEXT_LOW_SIM_THRESHOLD:
            pair_issues.append(f"{source}과 현재 답변의 의미 유사도가 낮아 같은 경험을 이어서 설명하는지 확인이 필요합니다.")
            pair_suggestions.append("현재 답변이 기존 경험의 같은 사례를 보완하는 내용인지 확인해 주세요.")

        if sim is not None and sim >= CONTEXT_RELATED_THRESHOLD and pair_issues:
            pair_suggestions.append("같은 경험을 말하고 있을 가능성이 높으므로, 역할·권한·기여도 표현이 충돌하지 않게 정리해 주세요.")

        pair_results.append({
            "source": source,
            "similarity": sim,
            "nli": nli,
            "issues": pair_issues,
        })
        issues.extend(pair_issues)
        suggestions.extend(pair_suggestions)

    # 전체 이전 답변을 합친 누적 맥락과도 비교한다. 단, 이 결과는 보조 신호로만 사용한다.
    global_text = "\n".join(prev["text"] for prev in previous_answers)
    global_similarity = _calc_similarity(global_text, latest_text)
    global_nli = _calc_bidirectional_nli(global_text, latest_text)
    global_issues, global_suggestions = _check_rule_conflicts(global_text, latest_text, "전체 이전 답변")

    if global_nli is not None and global_nli["contradiction"] >= NLI_CONTRADICTION_THRESHOLD:
        global_issues.append(f"전체 이전 답변과 현재 답변 사이의 NLI contradiction 점수가 높아 누적 맥락과 충돌할 가능성이 있습니다. (score={global_nli['contradiction']})")
        global_suggestions.append("전체 답변 흐름에서 역할, 권한, 기여도, 시점을 한 번에 정리해 주세요.")

    if global_similarity is not None and global_similarity < CONTEXT_LOW_SIM_THRESHOLD:
        global_issues.append("전체 이전 답변 흐름과 현재 답변의 의미 유사도가 낮아 맥락 이탈 가능성이 있습니다.")
        global_suggestions.append("현재 답변이 기존 답변의 같은 경험을 이어서 설명하는지 확인해 주세요.")

    issues.extend(global_issues)
    suggestions.extend(global_suggestions)

    return {
        "status": "주의" if issues else "양호",
        "issues": list(dict.fromkeys(issues)),
        "suggestions": list(dict.fromkeys(suggestions)),
        "method": "pairwise_rule_nli_sbert",
        "pair_results": pair_results,
        "global_result": {
            "source": "전체 이전 답변",
            "similarity": global_similarity,
            "nli": global_nli,
            "issues": global_issues,
        },
    }


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
    """
    기존 호출 구조를 유지하기 위한 래퍼 함수.
    실제 일관성 검사는 생성 모델 Judge가 아니라 pairwise 규칙/NLI/SBERT 기반으로 수행한다.
    """
    return check_consistency_pairwise(session, latest_answer)


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
    analysis = models.load_analysis_model(USE_STAGE3_FINETUNED)

    run_session(gen_model, gen_tokenizer, analysis)


if __name__ == "__main__":
    main()
