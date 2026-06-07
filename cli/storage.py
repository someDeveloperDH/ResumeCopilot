"""로컬 JSON 세션 저장소"""

import os
import json
from collections import Counter

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "../storage/sessions")


def save_session(session_dict: dict) -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    path = os.path.join(STORAGE_DIR, f"{session_dict['session_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_dict, f, ensure_ascii=False, indent=2)
    return path


def load_all() -> list[dict]:
    if not os.path.exists(STORAGE_DIR):
        return []
    sessions = []
    for fname in sorted(os.listdir(STORAGE_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(STORAGE_DIR, fname), encoding="utf-8") as f:
                sessions.append(json.load(f))
    return sessions


def get_top_jobs(n: int = 3) -> list[str]:
    """전체 세션 기반 가장 많이 등장한 적합 직무 TOP N"""
    counter = Counter()
    for s in load_all():
        for job in s.get("suitable_jobs", []):
            counter[job] += 1
    return [job for job, _ in counter.most_common(n)]


def print_history():
    """저장된 세션 요약 출력"""
    sessions = load_all()
    if not sessions:
        print("저장된 세션이 없습니다.")
        return
    print(f"\n{'='*55}")
    print(f"  저장된 세션 {len(sessions)}개")
    print(f"{'='*55}")
    for s in sessions:
        jobs = " > ".join(s.get("suitable_jobs", []))
        card = s.get("answer_card", {})
        title = card.get("core_experience") or s.get("question", "")[:30]
        score = card.get("after_score", s.get("intent_score", "-"))
        input_type = s.get("input_type") or card.get("input_type", "-")
        print(f"  [{s['date']}] [{s['competency']}] {title}...")
        print(f"           직무: {jobs or '-'}  score: {score}  type: {input_type}")
    top = get_top_jobs()
    if top:
        print(f"\n  전체 적합 직무 TOP3: {' > '.join(top)}")
    print(f"{'='*55}\n")
