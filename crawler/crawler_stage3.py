"""
마커리어 Stage 3 분석 데이터 수집 크롤러
— 질문 + 답변 쌍, suitable_job 라벨링

실행 (test4 루트에서):
    python crawler/crawler_stage3.py --output-dir ./data/stage3 --label-method keyword
    python crawler/crawler_stage3.py --output-dir ./data/stage3 --label-method llm
    python crawler/crawler_stage3.py --start-id 34000 --output-dir ./data/stage3
"""

import argparse
import csv
import os
import random
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────
ID_START       = 30000
ID_END         = 37000
DELAY_MIN      = 1.5
DELAY_MAX      = 3.5
MAX_RETRY      = 3
RETRY_WAIT     = 60
BASE_URL       = "https://linkareer.com/cover-letter"
USER_AGENT     = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# 클래스별 목표 수
CLASS_TRAIN = {"backend": 100, "ai_ml": 100, "product": 100}
CLASS_TEST  = {"backend": 25,  "ai_ml": 25,  "product": 25}
JOBS        = list(CLASS_TRAIN.keys())

TRAIN_FILE   = "stage3_train.csv"
TEST_FILE    = "stage3_test.csv"
VISITED_FILE = "visited_urls.txt"
CSV_COLUMNS  = ["question", "answer", "intent_score", "suitable_job", "url"]

ANSWER_MAX_LEN  = 2000   # 이 이상이면 앞 1500자만 저장
ANSWER_SAVE_LEN = 1500
ANSWER_MIN_LEN  = 50     # 이 미만이면 답변 없는 것으로 간주

# IT 직무 키워드
IT_KEYWORDS = [
    "IT", "개발", "소프트웨어", "프로그래머", "엔지니어", "데이터",
    "AI", "인공지능", "백엔드", "프론트", "풀스택", "DevOps",
    "클라우드", "보안", "시스템", "네트워크", "QA", "데이터엔지니어",

]

# 직무 분류 키워드
JOB_KEYWORDS = {
    "backend": [
        "서버", "API", "데이터베이스", "DB", "SQL", "인프라", "클라우드",
        "AWS", "Spring", "Django", "백엔드", "보안", "시스템", "쿼리",
        "Docker", "Kubernetes", "네트워크", "성능", "최적화", "배포",
        "MSA", "마이크로서비스", "Redis", "NoSQL", "Linux", "CI/CD",
    ],
    "ai_ml": [
        "머신러닝", "딥러닝", "모델", "데이터 분석", "AI", "인공지능",
        "LLM", "통계", "분석", "예측", "군집", "분류", "학습", "정확도",
        "회귀", "데이터셋", "feature", "파이프라인", "PyTorch", "TensorFlow",
        "자연어", "NLP", "컴퓨터비전", "강화학습", "엔지니어", "ML",
        "NLP",
    ],
    "product": [
        "기획", "팀원", "소통", "커뮤니케이션", "사용자", "고객", "팀",
        "서비스", "전략", "PM", "리더", "조율", "프로젝트 관리", "노션",
        "협업", "일정", "로드맵", "요구사항", "이해관계자", "백로그",
        "스프린트", "애자일", "스크럼","PM",
    ],
}


# ── 직무 분류 (키워드 방식) ──────────────────────────────────────────
def classify_by_keyword(text: str) -> str:
    """답변 텍스트에서 클래스 키워드 빈도를 세어 suitable_job 반환"""
    counts = {job: 0 for job in JOBS}
    for job, keywords in JOB_KEYWORDS.items():
        for kw in keywords:
            counts[job] += text.count(kw)

    best = max(counts, key=lambda j: counts[j])
    # 모두 0이거나 동점이면 product를 기본값으로
    if counts[best] == 0:
        return "product"
    return best


# ── 직무 분류 (LLM 방식) ────────────────────────────────────────────
def classify_by_llm(question: str, answer: str) -> str:
    """
    OpenAI API로 직무 분류. 실패 시 키워드 방식으로 fallback.
    OPENAI_API_KEY 환경변수 필요.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return classify_by_keyword(answer)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            f"다음 자소서 질문과 답변을 읽고 backend / ai_ml / product 중 "
            f"가장 적합한 IT 직무 하나만 출력하세요. 다른 설명 없이 단어 하나만.\n\n"
            f"질문: {question}\n\n답변: {answer[:500]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in JOBS:
            return label
    except Exception:
        pass

    # fallback
    return classify_by_keyword(answer)


# ── 질문+답변 쌍 추출 ────────────────────────────────────────────────
def is_question(text: str) -> bool:
    """텍스트가 자소서 질문인지 판단 (150자 이내 + 질문 패턴)"""
    t = text.strip()
    if not t or len(t) < 5 or len(t) > 150:
        return False
    patterns = [
        r'^\d+[\.\)]\s',
        r'^\[\d+[-–]\d+\]',
        r'\?$',
        r'(해\s*주\s*세\s*요|말씀해주세요|기술해주세요|서술해주세요|설명해주세요)\s*\.?\s*$',
        r'(경험을\s*(써|작성|기술|서술)|역량을\s*(써|보여|기술))',
    ]
    return any(re.search(p, t) for p in patterns)


def extract_pairs(page_text: str) -> list[tuple[str, str]]:
    """
    페이지 텍스트에서 모든 (질문, 답변) 쌍을 추출.

    줄 단위 스캔 방식:
      - 질문 줄 발견 → 현재 질문 확정, 답변 라인 수집 시작
      - 다음 질문 줄 발견 → 이전 Q&A 쌍 저장 후 새 질문으로 전환
      - 페이지 끝 → 마지막 Q&A 쌍 저장

    단락 파싱 방식보다 줄 단위가 안정적인 이유:
    Playwright inner_text 는 단일 개행(\n)으로 요소를 구분하는 경우가 많아
    double newline 기준 단락 분리 시 Q&A 쌍이 하나의 블록으로 합쳐질 수 있음.
    """
    lines = [l.strip() for l in page_text.splitlines()]

    pairs: list[tuple[str, str]] = []
    current_q: str | None = None
    answer_lines: list[str] = []

    def _flush(q: str, a_lines: list[str]):
        """현재까지 모은 Q&A를 pairs에 추가"""
        answer = " ".join(a_lines).strip()
        if len(answer) < ANSWER_MIN_LEN:
            return
        if len(answer) > ANSWER_MAX_LEN:
            answer = answer[:ANSWER_SAVE_LEN]
        # 번호 접두사 제거
        q = re.sub(r'^\d+[\.\)]\s*', '', q).strip()
        q = re.sub(r'^\[\d+[-–]\d+\]\s*', '', q).strip()
        if q:
            pairs.append((q, answer))

    for line in lines:
        if not line:
            continue

        if is_question(line):
            # 이전 질문+답변 쌍 저장
            if current_q is not None:
                _flush(current_q, answer_lines)
            # 새 질문 시작
            current_q   = line
            answer_lines = []
        elif current_q is not None:
            # 너무 짧은 UI 요소(버튼, 메뉴 등) 제외
            if len(line) > 10:
                answer_lines.append(line)

    # 마지막 쌍 처리
    if current_q is not None:
        _flush(current_q, answer_lines)

    return pairs


def is_it_job(title: str) -> bool:
    return any(kw in title for kw in IT_KEYWORDS)


# ── Playwright 크롤러 ────────────────────────────────────────────────
def crawl_with_playwright(url: str) -> tuple[str, str, int] | None:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return None

    for attempt in range(MAX_RETRY):
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(user_agent=USER_AGENT)
                page    = context.new_page()
                resp    = page.goto(url, timeout=15000, wait_until="networkidle")
                status  = resp.status if resp else 0

                if status in (403, 429):
                    browser.close()
                    if attempt < MAX_RETRY - 1:
                        print(f"  [{status}] {url} — {RETRY_WAIT}초 대기 후 재시도")
                        time.sleep(RETRY_WAIT)
                    continue

                page.wait_for_timeout(2000)
                title = page.title()
                text  = page.inner_text("body")
                browser.close()
                return title, text, status

        except PWTimeout:
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
        except Exception:
            break
    return None


# ── Selenium fallback ────────────────────────────────────────────────
def crawl_with_selenium(url: str) -> tuple[str, str, int] | None:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        return None

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--user-agent={USER_AGENT}")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")

    for attempt in range(MAX_RETRY):
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=opts
            )
            driver.set_page_load_timeout(15)
            driver.get(url)
            time.sleep(3)
            title = driver.title
            text  = driver.find_element("tag name", "body").text
            driver.quit()
            return title, text, 200
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(RETRY_WAIT)
    return None


def fetch_page(url: str) -> tuple[str, str, int] | None:
    result = crawl_with_playwright(url)
    if result is None:
        result = crawl_with_selenium(url)
    return result


# ── CSV / 방문 기록 ──────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_visited(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text(encoding="utf-8").splitlines())


def append_visited(url: str, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(url + "\n")


# ── 수집 현황 카운터 ─────────────────────────────────────────────────
def count_by_class(rows: list[dict]) -> dict[str, int]:
    counts = {j: 0 for j in JOBS}
    for r in rows:
        job = r.get("suitable_job", "")
        if job in counts:
            counts[job] += 1
    return counts


def is_train_full(counts: dict[str, int]) -> bool:
    return all(counts[j] >= CLASS_TRAIN[j] for j in JOBS)


def is_test_full(counts: dict[str, int]) -> bool:
    return all(counts[j] >= CLASS_TEST[j] for j in JOBS)


def progress_str(train_rows: list[dict], test_rows: list[dict], cid: int) -> str:
    tc = count_by_class(train_rows)
    ec = count_by_class(test_rows)
    return (
        f"[진행] "
        f"train(b:{tc['backend']}/{CLASS_TRAIN['backend']} "
        f"a:{tc['ai_ml']}/{CLASS_TRAIN['ai_ml']} "
        f"p:{tc['product']}/{CLASS_TRAIN['product']}) "
        f"test(b:{ec['backend']}/{CLASS_TEST['backend']} "
        f"a:{ec['ai_ml']}/{CLASS_TEST['ai_ml']} "
        f"p:{ec['product']}/{CLASS_TEST['product']}) "
        f"| ID: {cid}"
    )


# ── 메인 크롤러 ──────────────────────────────────────────────────────
def run_crawler(start_id: int, output_dir: Path, label_method: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path   = output_dir / TRAIN_FILE
    test_path    = output_dir / TEST_FILE
    visited_path = output_dir / VISITED_FILE

    train_rows = load_csv(train_path)
    test_rows  = load_csv(test_path)
    visited    = load_visited(visited_path)

    # 중복 방지: (question+answer) 조합 집합
    existing = {
        (r.get("question", "").strip(), r.get("answer", "").strip())
        for r in train_rows + test_rows
    }

    # Ctrl+C 인터럽트 핸들러
    def on_interrupt(sig, frame):
        print("\n\n  [중단] 현재까지 저장합니다.")
        save_csv(train_rows, train_path)
        save_csv(test_rows,  test_path)
        _print_summary(train_rows, test_rows, visited, datetime.now() - start_time)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    start_time = datetime.now()
    print(f"\n크롤링 시작 | ID: {start_id} ~ {ID_END} | 라벨링: {label_method}")
    print(f"목표: train {sum(CLASS_TRAIN.values())}개 / test {sum(CLASS_TEST.values())}개\n")

    total_visited = len(visited)
    it_confirmed  = 0

    for cid in range(start_id, ID_END + 1):
        # 모든 클래스 목표 달성 시 종료
        if is_train_full(count_by_class(train_rows)) and is_test_full(count_by_class(test_rows)):
            print("\n  [완료] 전체 목표 달성!")
            break

        url = f"{BASE_URL}/{cid}"
        if url in visited:
            continue

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        result = fetch_page(url)
        append_visited(url, visited_path)
        visited.add(url)
        total_visited += 1

        if result is None:
            print(f"[실패] ID {cid} - 페이지 로드 불가")
            continue

        title, text, _ = result

        if not is_it_job(title):
            print(f"[스킵] ID {cid} - IT 직무 아님")
            continue

        it_confirmed += 1
        pairs = extract_pairs(text)
        if not pairs:
            print(f"[스킵] ID {cid} - 답변 없음")
            continue

        saved_any = False
        for question, answer in pairs:
            key = (question.strip(), answer.strip())
            if key in existing:
                continue

            # 직무 분류
            if label_method == "llm":
                job = classify_by_llm(question, answer)
            else:
                job = classify_by_keyword(answer)

            train_counts = count_by_class(train_rows)
            test_counts  = count_by_class(test_rows)

            # 저장 대상 결정 (클래스 균형 유지)
            if train_counts[job] < CLASS_TRAIN[job]:
                target = train_rows
                dest   = "train"
            elif test_counts[job] < CLASS_TEST[job]:
                target = test_rows
                dest   = "test"
            else:
                print(f"[스킵] ID {cid} - {job} 목표치 초과")
                continue

            row = {
                "question":     question,
                "answer":       answer,
                "intent_score": "",
                "suitable_job": job,
                "url":          url,
            }
            target.append(row)
            existing.add(key)
            saved_any = True

            print(
                f"[저장] {job} | {url} "
                f"(질문 길이 {len(question)}, 답변 길이 {len(answer)})"
            )

        if saved_any:
            save_csv(train_rows, train_path)
            save_csv(test_rows,  test_path)
            print(progress_str(train_rows, test_rows, cid))

    # 최종 저장 및 요약
    save_csv(train_rows, train_path)
    save_csv(test_rows,  test_path)
    _print_summary(train_rows, test_rows, visited, datetime.now() - start_time,
                   total_visited, it_confirmed)


def _print_summary(
    train_rows: list[dict],
    test_rows: list[dict],
    visited: set[str],
    elapsed,
    total_visited: int = 0,
    it_confirmed: int = 0,
):
    tc = count_by_class(train_rows)
    ec = count_by_class(test_rows)
    print(f"\n{'='*60}")
    print(f"  수집 완료 요약")
    print(f"{'='*60}")
    print(f"  총 방문 페이지   : {total_visited or len(visited)}개")
    print(f"  IT 직무 확인     : {it_confirmed}개")
    print(f"  수집 쌍 (train)  : {len(train_rows)}개")
    print(f"    backend        : {tc['backend']}")
    print(f"    ai_ml          : {tc['ai_ml']}")
    print(f"    product        : {tc['product']}")
    print(f"  수집 쌍 (test)   : {len(test_rows)}개")
    print(f"    backend        : {ec['backend']}")
    print(f"    ai_ml          : {ec['ai_ml']}")
    print(f"    product        : {ec['product']}")
    print(f"  intent_score     : 전부 비워짐 (별도 라벨링 작업 필요)")
    print(f"  소요 시간        : {str(elapsed).split('.')[0]}")
    print(f"{'='*60}\n")


# ── CLI 진입점 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="마커리어 Stage 3 데이터 크롤러")
    parser.add_argument(
        "--start-id", type=int, default=ID_START,
        help=f"시작 ID (기본값: {ID_START})"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/stage3",
        help="저장 경로 (기본값: ./data/stage3)"
    )
    parser.add_argument(
        "--label-method", choices=["keyword", "llm"], default="keyword",
        help="직무 분류 방법: keyword(기본, 무료) / llm(정확, OPENAI_API_KEY 필요)"
    )
    args = parser.parse_args()

    run_crawler(args.start_id, Path(args.output_dir), args.label_method)


if __name__ == "__main__":
    main()
