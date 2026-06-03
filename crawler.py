"""
마커리어 IT 자소서 질문 수집 크롤러

실행:
    python crawler.py                              # 기본 실행 (ID 30000~37000)
    python crawler.py --start-id 34000             # 중단 후 재개
    python crawler.py --output-dir ./data/stage1   # 저장 경로 지정
    python crawler.py --status                     # 현재 수집 현황만 출력
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
ID_START     = 30000
ID_END       = 37000
TRAIN_TARGET = 500
TEST_TARGET  = 100
DELAY_MIN    = 1.5
DELAY_MAX    = 3.5
MAX_RETRY    = 3
RETRY_WAIT   = 60      # 403/429 응답 시 대기 (초)
BASE_URL     = "https://linkareer.com/cover-letter"
USER_AGENT   = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TRAIN_FILE   = "stage1_train.csv"
TEST_FILE    = "stage1_test.csv"
VISITED_FILE = "visited_urls.txt"
CSV_COLUMNS  = ["question_id", "question", "url", "competency"]

# IT 직무 키워드
IT_KEYWORDS = [
    "IT", "개발", "소프트웨어", "프로그래머", "엔지니어", "데이터",
    "AI", "인공지능", "백엔드", "프론트", "풀스택", "DevOps",
    "클라우드", "보안", "사이버", "시스템", "네트워크", "QA",
]


# ── 질문 추출 ────────────────────────────────────────────────────────
def is_question(text: str) -> bool:
    """텍스트 블록이 질문인지 판단"""
    t = text.strip()

    # 길이 필터: 질문은 150자 이내, 답변(200자 이상)은 제외
    if not t or len(t) < 5 or len(t) > 150:
        return False

    # 질문 패턴 매칭
    patterns = [
        r'^\d+[\.\)]\s',                                        # "1. " "1) " 시작
        r'^\[\d+[-–]\d+\]',                                     # "[1-1]" 형식
        r'\?$',                                                  # 물음표 끝
        r'(해\s*주\s*세\s*요|말씀해주세요|기술해주세요|서술해주세요|설명해주세요)\s*\.?\s*$',
        r'(경험을\s*(써|작성|기술|서술)|역량을\s*(써|보여|기술))',
    ]
    return any(re.search(p, t) for p in patterns)


def extract_questions(page_text: str) -> list[str]:
    """
    페이지 텍스트에서 질문만 추출.
    질문과 답변이 번갈아 등장하므로 짧은 블록(질문) / 긴 블록(답변) 으로 분리.
    """
    # 빈 줄로 단락 분리
    paragraphs = re.split(r'\n{2,}', page_text)
    questions = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 단락 안의 각 줄도 따로 확인 (줄바꿈 1개로 구분된 경우)
        for line in para.splitlines():
            line = line.strip()
            if is_question(line):
                # 번호 접두사 제거
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                cleaned = re.sub(r'^\[\d+[-–]\d+\]\s*', '', cleaned).strip()
                if cleaned and cleaned not in questions:
                    questions.append(cleaned)

    return questions


def is_it_job(title: str) -> bool:
    """페이지 제목 기준 IT 직무 여부 확인"""
    return any(kw in title for kw in IT_KEYWORDS)


# ── Playwright 크롤러 ────────────────────────────────────────────────
def crawl_with_playwright(url: str) -> tuple[str, str, int] | None:
    """
    (title, body_text, status_code) 반환.
    로드 실패 시 None 반환.
    """
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

                response = page.goto(url, timeout=15000, wait_until="networkidle")
                status   = response.status if response else 0

                # 차단 응답: 대기 후 재시도
                if status in (403, 429):
                    browser.close()
                    if attempt < MAX_RETRY - 1:
                        print(f"  [{status}] {url} — {RETRY_WAIT}초 대기 후 재시도")
                        time.sleep(RETRY_WAIT)
                    continue

                page.wait_for_timeout(2000)   # JS 렌더링 완료 대기
                title = page.title()
                text  = page.inner_text("body")
                browser.close()
                return title, text, status

        except PWTimeout:
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
        except Exception as e:
            break

    return None


# ── Selenium fallback ────────────────────────────────────────────────
def crawl_with_selenium(url: str) -> tuple[str, str, int] | None:
    """Playwright 실패 시 Selenium으로 재시도"""
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
    """Playwright → Selenium 순으로 시도"""
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


# ── 현황 출력 ────────────────────────────────────────────────────────
def print_status(output_dir: Path):
    train_rows = load_csv(output_dir / TRAIN_FILE)
    test_rows  = load_csv(output_dir / TEST_FILE)
    visited    = load_visited(output_dir / VISITED_FILE)
    print(f"\n{'='*50}")
    print(f"  수집 현황")
    print(f"{'='*50}")
    print(f"  train   : {len(train_rows)} / {TRAIN_TARGET}")
    print(f"  test    : {len(test_rows)} / {TEST_TARGET}")
    print(f"  방문 URL: {len(visited)}개")
    print(f"  저장 경로: {output_dir.resolve()}")
    print(f"{'='*50}\n")


# ── 메인 크롤러 ──────────────────────────────────────────────────────
def run_crawler(start_id: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path   = output_dir / TRAIN_FILE
    test_path    = output_dir / TEST_FILE
    visited_path = output_dir / VISITED_FILE

    train_rows = load_csv(train_path)
    test_rows  = load_csv(test_path)
    visited    = load_visited(visited_path)

    # 기존 질문 목록 (중복 방지용)
    existing_q = {
        r["question"].strip()
        for r in train_rows + test_rows
        if r.get("question")
    }

    q_counter = len(train_rows) + len(test_rows) + 1

    def make_qid(n: int) -> str:
        return f"Q{n:03d}"

    # Ctrl+C 인터럽트 핸들러: 현재까지 저장 후 종료
    def on_interrupt(sig, frame):
        print("\n\n  [중단] Ctrl+C 감지 — 현재까지 저장합니다.")
        save_csv(train_rows, train_path)
        save_csv(test_rows,  test_path)
        print_status(output_dir)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    start_time = datetime.now()
    print(f"\n크롤링 시작 | ID: {start_id} ~ {ID_END}")
    print(f"목표: train {TRAIN_TARGET}개 / test {TEST_TARGET}개\n")

    for cid in range(start_id, ID_END + 1):
        # 목표 달성 시 조기 종료
        if len(train_rows) >= TRAIN_TARGET and len(test_rows) >= TEST_TARGET:
            print("\n  [완료] 목표 수량 달성!")
            break

        url = f"{BASE_URL}/{cid}"

        # 이미 방문한 URL 스킵
        if url in visited:
            continue

        # 요청 간 지연 (고정 딜레이 금지 — 랜덤)
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        result = fetch_page(url)
        append_visited(url, visited_path)
        visited.add(url)

        if result is None:
            print(f"[실패] ID {cid} - 페이지 로드 불가")
            continue

        title, text, status = result

        # IT 직무 여부 확인
        if not is_it_job(title):
            print(f"[스킵] ID {cid} - IT 직무 아님")
            continue

        # 질문 추출
        questions = extract_questions(text)
        if not questions:
            print(f"[스킵] ID {cid} - 질문 없음 (답변만 있는 페이지)")
            continue

        # 새 질문 저장
        added = 0
        for q in questions:
            if q in existing_q:
                continue

            if len(train_rows) < TRAIN_TARGET:
                target = train_rows
                dest   = "train"
            elif len(test_rows) < TEST_TARGET:
                target = test_rows
                dest   = "test"
            else:
                break

            target.append({
                "question_id": make_qid(q_counter),
                "question":    q,
                "url":         url,
                "competency":  "",
            })
            existing_q.add(q)
            q_counter += 1
            added += 1

        if added > 0:
            save_csv(train_rows, train_path)
            save_csv(test_rows,  test_path)
            print(
                f"[진행] train: {len(train_rows)}/{TRAIN_TARGET} | "
                f"test: {len(test_rows)}/{TEST_TARGET} | "
                f"현재 ID: {cid}"
            )
            print(
                f"[저장] Q{q_counter-added:03d}~Q{q_counter-1:03d} "
                f"({added}개) 수집 완료 → {url}"
            )
        else:
            print(f"[스킵] ID {cid} - 중복 질문만 있음")

    # 최종 저장 및 요약 리포트
    save_csv(train_rows, train_path)
    save_csv(test_rows,  test_path)

    elapsed = datetime.now() - start_time
    total_visited = len(load_visited(visited_path))
    it_pages = len(train_rows) + len(test_rows)

    print(f"\n{'='*55}")
    print(f"  수집 완료 요약")
    print(f"{'='*55}")
    print(f"  총 방문 페이지 : {total_visited}개")
    print(f"  IT 직무 확인   : {it_pages}개 페이지")
    print(f"  수집된 질문    : train {len(train_rows)}개 / test {len(test_rows)}개")
    print(f"  소요 시간      : {str(elapsed).split('.')[0]}")
    print(f"  저장 경로      : {output_dir.resolve()}")
    print(f"{'='*55}\n")


# ── CLI 진입점 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="마커리어 IT 자소서 질문 크롤러")
    parser.add_argument(
        "--start-id", type=int, default=ID_START,
        help=f"시작 ID (기본값: {ID_START}). 중단 후 재개 시 사용."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/stage1",
        help="CSV 저장 경로 (기본값: ./data/stage1)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 수집 현황만 출력하고 종료"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.status:
        print_status(output_dir)
        return

    run_crawler(args.start_id, output_dir)


if __name__ == "__main__":
    main()
