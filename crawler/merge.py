"""
기존 batch CSV 파일들과 크롤러 결과를 병합하여
stage1_train.csv / stage1_test.csv 생성

실행 (test4 루트에서):
    python crawler/merge.py
    python crawler/merge.py --input-dir ./data/stage1 --batch-dir ./data/stage1/batches
"""

import argparse
import csv
from pathlib import Path

TRAIN_TARGET = 500
TEST_TARGET  = 100
CSV_COLUMNS  = ["question_id", "question", "url", "competency"]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [경고] 파일 없음: {path}")
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def merge(input_dir: Path, batch_dir: Path):
    all_rows: list[dict] = []

    # 기존 batch 파일 로드 (batch_01.csv ~ batch_04.csv 등)
    if batch_dir.exists():
        batch_files = sorted(batch_dir.glob("batch_*.csv"))
        if batch_files:
            print("\n[1] 기존 batch 파일 로드")
            for bf in batch_files:
                rows = load_csv(bf)
                all_rows.extend(rows)
                print(f"    {bf.name}: {len(rows)}개")
        else:
            print(f"\n[1] batch 파일 없음: {batch_dir}")
    else:
        print(f"\n[1] batch 디렉토리 없음: {batch_dir}")

    # 크롤러로 새로 수집한 파일도 병합
    print("\n[2] 크롤러 수집 파일 로드")
    for crawler_file in sorted(input_dir.glob("stage1_*.csv")):
        rows = load_csv(crawler_file)
        all_rows.extend(rows)
        print(f"    {crawler_file.name}: {len(rows)}개")

    print(f"\n    병합 전 총 행 수: {len(all_rows)}개")

    # 중복 제거 (question 기준, strip 후 비교)
    seen: set[str] = set()
    deduped: list[dict] = []
    dup_count = 0

    for row in all_rows:
        q = (row.get("question") or "").strip()
        if not q:
            continue
        if q in seen:
            dup_count += 1
            continue
        seen.add(q)
        deduped.append(row)

    print(f"    중복 제거: {dup_count}개 → 남은 항목: {len(deduped)}개")

    if len(deduped) == 0:
        print("\n  [경고] 병합할 데이터가 없습니다.")
        return

    # question_id Q001 부터 재부여
    for i, row in enumerate(deduped, 1):
        row["question_id"] = f"Q{i:03d}"
        # competency 컬럼 없는 경우 빈칸으로 통일
        if "competency" not in row:
            row["competency"] = ""
        # url 컬럼 없는 경우 빈칸으로 통일
        if "url" not in row:
            row["url"] = ""

    # train / test 분리
    train_rows = deduped[:TRAIN_TARGET]
    test_rows  = deduped[TRAIN_TARGET:TRAIN_TARGET + TEST_TARGET]
    extra      = len(deduped) - len(train_rows) - len(test_rows)

    train_path = input_dir / "stage1_train.csv"
    test_path  = input_dir / "stage1_test.csv"

    save_csv(train_rows, train_path)
    save_csv(test_rows,  test_path)

    print(f"\n[3] 저장 완료")
    print(f"    train ({len(train_rows)}개) → {train_path}")
    print(f"    test  ({len(test_rows)}개) → {test_path}")

    if extra > 0:
        print(f"    [참고] 목표 초과 {extra}개는 저장되지 않음")
    if len(train_rows) < TRAIN_TARGET:
        print(f"    [경고] train 목표 미달: {len(train_rows)}/{TRAIN_TARGET}")
    if len(test_rows) < TEST_TARGET:
        print(f"    [경고] test 목표 미달: {len(test_rows)}/{TEST_TARGET}")


def main():
    parser = argparse.ArgumentParser(description="batch CSV 병합 스크립트")
    parser.add_argument(
        "--input-dir", default="./data/stage1",
        help="크롤러 결과 및 최종 출력 경로 (기본값: ./data/stage1)"
    )
    parser.add_argument(
        "--batch-dir", default="./data/stage1/batches",
        help="기존 batch 파일 경로 (기본값: ./data/stage1/batches)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    batch_dir = Path(args.batch_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  batch CSV 병합 시작")
    print("=" * 50)

    merge(input_dir, batch_dir)

    print("\n완료!\n")


if __name__ == "__main__":
    main()
