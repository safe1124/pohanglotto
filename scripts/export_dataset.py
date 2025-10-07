#!/usr/bin/env python3
"""lotto.sql을 기반으로 웹에서 사용하는 lotto_dataset.js를 생성합니다."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def load_rows_from_sql(sql_path: Path) -> list[tuple]:
    """lotto.sql의 INSERT 구문을 실행해 정렬된 회차 데이터를 반환합니다."""
    if not sql_path.exists():
        raise FileNotFoundError(f"lotto.sql 파일을 찾을 수 없습니다: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")

    with sqlite3.connect(":memory:") as conn:
        conn.executescript(sql_text)
        cursor = conn.execute(
            "SELECT draw_no, n1, n2, n3, n4, n5, n6, bonus FROM lotto ORDER BY draw_no ASC"
        )
        return cursor.fetchall()


def build_dataset(rows: list[tuple]) -> list[dict]:
    """SELECT 결과를 정적 사이트에서 사용하는 JSON 호환 리스트로 변환합니다."""
    dataset: list[dict] = []
    for row in rows:
        draw_no, *numbers, bonus = row
        entry: dict = {
            "round": int(draw_no),
            "numbers": [int(num) for num in numbers],
        }
        if bonus is not None:
            entry["bonus"] = int(bonus)
        dataset.append(entry)
    return dataset


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sql_path = root / "lotto.sql"
    output_path = root / "data" / "lotto_dataset.js"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows_from_sql(sql_path)
    dataset = build_dataset(rows)

    with output_path.open("w", encoding="utf-8") as fp:
        fp.write("window.LOTTO_DATA = ")
        json.dump(dataset, fp, ensure_ascii=False)
        fp.write(";\n")

    print(f"✅ 변환 완료: {len(dataset)}개 회차를 {output_path.relative_to(root)}에 저장했습니다.")


if __name__ == "__main__":
    main()
