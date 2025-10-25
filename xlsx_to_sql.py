import re
import sys
from pathlib import Path

import pandas as pd


def normalize_column_name(name: str) -> str:
    # Lowercase, strip spaces, remove special chars for easier matching
    n = str(name).strip().lower()
    n = re.sub(r"\s+", "", n)
    n = n.replace("-", "").replace("_", "")
    return n


def extract_int(value):
    if pd.isna(value):
        return None
    s = str(value)
    # remove everything except digits
    digits = re.sub(r"[^0-9]", "", s)
    if digits == "":
        return None
    try:
        return int(digits)
    except Exception:
        return None


def detect_columns(df: pd.DataFrame):
    normalized = {col: normalize_column_name(col) for col in df.columns}

    # draw number candidates
    draw_keys = [
        "회차",
        "추첨회차",
        "회",  # sometimes just "n회" style
        "draw",
        "round",
        "drawno",
        "roundno",
    ]

    # bonus number indicators
    bonus_keys = ["보너스", "bonus", "보너스번호", "bonusnumber"]

    # prize amount indicators
    prize_keys = [
        "당첨금액",
        "1등당첨금액",
        "총당첨금",
        "상금",
        "prize",
        "prizeamount",
        "jackpot",
    ]

    draw_col = None
    bonus_col = None
    prize_col = None
    number_cols = []

    # First, try explicit picks: n1..n6 like names
    explicit_number_patterns = [
        re.compile(r"^(?:num|no|n|번호|당첨번호)?0*1$"),
        re.compile(r"^(?:num|no|n|번호|당첨번호)?0*2$"),
        re.compile(r"^(?:num|no|n|번호|당첨번호)?0*3$"),
        re.compile(r"^(?:num|no|n|번호|당첨번호)?0*4$"),
        re.compile(r"^(?:num|no|n|번호|당첨번호)?0*5$"),
        re.compile(r"^(?:num|no|n|번호|당첨번호)?0*6$"),
    ]

    # Pass 1: identify draw, bonus, prize by keywords
    for orig, norm in normalized.items():
        # draw
        if draw_col is None and any(k in norm for k in draw_keys):
            draw_col = orig
        # bonus
        if bonus_col is None and any(k in norm for k in bonus_keys):
            bonus_col = orig
        # prize
        if prize_col is None and any(k in norm for k in prize_keys):
            prize_col = orig

    # Pass 2: numbers. Prefer columns that look like position 1..6 or contain '당첨번호'
    candidates = []
    for orig, norm in normalized.items():
        if orig in {draw_col, bonus_col, prize_col}:
            continue
        if "당첨번호" in norm or norm.startswith("num") or norm.startswith("번호") or norm.startswith("n"):
            candidates.append((orig, norm))

    # If we found at least 6, try to sort by trailing number
    def trailing_int(s: str):
        m = re.search(r"(\d+)$", s)
        return int(m.group(1)) if m else 999

    if len(candidates) >= 6:
        candidates.sort(key=lambda x: trailing_int(x[1]))
        number_cols = [c[0] for c in candidates[:6]]

    # If not enough, try explicit patterns against all columns
    if len(number_cols) < 6:
        matched = []
        for patt in explicit_number_patterns:
            for orig, norm in normalized.items():
                if orig in {draw_col, bonus_col, prize_col}:
                    continue
                if patt.match(norm):
                    matched.append(orig)
                    break
        if len(matched) >= 6:
            number_cols = matched[:6]

    # Fallback: try columns that contain only numeric-looking data and have 6 smallest-median
    if len(number_cols) < 6:
        numeric_like = []
        for orig in df.columns:
            if orig in {draw_col, bonus_col, prize_col}:
                continue
            series = df[orig].dropna().astype(str)
            digits_ratio = (series.str.replace(r"[^0-9]", "", regex=True).str.len() > 0).mean() if len(series) else 0
            if digits_ratio > 0.9:
                numeric_like.append(orig)
        if len(numeric_like) >= 6:
            number_cols = numeric_like[:6]

    # As a final safety, if bonus was not identified but we have 7 numeric-like columns, treat last as bonus
    if bonus_col is None and len(number_cols) >= 7:
        bonus_col = number_cols[6]
        number_cols = number_cols[:6]

    # Ensure we have essentials
    if draw_col is None:
        # Try any column with mostly increasing integer-like values or containing '회'
        for orig, norm in normalized.items():
            if any(k in norm for k in ["회", "round", "draw"]):
                draw_col = orig
                break

    if len(number_cols) != 6:
        raise ValueError(
            f"Unable to reliably detect 6 winning number columns. Detected: {number_cols}"
        )

    return {
        "draw": draw_col,
        "numbers": number_cols,
        "bonus": bonus_col,
        "prize": prize_col,
    }


def build_sql(df: pd.DataFrame, cols: dict, table_name: str = "lotto") -> str:
    # Schema: draw_no INTEGER, n1..n6 INTEGER, bonus INTEGER NULL, prize_amount INTEGER NULL
    create_stmt = (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        "  draw_no INTEGER,\n"
        "  n1 INTEGER,\n"
        "  n2 INTEGER,\n"
        "  n3 INTEGER,\n"
        "  n4 INTEGER,\n"
        "  n5 INTEGER,\n"
        "  n6 INTEGER,\n"
        "  bonus INTEGER,\n"
        "  prize_amount INTEGER\n"
        ");\n\n"
    )

    lines = [create_stmt]

    draw_col = cols.get("draw")
    ncols = cols["numbers"]
    bonus_col = cols.get("bonus")
    prize_col = cols.get("prize")

    for _, row in df.iterrows():
        draw_no = extract_int(row[draw_col]) if draw_col in row else None
        nums = [extract_int(row[c]) for c in ncols]
        bonus = extract_int(row[bonus_col]) if bonus_col and bonus_col in row else None
        prize = extract_int(row[prize_col]) if prize_col and prize_col in row else None

        # skip rows without essential fields
        if draw_no is None or any(n is None for n in nums):
            continue

        values = [draw_no] + nums + [bonus if bonus is not None else "NULL"] + [prize if prize is not None else "NULL"]

        # Build VALUES string with NULL support
        def v(x):
            return "NULL" if x == "NULL" else str(int(x))

        values_sql = ", ".join(v(x) for x in values)
        lines.append(f"INSERT INTO {table_name} VALUES ({values_sql});")

    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).resolve().parent
    xlsx_path = root / "lotto.xlsx"
    if len(sys.argv) > 1:
        xlsx_path = Path(sys.argv[1]).expanduser().resolve()

    if not xlsx_path.exists():
        print(f"❌ Excel file not found: {xlsx_path}")
        sys.exit(1)

    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"❌ Failed to read Excel: {e}")
        sys.exit(1)

    # Drop empty rows/columns
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"Unnamed: \d+")]

    try:
        cols = detect_columns(df)
    except Exception as e:
        print(f"❌ Column detection error: {e}")
        print("Columns found:", list(df.columns))
        sys.exit(1)

    sql = build_sql(df, cols, table_name="lotto")
    out_path = root / "lotto.sql"
    out_path.write_text(sql, encoding="utf-8")
    print(f"✅ Generated SQL at: {out_path}")


if __name__ == "__main__":
    main()


