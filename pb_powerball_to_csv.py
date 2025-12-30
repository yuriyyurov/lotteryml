#!/usr/bin/env python3
"""
Convert Florida Lottery "POWERBALL & POWERBALL DOUBLE PLAY Winning Numbers History" PDF
into a simple CSV suitable for this repo's training scripts.

Outputs only rows where Game == "POWERBALL" (skips "POWERBALL DP").

CSV schema:
  date,n1,n2,n3,n4,n5,pb
Dates are normalized to YYYY-MM-DD.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import pdfplumber


ROW_RE = re.compile(
    r"^\s*(?P<md>\d{1,2})/(?P<dd>\d{1,2})/(?P<yy>\d{2})\s+"
    r"(?P<n1>\d{1,2})\s+(?P<n2>\d{1,2})\s+(?P<n3>\d{1,2})\s+(?P<n4>\d{1,2})\s+(?P<n5>\d{1,2})\s+"
    r"PB\s+(?P<pb>\d{1,2})"
    r"(?:\s+X\d+)?\s+"
    r"(?P<game>POWERBALL|POWERBALL\s+DP)\s*$"
)


def parse_date(md: str, dd: str, yy: str) -> str:
    y = 2000 + int(yy)
    d = dt.date(y, int(md), int(dd))
    return d.isoformat()


def extract_powerball_rows(pdf_path: Path) -> list[tuple[str, int, int, int, int, int, int]]:
    rows: list[tuple[str, int, int, int, int, int, int]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                m = ROW_RE.match(line)
                if not m:
                    continue
                game = re.sub(r"\s+", " ", m.group("game")).strip()
                if game != "POWERBALL":
                    continue

                date_iso = parse_date(m.group("md"), m.group("dd"), m.group("yy"))
                nums = [int(m.group(f"n{i}")) for i in range(1, 6)]
                pb = int(m.group("pb"))
                rows.append((date_iso, *nums, pb))

    # De-dup if the PDF has repeated header/footer artifacts
    rows = list(dict.fromkeys(rows))

    # Sort ascending by date (matches repo expectation: oldest -> newest)
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=Path("pb-history.pdf"))
    ap.add_argument("--out", type=Path, default=Path("powerball.csv"))
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    rows = extract_powerball_rows(args.pdf)
    if not rows:
        raise SystemExit("No POWERBALL rows found. Check PDF format or regex.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "n1", "n2", "n3", "n4", "n5", "pb"])
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()



