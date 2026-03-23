#!/usr/bin/env python3
"""Summarize latest shape report total-throughput tables across models.

Inputs:
 - reports/<model>/bench-shape-report-YYYY-MM-DD.md

Outputs:
 - reports/flagtune_summary.md
 - reports/flagtune_summary.xlsx (one model per sheet)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import sys


TOTAL_SECTION_PATTERN = re.compile(r"^##\s+Total Token Mean Throughput", re.IGNORECASE)
REPORT_PATTERN = re.compile(r"^bench-shape-report-(\d{4}-\d{2}-\d{2})\.md$")


@dataclass
class ModelTable:
    model: str
    report_path: Path
    header: list[str]
    rows: list[list[str]]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def parse_markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 2:
        return [], []

    def split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = split_row(table_lines[0])
    rows = []
    for line in table_lines[2:]:
        row = split_row(line)
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))
        elif len(row) > len(header):
            row = row[: len(header)]
        rows.append(row)
    return header, rows


def extract_total_table(report_path: Path) -> tuple[list[str], list[list[str]]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if TOTAL_SECTION_PATTERN.match(line.strip()):
            start = i + 1
            break

    if start is None:
        return [], []

    block: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## ") and block:
            break
        if stripped.startswith("|") or (not stripped and block):
            block.append(line)
        elif block:
            break

    return parse_markdown_table(block)


def normalize_pretune_token(name: str) -> str:
    return (
        name.replace("_pretune", "")
        .replace("pretune_", "")
        .replace("_flagtune", "")
        .replace("flagtune_", "")
    )


def is_tuning_variant(name: str) -> bool:
    lower = name.lower()
    return "pretune" in lower or "flagtune" in lower


def rename_header_labels(header: list[str]) -> list[str]:
    return [h.replace("pretune", "flagtune") for h in header]


def to_float(value: str):
    text = "" if value is None else str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def drop_stat_rows(rows: list[list[str]]) -> list[list[str]]:
    filtered = []
    for row in rows:
        if not row:
            continue
        tag = str(row[0]).strip().lower()
        if tag in {"min", "max", "average"}:
            continue
        filtered.append(row)
    return filtered


def append_flagtune_gain_column(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    gain_col = "flagtune_gain"
    if gain_col in header:
        return header, rows

    new_header = header + [gain_col]

    try:
        base_idx = header.index("gems_mm")
        tuned_idx = header.index("gems_mm_flagtune")
    except ValueError:
        return new_header, [row + ["N/A"] for row in rows]

    new_rows = []
    for row in rows:
        safe_row = row + [""] * (len(header) - len(row))
        base = to_float(safe_row[base_idx])
        tuned = to_float(safe_row[tuned_idx])
        if base is None or base == 0 or tuned is None:
            gain = "N/A"
        else:
            gain = f"{((tuned - base) / base) * 100:.2f}%"
        new_rows.append(safe_row + [gain])

    return new_header, new_rows


def reorder_columns(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not header:
        return header, rows

    scenario_col = "Scenario" if "Scenario" in header else header[0]

    value_cols = [c for c in header if c != scenario_col and "/" not in c]
    ratio_cols = [c for c in header if "/" in c]

    def order_grouped(cols: list[str], key_fn):
        if not cols:
            return []
        first_seen: dict[str, int] = {}
        for i, c in enumerate(cols):
            k = key_fn(c)
            if k not in first_seen:
                first_seen[k] = i

        def sort_key(c: str):
            k = key_fn(c)
            is_pretune = 1 if is_tuning_variant(c) else 0
            return (first_seen[k], is_pretune)

        return sorted(cols, key=sort_key)

    ordered_values = order_grouped(value_cols, lambda c: normalize_pretune_token(c.lower()))

    def ratio_group(col: str) -> str:
        left, _, right = col.partition("/")
        return normalize_pretune_token(left.lower()) + "/" + right.lower()

    ordered_ratios = order_grouped(ratio_cols, ratio_group)

    ordered_header = [scenario_col] + ordered_values + ordered_ratios
    idx = [header.index(c) for c in ordered_header]

    ordered_rows: list[list[str]] = []
    for row in rows:
        safe_row = row + [""] * (len(header) - len(row))
        ordered_rows.append([safe_row[i] for i in idx])

    return rename_header_labels(ordered_header), ordered_rows


def discover_latest_date(reports_dir: Path) -> str:
    dates = set()
    for p in reports_dir.glob("*/bench-shape-report-*.md"):
        m = REPORT_PATTERN.match(p.name)
        if m:
            dates.add(m.group(1))
    if not dates:
        raise FileNotFoundError("No bench-shape-report-YYYY-MM-DD.md found in reports/<model>/")
    return max(dates)


def collect_model_tables(reports_dir: Path, day: str) -> list[ModelTable]:
    tables: list[ModelTable] = []
    for model_dir in sorted([p for p in reports_dir.iterdir() if p.is_dir()]):
        report = model_dir / f"bench-shape-report-{day}.md"
        if not report.exists():
            continue
        header, rows = extract_total_table(report)
        if not header or not rows:
            print(f"[WARN] Skip {model_dir.name}: no total throughput table in {report.name}")
            continue
        header, rows = reorder_columns(header, rows)
        rows = drop_stat_rows(rows)
        header, rows = append_flagtune_gain_column(header, rows)
        tables.append(ModelTable(model=model_dir.name, report_path=report, header=header, rows=rows))
    return tables


def write_markdown(out_path: Path, tables: list[ModelTable]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for t in tables:
            f.write(f"## {t.model}\n")
            f.write("\n")
            f.write("| " + " | ".join(t.header) + " |\n")
            f.write("| " + " | ".join(["---"] * len(t.header)) + " |\n")
            for row in t.rows:
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")


def sanitize_sheet_name(name: str, used: set[str]) -> str:
    safe = re.sub(r"[\\/*?:\[\]]", "_", name).strip() or "Sheet"
    safe = safe[:31]
    if safe not in used:
        used.add(safe)
        return safe

    base = safe[:28] if len(safe) > 28 else safe
    i = 2
    while True:
        cand = f"{base}_{i}"[:31]
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def write_excel_xlsx(out_path: Path, tables: list[ModelTable]) -> None:
    from openpyxl import Workbook  # type: ignore

    def maybe_number(value: str):
        text = "" if value is None else str(value).strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            try:
                return float(text)
            except Exception:
                return text
        return text

    used: set[str] = set()
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    for t in tables:
        sheet = sanitize_sheet_name(t.model, used)
        ws = workbook.create_sheet(title=sheet)
        ws.append(t.header)

        for row in t.rows:
            ws.append([maybe_number(c) for c in row])

    workbook.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize latest Total Token Mean Throughput tables")
    parser.add_argument("--reports-dir", default="reports", help="Reports root directory (default: reports)")
    parser.add_argument("--date", default="", help="Optional date YYYY-MM-DD; default latest available")
    args = parser.parse_args()

    root = project_root()
    reports_dir = root / args.reports_dir
    if not reports_dir.exists():
        print(f"Error: reports directory not found: {reports_dir}")
        return 1

    day = args.date or discover_latest_date(reports_dir)
    tables = collect_model_tables(reports_dir, day)
    if not tables:
        print(f"Error: no usable bench-shape-report for date {day}")
        return 1

    md_out = reports_dir / "flagtune_summary.md"
    xlsx_out = reports_dir / "flagtune_summary.xlsx"
    write_markdown(md_out, tables)
    try:
        write_excel_xlsx(xlsx_out, tables)
    except Exception as e:
        print(f"Error: failed to write Excel with openpyxl: {e}")
        return 1

    print(f"Summary date: {day}")
    print(f"Markdown written: {md_out}")
    print(f"Excel written: {xlsx_out}")
    print(f"Models included: {', '.join([t.model for t in tables])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
