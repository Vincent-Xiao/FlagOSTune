#!/usr/bin/env python3
"""
Merge the final summary tables from:
- reports/bench-report-<DATE>.md
- reports/bench-plugin-report-<DATE>.md

Specifically:
- Merge both files' "Output Token Mean Throughput （tokens/s）对比" tables
  into a single combined table.
- Merge both files' "Total Token Mean Throughput （tokens/s）对比" tables
  into a single combined table.

Write the result to:
- reports/bench-report-merge-<DATE>.md

Usage (inside conda `xh_vllm` environment):

    (xh_vllm) $ python processing/merge_plugin.py --f 2026-01-28
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import List, Tuple, Dict


OUTPUT_MARKER = "## Output Token Mean Throughput （tokens/s）对比"
TOTAL_MARKER = "## Total Token Mean Throughput （tokens/s）对比"


def extract_table_after_marker(text: str, marker: str) -> Tuple[List[str], List[List[str]]]:
    """
    从文本中找到最后一次出现的 marker 标题，提取紧跟其后的第一个 markdown 表格。
    返回 (header, rows)，其中 header 是列名列表，rows 是每行的列值列表。
    """
    lines = text.splitlines()

    # 找到最后一次出现 marker 的行号
    marker_idx = None
    for i, line in enumerate(lines):
        if line.strip() == marker:
            marker_idx = i
    if marker_idx is None:
        raise RuntimeError(f"Marker '{marker}' not found")

    # 从 marker 之后开始找第一个以 '|' 开头的表格
    i = marker_idx + 1
    # 跳过空行
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not lines[i].lstrip().startswith("|"):
        raise RuntimeError(f"No table found after marker '{marker}'")

    table_lines: List[str] = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        table_lines.append(lines[i])
        i += 1

    if len(table_lines) < 2:
        raise RuntimeError(f"Incomplete table after marker '{marker}'")

    # 解析 markdown 表格
    def parse_row(line: str) -> List[str]:
        # 去掉前后竖线，然后按 '|' 分割并 strip
        inner = line.strip().strip("|")
        return [c.strip() for c in inner.split("|")]

    header = parse_row(table_lines[0])
    # 跳过分隔行 (第二行), 其余为数据行
    rows = [parse_row(l) for l in table_lines[2:]]
    return header, rows


def merge_tables(
    base_header: List[str],
    base_rows: List[List[str]],
    plugin_header: List[str],
    plugin_rows: List[List[str]],
) -> Tuple[List[str], List[List[str]]]:
    """
    按 Scenario 行合并两个表，并将加速比列放在所有原始列之后。

    规则：
    - 第一列必须是 'Scenario'
    - 其余列按照是否为加速比列（包含 '/cuda'）拆分：
        * 先放 base 的非加速比列
        * 再放 plugin 的非加速比列
        * 然后 base 的加速比列
        * 最后 plugin 的加速比列
    - 行集合为两个表 Scenario 的并集，按 base 的顺序优先，其次是 plugin 独有的行。
    - 对缺失的部分用 'N/A' 填充。
    """
    if not base_header or not plugin_header:
        raise RuntimeError("Empty header in tables to merge")
    if base_header[0].lower() != "scenario" or plugin_header[0].lower() != "scenario":
        raise RuntimeError("First column of both tables must be 'Scenario'")

    def split_indices(header: List[str]) -> Tuple[List[int], List[int]]:
        """返回 (value_indices, ratio_indices)，索引基于整行（含 Scenario）。"""
        value_idx: List[int] = []
        ratio_idx: List[int] = []
        for j, name in enumerate(header[1:], start=1):
            if "/cuda" in name:
                ratio_idx.append(j)
            else:
                value_idx.append(j)
        return value_idx, ratio_idx

    base_val_idx, base_ratio_idx = split_indices(base_header)
    plugin_val_idx, plugin_ratio_idx = split_indices(plugin_header)

    base_map: Dict[str, List[str]] = {r[0]: r for r in base_rows}
    plugin_map: Dict[str, List[str]] = {r[0]: r for r in plugin_rows}

    scenarios: List[str] = []
    for s in [r[0] for r in base_rows]:
        if s not in scenarios:
            scenarios.append(s)
    for s in [r[0] for r in plugin_rows]:
        if s not in scenarios:
            scenarios.append(s)

    merged_header: List[str] = ["Scenario"]
    merged_header += [base_header[i] for i in base_val_idx]
    merged_header += [plugin_header[i] for i in plugin_val_idx]
    merged_header += [base_header[i] for i in base_ratio_idx]
    merged_header += [plugin_header[i] for i in plugin_ratio_idx]

    def gather_cells(
        row_map: Dict[str, List[str]],
        scen: str,
        indices: List[int],
    ) -> List[str]:
        if scen not in row_map:
            return ["N/A"] * len(indices)
        row = row_map[scen]
        return [row[i] if i < len(row) else "N/A" for i in indices]

    merged_rows: List[List[str]] = []
    for scen in scenarios:
        row: List[str] = [scen]
        # base 非加速比列
        row.extend(gather_cells(base_map, scen, base_val_idx))
        # plugin 非加速比列
        row.extend(gather_cells(plugin_map, scen, plugin_val_idx))
        # base 加速比列
        row.extend(gather_cells(base_map, scen, base_ratio_idx))
        # plugin 加速比列
        row.extend(gather_cells(plugin_map, scen, plugin_ratio_idx))
        merged_rows.append(row)

    return merged_header, merged_rows


def table_to_markdown(header: List[str], rows: List[List[str]]) -> str:
    """将 (header, rows) 转为 markdown 表格字符串。"""
    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge base and plugin benchmark reports for a given date.")
    parser.add_argument(
        "--f",
        dest="date",
        type=str,
        required=True,
        help="Date string like 2026-01-28 to choose which reports to merge.",
    )
    args = parser.parse_args()

    date_str = args.date

    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"

    base_path = reports_dir / f"bench-report-{date_str}.md"
    plugin_path = reports_dir / f"bench-plugin-report-{date_str}.md"
    out_path = reports_dir / f"bench-report-merge-{date_str}.md"

    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} not found")
    if not plugin_path.exists():
        raise FileNotFoundError(f"{plugin_path} not found")

    base_text = base_path.read_text(encoding="utf-8")
    plugin_text = plugin_path.read_text(encoding="utf-8")

    # 提取 Output 表
    base_out_header, base_out_rows = extract_table_after_marker(base_text, OUTPUT_MARKER)
    plugin_out_header, plugin_out_rows = extract_table_after_marker(plugin_text, OUTPUT_MARKER)
    merged_out_header, merged_out_rows = merge_tables(
        base_out_header, base_out_rows, plugin_out_header, plugin_out_rows
    )

    # 提取 Total 表
    base_tot_header, base_tot_rows = extract_table_after_marker(base_text, TOTAL_MARKER)
    plugin_tot_header, plugin_tot_rows = extract_table_after_marker(plugin_text, TOTAL_MARKER)
    merged_tot_header, merged_tot_rows = merge_tables(
        base_tot_header, base_tot_rows, plugin_tot_header, plugin_tot_rows
    )

    # 生成合并报告
    parts: List[str] = []
    parts.append(f"# Merged Benchmark Report {date_str}")
    parts.append("")
    parts.append("## Output Token Mean Throughput （tokens/s）对比")
    parts.append("")
    parts.append(table_to_markdown(merged_out_header, merged_out_rows))
    parts.append("")
    parts.append("## Total Token Mean Throughput （tokens/s）对比")
    parts.append("")
    parts.append(table_to_markdown(merged_tot_header, merged_tot_rows))
    parts.append("")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Merged report written to: {out_path}")


if __name__ == "__main__":
    main()

