#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

DEFAULT_INFO_RE = re.compile(r"\[INFO\].*with default configuration\.")
EXPAND_INFO_RE = re.compile(r"\[INFO\].*with expand configuration\.")
DATA_LINE_RE = re.compile(
    r"^(?P<status>\S+)\s+"
    r"(?P<torch>[\d.eE+-]+)\s+"
    r"(?P<gems>[\d.eE+-]+)\s+"
    r"(?P<speedup>[\d.eE+-]+)\s+"
    r"(?P<tflops>[\d.eE+-]+)\s+"
    r"(?P<shape>\[.*\])\s*$"
)
TORCH_SIZE_RE = re.compile(r"torch\.Size\(\[(?P<dims>[^\]]+)\]\)")

OP_SHAPE_SOURCE_MAP = {
    "w8a8_block_fp8_matmul": "mm",
    "w8a8_block_fp8_matmul_deepgemm": "mm",
}

TABLE_HEADERS = [
    "Shape (B, M, N, K)",
    "Count",
    "Torch Latency with default configuration (ms)",
    "Gems Latency with default configuration (ms)",
    "Gems Speedup with default configuration",
    "Torch Latency with expand configuration (ms)",
    "Gems Latency with expand configuration (ms)",
    "Gems Speedup with expand configuration",
    "Speedup Gain",
]


def parse_count_map(count_yaml_path: Path, target_op: str) -> dict[tuple[int, int, int, int], int]:
    count_map: dict[tuple[int, int, int, int], int] = {}
    current_op: str | None = None
    in_shapes = False
    current_shape: list[int] | None = None
    target_shape_op = OP_SHAPE_SOURCE_MAP.get(target_op, target_op)

    with count_yaml_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            op_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\:$", line)
            if op_match:
                current_op = op_match.group(1)
                in_shapes = False
                current_shape = None
                continue

            if current_op != target_shape_op:
                continue

            if line.strip() == "shapes:":
                in_shapes = True
                continue

            if line.strip().startswith("shape_desc:"):
                in_shapes = False
                current_shape = None
                continue

            if not in_shapes:
                continue

            if line.startswith("  - - "):
                first = int(line.split("  - - ", 1)[1].strip())
                current_shape = [first]
                continue

            dim_match = re.match(r"^\s*-\s*(\d+)\s*$", line)
            if dim_match and current_shape is not None:
                current_shape.append(int(dim_match.group(1)))
                if len(current_shape) == 5:
                    b, m, n, k, count = current_shape
                    count_map[(b, m, n, k)] = count
                    current_shape = None

    return count_map


def infer_bmnk_from_shape_text(shape_text: str) -> tuple[int, int, int, int] | None:
    sizes: list[tuple[int, ...]] = []
    for match in TORCH_SIZE_RE.finditer(shape_text):
        dims = tuple(
            int(token.strip())
            for token in match.group("dims").split(",")
            if token.strip()
        )
        sizes.append(dims)

    if len(sizes) < 2:
        return None

    first, second = sizes[0], sizes[1]
    if len(first) != 2 or len(second) != 2:
        return None

    m, k1 = first
    dim2_a, dim2_b = second

    if dim2_a == k1:
        return (1, m, dim2_b, k1)
    if dim2_b == k1:
        return (1, m, dim2_a, k1)

    return None


def convert_shape_to_bmnk_and_count(
    shape_text: str, count_map: dict[tuple[int, int, int, int], int]
) -> tuple[str, str]:
    bmnk = infer_bmnk_from_shape_text(shape_text)
    if bmnk is None:
        return shape_text, "-"

    count = count_map.get(bmnk, "-")
    b, m, n, k = bmnk
    return f"{b}, {m}, {n}, {k}", str(count)


def calc_gain_percent(default_speedup: str, expand_speedup: str) -> str:
    try:
        default_val = float(default_speedup)
        expand_val = float(expand_speedup)
    except ValueError:
        return "-"

    if default_val == 0:
        return "-"

    gain = (expand_val / default_val - 1.0) * 100.0
    return f"{gain:.2f}%"


def parse_gain_value(gain_text: str) -> float:
    if not gain_text.endswith("%"):
        return float("-inf")
    try:
        return float(gain_text[:-1])
    except ValueError:
        return float("-inf")


def parse_count_value(count_text: str) -> int:
    try:
        return int(count_text)
    except ValueError:
        return -1


def parse_model_yaml(model_yaml_path: Path) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current_block: dict[str, object] | None = None
    in_shapes = False
    current_shape: list[int] | None = None

    with model_yaml_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            op_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\:$", line)
            if op_match:
                current_block = {
                    "op": op_match.group(1),
                    "shapes": [],
                    "shape_desc": None,
                }
                blocks.append(current_block)
                in_shapes = False
                current_shape = None
                continue

            if current_block is None:
                continue

            if line.strip() == "shapes:":
                in_shapes = True
                continue

            if line.strip().startswith("shape_desc:"):
                current_block["shape_desc"] = line.split(":", 1)[1].strip()
                in_shapes = False
                current_shape = None
                continue

            if not in_shapes:
                continue

            if line.startswith("  - - "):
                first = int(line.split("  - - ", 1)[1].strip())
                current_shape = [first]
                current_block["shapes"].append(current_shape)
                continue

            dim_match = re.match(r"^\s*-\s*(\d+)\s*$", line)
            if dim_match and current_shape is not None:
                current_shape.append(int(dim_match.group(1)))

    return blocks


def write_model_shapes_yaml(output_path: Path, blocks: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for idx, block in enumerate(blocks):
            op = block["op"]
            shapes = block["shapes"]
            shape_desc = block["shape_desc"]
            file.write(f"{op}:\n")
            file.write("  shapes:\n")
            for shape in shapes:
                file.write("  - - ")
                file.write(f"{shape[0]}\n")
                for dim in shape[1:]:
                    file.write(f"    - {dim}\n")
            if shape_desc:
                file.write(f"  shape_desc: {shape_desc}\n")
            if idx != len(blocks) - 1:
                file.write("\n")


def split_and_write_gain_lose_yaml(
    model_yaml_path: Path,
    gain_yaml_path: Path,
    lose_yaml_path: Path,
    op: str,
    rows_by_gain: list[list[str]],
) -> tuple[int, int]:
    source_blocks = parse_model_yaml(model_yaml_path)
    shape_to_gain: dict[tuple[int, int, int, int], float] = {}
    target_shape_op = OP_SHAPE_SOURCE_MAP.get(op, op)

    for row in rows_by_gain:
        shape_text = row[0]
        gain_text = row[8]
        parts = [part.strip() for part in shape_text.split(",")]
        if len(parts) != 4:
            continue
        try:
            key = tuple(int(part) for part in parts)
        except ValueError:
            continue
        shape_to_gain[key] = parse_gain_value(gain_text)

    gain_blocks: list[dict[str, object]] = []
    lose_blocks: list[dict[str, object]] = []
    gain_count = 0
    lose_count = 0

    for block in source_blocks:
        block_op = block["op"]
        block_shapes = block["shapes"]
        block_shape_desc = block["shape_desc"]

        if block_op != target_shape_op:
            copied_shapes = [shape.copy() for shape in block_shapes]
            gain_blocks.append({"op": block_op, "shapes": copied_shapes, "shape_desc": block_shape_desc})
            lose_blocks.append({"op": block_op, "shapes": copied_shapes, "shape_desc": block_shape_desc})
            continue

        gain_shapes: list[list[int]] = []
        lose_shapes: list[list[int]] = []
        for shape in block_shapes:
            if len(shape) < 4:
                continue
            key = (shape[0], shape[1], shape[2], shape[3])
            gain_value = shape_to_gain.get(key, float("-inf"))
            if gain_value > 0:
                gain_shapes.append(shape.copy())
                gain_count += 1
            else:
                lose_shapes.append(shape.copy())
                lose_count += 1

        gain_blocks.append({"op": block_op, "shapes": gain_shapes, "shape_desc": block_shape_desc})
        lose_blocks.append({"op": block_op, "shapes": lose_shapes, "shape_desc": block_shape_desc})

    write_model_shapes_yaml(gain_yaml_path, gain_blocks)
    write_model_shapes_yaml(lose_yaml_path, lose_blocks)
    return gain_count, lose_count


def append_table(lines: list[str], title: str, rows: list[list[str]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Shape (B, M, N, K) | Count | Default Configuration |  |  | Expand Configuration |  |  | Speedup Gain |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append("|  |  | Torch Latency (ms) | Gems Latency (ms) | Gems Speedup | Torch Latency (ms) | Gems Latency (ms) | Gems Speedup | Expand vs Default |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def build_table_rows(
    sections: dict[str, dict[str, tuple[str, str, str]]],
    shape_order: list[str],
    count_map: dict[tuple[int, int, int, int], int],
) -> tuple[list[list[str]], list[list[str]]]:
    table_rows: list[list[str]] = []

    for shape in shape_order:
        default_metrics = sections["default"].get(shape, ("-", "-", "-"))
        expand_metrics = sections["expand"].get(shape, ("-", "-", "-"))

        shape_bmnk, count = convert_shape_to_bmnk_and_count(shape, count_map)
        gain = calc_gain_percent(default_metrics[2], expand_metrics[2])

        row = [
            shape_bmnk,
            count,
            default_metrics[0],
            default_metrics[1],
            default_metrics[2],
            expand_metrics[0],
            expand_metrics[1],
            expand_metrics[2],
            gain,
        ]
        table_rows.append(row)

    rows_by_gain = sorted(table_rows, key=lambda row: parse_gain_value(row[8]), reverse=True)
    rows_by_count = sorted(table_rows, key=lambda row: parse_count_value(row[1]), reverse=True)
    return rows_by_gain, rows_by_count


def write_excel_report(xlsx_path: Path, rows_by_gain: list[list[str]], rows_by_count: list[list[str]]) -> None:
    def style_sheet(ws) -> None:
        header_font = Font(bold=True)
        center_alignment = Alignment(horizontal="center", vertical="center")

        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.alignment = center_alignment

        for row_idx in (1, 2):
            for cell in ws[row_idx]:
                cell.font = header_font

        for col_idx in range(1, ws.max_column + 1):
            column_letter = get_column_letter(col_idx)
            max_len = 0
            for cell in ws[column_letter]:
                value = "" if cell.value is None else str(cell.value)
                if len(value) > max_len:
                    max_len = len(value)

            adjusted_width = max(12, min(max_len + 2, 60))
            ws.column_dimensions[column_letter].width = adjusted_width

    def write_sheet(ws, rows: list[list[str]]) -> None:
        ws.append([
            "Shape (B, M, N, K)",
            "Count",
            "Default Configuration",
            "",
            "",
            "Expand Configuration",
            "",
            "",
            "Speedup Gain",
        ])
        ws.append([
            "",
            "",
            "Torch Latency (ms)",
            "Gems Latency (ms)",
            "Gems Speedup",
            "Torch Latency (ms)",
            "Gems Latency (ms)",
            "Gems Speedup",
            "Expand vs Default",
        ])

        ws.merge_cells("A1:A2")
        ws.merge_cells("B1:B2")
        ws.merge_cells("C1:E1")
        ws.merge_cells("F1:H1")
        ws.merge_cells("I1:I2")

        for row in rows:
            ws.append(row)

        style_sheet(ws)

    workbook = Workbook()

    ws_gain = workbook.active
    ws_gain.title = "Sorted by Speedup Gain"
    write_sheet(ws_gain, rows_by_gain)

    ws_count = workbook.create_sheet(title="Sorted by Count")
    write_sheet(ws_count, rows_by_count)

    workbook.save(xlsx_path)


def parse_log(log_path: Path) -> tuple[dict[str, dict[str, tuple[str, str, str]]], list[str]]:
    sections: dict[str, dict[str, tuple[str, str, str]]] = {"default": {}, "expand": {}}
    shape_order: list[str] = []
    current_section: str | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if DEFAULT_INFO_RE.search(line):
                current_section = "default"
                continue
            if EXPAND_INFO_RE.search(line):
                current_section = "expand"
                continue

            if current_section is None:
                continue

            match = DATA_LINE_RE.match(line)
            if not match:
                continue

            shape = match.group("shape")
            metrics = (match.group("torch"), match.group("gems"), match.group("speedup"))
            sections[current_section][shape] = metrics

            if shape not in shape_order:
                shape_order.append(shape)

    return sections, shape_order


def build_report_content(
    model: str,
    op: str,
    rows_by_gain: list[list[str]],
    rows_by_count: list[list[str]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Performance Summary: {model} / {op}")
    lines.append("")
    lines.append(f"- Source log: `log/flagtune/{model}/{op}/pretune/pretune.log`")
    lines.append(f"- Count reference: `flagtune/shape-config/{model}_count.yaml`")
    lines.append(f"- Rows: {len(rows_by_gain)}")
    lines.append("")

    append_table(lines, "Sorted by Speedup Gain", rows_by_gain)
    append_table(lines, "Sorted by Count", rows_by_count)

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown summary from pretune log",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="qwen3.5", help="Model name")
    parser.add_argument("--op", default="mm", help="Operator name")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    log_path = repo_root / "log" / "flagtune" / args.model / args.op / "pretune" / "pretune.log"
    count_yaml_path = repo_root / "flagtune" / "shape-config" / f"{args.model}_count.yaml"
    model_yaml_path = repo_root / "flagtune" / "shape-config" / f"{args.model}.yaml"
    gain_yaml_path = repo_root / "flagtune" / "shape-config" / f"{args.model}_gain.yaml"
    lose_yaml_path = repo_root / "flagtune" / "shape-config" / f"{args.model}_lose.yaml"
    output_path = repo_root / "flagtune" / "reports" / f"{args.model}_{args.op}.md"
    output_xlsx_path = repo_root / "flagtune" / "reports" / f"{args.model}_{args.op}.xlsx"

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    if not count_yaml_path.exists():
        raise FileNotFoundError(f"Count yaml not found: {count_yaml_path}")
    if not model_yaml_path.exists():
        raise FileNotFoundError(f"Model yaml not found: {model_yaml_path}")

    sections, shape_order = parse_log(log_path)
    count_map = parse_count_map(count_yaml_path, args.op)
    if not shape_order:
        raise ValueError("No performance rows were parsed from the log file")

    rows_by_gain, rows_by_count = build_table_rows(sections, shape_order, count_map)

    report = build_report_content(args.model, args.op, rows_by_gain, rows_by_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    write_excel_report(output_xlsx_path, rows_by_gain, rows_by_count)
    gain_count, lose_count = split_and_write_gain_lose_yaml(
        model_yaml_path,
        gain_yaml_path,
        lose_yaml_path,
        args.op,
        rows_by_gain,
    )

    print(f"Generated report: {output_path}")
    print(f"Generated report: {output_xlsx_path}")
    print(f"Generated gain yaml: {gain_yaml_path} (shapes={gain_count})")
    print(f"Generated lose yaml: {lose_yaml_path} (shapes={lose_count})")
    print(f"Rows: {len(rows_by_gain)}")


if __name__ == "__main__":
    main()
