#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

DATA_LINE_RE = re.compile(
    r"^(?P<status>\S+)\s+"
    r"(?P<torch>[\d.eE+-]+)\s+"
    r"(?P<gems>[\d.eE+-]+)\s+"
    r"(?P<speedup>[\d.eE+-]+)\s+"
    r"(?P<tflops>[\d.eE+-]+)\s+"
    r"(?P<shape>\[.*\])\s*$"
)
TORCH_SIZE_RE = re.compile(r"torch\.Size\(\[(?P<dims>[^\]]+)\]\)")

TABLE_HEADERS = [
    "Shape (B, M, N, K)",
    "Count",
    "Torch Latency with left configuration (ms)",
    "Gems Latency with left configuration (ms)",
    "Gems Speedup with left configuration",
    "Torch Latency with right configuration (ms)",
    "Gems Latency with right configuration (ms)",
    "Gems Speedup with right configuration",
    "Speedup Gain",
]

COMPARISON_TABLE_TITLE = "Sorted by Speedup Gain"
SINGLE_CONFIG_TABLE_TITLE = "Performance Summary"


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_shape_source_op(op: str) -> str:
    if op.startswith("w8a8_block_fp8_matmul"):
        return "w8a8_block_fp8_matmul"
    return op


def parse_count_map(count_yaml_path: Path, target_op: str) -> dict[tuple[int, int, int, int], int]:
    count_map: dict[tuple[int, int, int, int], int] = {}
    current_op: str | None = None
    in_shapes = False
    current_shape: list[int] | None = None
    target_shape_op = resolve_shape_source_op(target_op)

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
    shape_text: str,
    count_map: dict[tuple[int, int, int, int], int],
    default_count: int | str = "-",
) -> tuple[str, str]:
    bmnk = infer_bmnk_from_shape_text(shape_text)
    if bmnk is None:
        return shape_text, str(default_count)

    count = count_map.get(bmnk, default_count)
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


def parse_speedup_value(speedup_text: str) -> float:
    try:
        return float(speedup_text)
    except ValueError:
        return float("-inf")


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
    include_right_comparison: bool = True,
) -> tuple[int, int]:
    source_blocks = parse_model_yaml(model_yaml_path)
    shape_to_gain: dict[tuple[int, int, int, int], float] = {}
    target_shape_op = resolve_shape_source_op(op)

    for row in rows_by_gain:
        shape_text = row[0]
        score_text = row[8] if include_right_comparison else row[4]
        parts = [part.strip() for part in shape_text.split(",")]
        if len(parts) != 4:
            continue
        try:
            key = tuple(int(part) for part in parts)
        except ValueError:
            continue
        if include_right_comparison:
            shape_to_gain[key] = parse_gain_value(score_text)
        else:
            try:
                shape_to_gain[key] = float(score_text)
            except ValueError:
                shape_to_gain[key] = float("-inf")

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
            if include_right_comparison:
                is_gain_shape = gain_value > 0
            else:
                is_gain_shape = gain_value >= 1.0

            if is_gain_shape:
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


def append_table(
    lines: list[str],
    title: str,
    rows: list[list[str]],
    left_report_label: str,
    right_report_label: str,
    include_right_comparison: bool,
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if include_right_comparison:
        lines.append(
            f"| Shape (B, M, N, K) | Count | {left_report_label} |  |  | "
            f"{right_report_label} |  |  | Speedup Gain |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append(
            f"|  |  | Torch Latency (ms) | Gems Latency (ms) | Gems Speedup | "
            f"Torch Latency (ms) | Gems Latency (ms) | Gems Speedup | {right_report_label} vs {left_report_label} |"
        )
    else:
        lines.append(f"| Shape (B, M, N, K) | Count | {left_report_label} |  |  |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append("|  |  | Torch Latency (ms) | Gems Latency (ms) | Gems Speedup |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def build_table_rows(
    sections: dict[str, dict[str, tuple[str, str, str]]],
    shape_order: list[str],
    count_map: dict[tuple[int, int, int, int], int],
    default_count: int | str = "-",
    include_right_comparison: bool = True,
) -> tuple[list[list[str]], list[list[str]]]:
    table_rows: list[list[str]] = []

    for shape in shape_order:
        left_metrics = sections["left"].get(shape, ("-", "-", "-"))
        shape_bmnk, count = convert_shape_to_bmnk_and_count(shape, count_map, default_count)
        row = [shape_bmnk, count, left_metrics[0], left_metrics[1], left_metrics[2]]

        if include_right_comparison:
            right_metrics = sections["right"].get(shape, ("-", "-", "-"))
            gain = calc_gain_percent(left_metrics[2], right_metrics[2])
            row.extend([right_metrics[0], right_metrics[1], right_metrics[2], gain])

        table_rows.append(row)

    if include_right_comparison:
        rows_by_gain = sorted(table_rows, key=lambda row: parse_gain_value(row[8]), reverse=True)
    else:
        rows_by_gain = sorted(table_rows, key=lambda row: parse_speedup_value(row[4]), reverse=True)
    rows_by_count = sorted(table_rows, key=lambda row: parse_count_value(row[1]), reverse=True)
    return rows_by_gain, rows_by_count


def write_excel_report(
    xlsx_path: Path,
    rows_by_gain: list[list[str]],
    rows_by_count: list[list[str]],
    left_report_label: str,
    right_report_label: str,
    include_right_comparison: bool = True,
    include_count_sheet: bool = True,
) -> None:
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
        if include_right_comparison:
            ws.append([
                "Shape (B, M, N, K)",
                "Count",
                left_report_label,
                "",
                "",
                right_report_label,
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
                f"{right_report_label} vs {left_report_label}",
            ])

            ws.merge_cells("A1:A2")
            ws.merge_cells("B1:B2")
            ws.merge_cells("C1:E1")
            ws.merge_cells("F1:H1")
            ws.merge_cells("I1:I2")
        else:
            ws.append([
                "Shape (B, M, N, K)",
                "Count",
                left_report_label,
                "",
                "",
            ])
            ws.append([
                "",
                "",
                "Torch Latency (ms)",
                "Gems Latency (ms)",
                "Gems Speedup",
            ])

            ws.merge_cells("A1:A2")
            ws.merge_cells("B1:B2")
            ws.merge_cells("C1:E1")

        for row in rows:
            ws.append(row)

        style_sheet(ws)

    workbook = Workbook()

    ws_gain = workbook.active
    ws_gain.title = COMPARISON_TABLE_TITLE if include_right_comparison else SINGLE_CONFIG_TABLE_TITLE
    write_sheet(ws_gain, rows_by_gain)

    if include_count_sheet:
        ws_count = workbook.create_sheet(title="Sorted by Count")
        write_sheet(ws_count, rows_by_count)

    workbook.save(xlsx_path)


def parse_log(
    log_path: Path,
    left_stage_label: str = "default",
    right_stage_label: str = "expand",
) -> tuple[dict[str, dict[str, tuple[str, str, str]]], list[str]]:
    left_info_re = re.compile(rf"\[INFO\].*with {re.escape(left_stage_label)} configuration\.")
    right_info_re = re.compile(rf"\[INFO\].*with {re.escape(right_stage_label)} configuration\.")
    sections: dict[str, dict[str, tuple[str, str, str]]] = {"left": {}, "right": {}}
    shape_order: list[str] = []
    current_section: str | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if left_info_re.search(line):
                current_section = "left"
                continue
            if right_info_re.search(line):
                current_section = "right"
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
    count_yaml_exists: bool,
    left_report_label: str,
    right_report_label: str,
    include_right_comparison: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# Performance Summary: {model} / {op}")
    lines.append("")
    lines.append(f"- Source log: `log/flagtune/{model}/{op}/pretune/pretune.log`")
    if include_right_comparison:
        lines.append(f"- Compare: `{left_report_label}` vs `{right_report_label}`")
    else:
        lines.append(f"- Configuration: `{left_report_label}`")
    if count_yaml_exists:
        lines.append(f"- Count reference: `FlagTune/shape-config/{model}_count.yaml`")
    else:
        lines.append("- Count reference: missing, all counts fallback to `1`")
    lines.append(f"- Rows: {len(rows_by_gain)}")
    lines.append("")

    primary_title = COMPARISON_TABLE_TITLE if include_right_comparison else SINGLE_CONFIG_TABLE_TITLE
    append_table(
        lines,
        primary_title,
        rows_by_gain,
        left_report_label,
        right_report_label,
        include_right_comparison,
    )
    if count_yaml_exists:
        append_table(
            lines,
            "Sorted by Count",
            rows_by_count,
            left_report_label,
            right_report_label,
            include_right_comparison,
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown summary from pretune log",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="qwen3.5", help="Model name")
    parser.add_argument("--op", default="mm", help="Operator name")
    parser.add_argument("--output-suffix", default="", help="Suffix appended to report filenames, for example _master")
    parser.add_argument("--left-stage-label", default="default", help="Log stage name for the left-side comparison")
    parser.add_argument("--right-stage-label", default="expand", help="Log stage name for the right-side comparison")
    parser.add_argument("--left-report-label", default="Default Configuration", help="Report column title for the left-side comparison")
    parser.add_argument("--right-report-label", default="Expand Configuration", help="Report column title for the right-side comparison")
    parser.add_argument(
        "--include-right-comparison",
        type=parse_bool_arg,
        default=True,
        help="Whether to include the right-side comparison columns in the report",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    flagtune_dir = Path(__file__).resolve().parent.parent
    log_path = repo_root / "log" / "flagtune" / args.model / args.op / "pretune" / "pretune.log"
    count_yaml_path = flagtune_dir / "shape-config" / f"{args.model}_count.yaml"
    model_yaml_path = flagtune_dir / "shape-config" / f"{args.model}.yaml"
    gain_yaml_path = flagtune_dir / "shape-config" / f"{args.model}_gain.yaml"
    lose_yaml_path = flagtune_dir / "shape-config" / f"{args.model}_lose.yaml"
    output_path = flagtune_dir / "reports" / f"{args.model}_{args.op}{args.output_suffix}.md"
    output_xlsx_path = flagtune_dir / "reports" / f"{args.model}_{args.op}{args.output_suffix}.xlsx"

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    if not model_yaml_path.exists():
        raise FileNotFoundError(f"Model yaml not found: {model_yaml_path}")

    sections, shape_order = parse_log(log_path, args.left_stage_label, args.right_stage_label)
    count_yaml_exists = count_yaml_path.exists()
    count_map = parse_count_map(count_yaml_path, args.op) if count_yaml_exists else {}
    if not shape_order:
        raise ValueError("No performance rows were parsed from the log file")

    default_count = 1 if not count_yaml_exists else "-"
    rows_by_gain, rows_by_count = build_table_rows(
        sections,
        shape_order,
        count_map,
        default_count,
        include_right_comparison=args.include_right_comparison,
    )

    report = build_report_content(
        args.model,
        args.op,
        rows_by_gain,
        rows_by_count,
        count_yaml_exists,
        args.left_report_label,
        args.right_report_label,
        args.include_right_comparison,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    write_excel_report(
        output_xlsx_path,
        rows_by_gain,
        rows_by_count,
        args.left_report_label,
        args.right_report_label,
        include_right_comparison=args.include_right_comparison,
        include_count_sheet=count_yaml_exists,
    )

    print(f"Generated report: {output_path}")
    print(f"Generated report: {output_xlsx_path}")
    gain_count, lose_count = split_and_write_gain_lose_yaml(
        model_yaml_path,
        gain_yaml_path,
        lose_yaml_path,
        args.op,
        rows_by_gain,
        include_right_comparison=args.include_right_comparison,
    )
    print(f"Generated gain yaml: {gain_yaml_path} (shapes={gain_count})")
    print(f"Generated lose yaml: {lose_yaml_path} (shapes={lose_count})")
    print(f"Rows: {len(rows_by_gain)}")


if __name__ == "__main__":
    main()
