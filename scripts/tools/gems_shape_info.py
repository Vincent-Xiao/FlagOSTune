#!/usr/bin/env python3
"""Extract operator shape statistics from FlagGems logs.

Supports two modes:
1) Single-file mode (legacy): read one input log and write one stats file.
2) Marker split mode: read raw gems-all.txt + marker.txt, then output per-scenario
    stats files where each scenario includes:
    - common prefix: lines 1..common (common = first marker value)
    - scenario split range based on marker boundaries.
"""

import argparse
from pathlib import Path
import re
from typing import Dict, Tuple

import yaml


def get_project_root() -> Path:
    """Get project root directory"""
    script_file = Path(__file__).resolve()
    tools_dir = script_file.parent
    return tools_dir.parent.parent


def get_tools_dir() -> Path:
    """Get tools directory"""
    return Path(__file__).resolve().parent


def load_tool_config() -> dict:
    """Load tool_config.yaml"""
    tool_config = get_tools_dir() / "tool_config.yaml"

    if not tool_config.exists():
        return {}

    with open(tool_config) as f:
        return yaml.safe_load(f) or {}


def normalize_text(text: str) -> str:
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    return re.sub(r"\s+", " ", printable).strip()


def parse_line(line: str):
    line = line.strip()
    if "[shape info]:" not in line:
        return None

    op_match = re.search(r"^\[DEBUG\]\s+([^:]+):", line)
    if not op_match:
        return None

    op_name = normalize_text(op_match.group(1))
    if "DEBUG" in op_name or "[" in op_name or "]" in op_name:
        return None
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", op_name):
        return None
    if not op_name.startswith("flag_gems.ops."):
        return None

    shape_part = line.split("[shape info]:", 1)[1].strip()

    left = shape_part.find("[")
    right = shape_part.find("]", left + 1) if left != -1 else -1
    if left == -1 or right == -1:
        return None

    shape_info = shape_part[left : right + 1]
    shape_info = normalize_text(shape_info)
    if "DEBUG" in shape_info:
        return None
    if not re.match(r"^\[\s*[-\d,\s]+\]$", shape_info):
        return None

    tail = shape_part[right + 1 :].lstrip()
    if tail.startswith("("):
        end_paren = tail.find(")")
        if end_paren != -1:
            paren = tail[: end_paren + 1]
            if "[" not in paren and "DEBUG" not in paren:
                paren = normalize_text(paren)
                if re.match(r"^\([A-Za-z0-9_,\s]+\)$", paren):
                    shape_info += paren

    if not shape_info:
        return None

    return op_name, shape_info


def parse_marker_line(line: str) -> tuple[str, int] | None:
    text = line.strip()
    if not text:
        return None

    sep = "：" if "：" in text else ":"
    if sep not in text:
        raise ValueError(f"Invalid marker line (missing separator): {line.rstrip()}")

    name, num_text = text.split(sep, 1)
    name = name.strip()
    num_text = num_text.strip()
    if not name:
        raise ValueError(f"Invalid marker line (empty scenario name): {line.rstrip()}")

    try:
        start_minus_1 = int(num_text)
    except ValueError as exc:
        raise ValueError(f"Invalid marker line (non-integer index): {line.rstrip()}") from exc

    if start_minus_1 < 0:
        raise ValueError(f"Invalid marker line (negative index): {line.rstrip()}")

    return name, start_minus_1


def load_markers(marker_path: Path) -> list[tuple[str, int]]:
    markers: list[tuple[str, int]] = []
    with marker_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parsed = parse_marker_line(raw)
            if parsed is None:
                continue
            markers.append(parsed)

    if not markers:
        raise ValueError(f"No valid marker entries found in: {marker_path}")

    names = [m[0] for m in markers]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate scenario names found in marker file")

    starts = [m[1] for m in markers]
    if len(set(starts)) != len(starts):
        raise ValueError("Duplicate start indices found in marker file")

    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            raise ValueError("Marker start indices must be strictly increasing by file order")

    return markers


def write_records(output_path: Path, records: Dict[Tuple[str, str], int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(
        records.items(),
        key=lambda item: (item[0][0], -item[1], item[0][1]),
    )
    with output_path.open("w", encoding="utf-8") as outfile:
        for (op_name, shape_info), count in sorted_items:
            outfile.write(f"{op_name}, [shape info]: {shape_info}, [count]: {count}\n")


def merge_records(base: Dict[Tuple[str, str], int], extra: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], int]:
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def extract_shape_info(input_path: Path, output_path: Path):
    records: Dict[Tuple[str, str], int] = {}

    with input_path.open("r", encoding="utf-8", errors="ignore") as infile:
        for raw_line in infile:
            parsed = parse_line(raw_line)
            if not parsed:
                continue

            key = parsed
            records[key] = records.get(key, 0) + 1

    write_records(output_path, records)

    print(f"Done. extracted={len(records)} -> {output_path}")


def extract_shape_info_by_marker(input_path: Path, marker_path: Path, output_dir: Path) -> None:
    markers = load_markers(marker_path)
    common_lines = markers[0][1]

    common_records: Dict[Tuple[str, str], int] = {}
    split_records: Dict[str, Dict[Tuple[str, str], int]] = {name: {} for name, _ in markers}
    split_line_counts = {name: 0 for name, _ in markers}

    total_lines = 0
    current_idx = -1
    current_name = None

    with input_path.open("r", encoding="utf-8", errors="ignore") as infile:
        for line_no, raw_line in enumerate(infile):
            total_lines += 1
            parsed = parse_line(raw_line)

            if line_no < common_lines:
                if parsed:
                    common_records[parsed] = common_records.get(parsed, 0) + 1
                continue

            while current_idx + 1 < len(markers) and line_no >= markers[current_idx + 1][1]:
                current_idx += 1
                current_name = markers[current_idx][0]

            if current_name is None:
                continue

            split_line_counts[current_name] += 1
            if parsed:
                scenario_records = split_records[current_name]
                scenario_records[parsed] = scenario_records.get(parsed, 0) + 1

    if total_lines < common_lines:
        raise ValueError(
            f"Input file has fewer lines than common prefix size: need {common_lines}, got {total_lines}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Split+stat done: {input_path}")
    print(f"Marker file: {marker_path}")
    print(f"Output dir: {output_dir}")

    for name, _ in markers:
        merged = merge_records(common_records, split_records[name])
        out_path = output_dir / f"{name}.txt"
        write_records(out_path, merged)

        split_lines = split_line_counts[name]
        total_out_lines = common_lines + split_lines
        print(
            f"{name}: common_lines={common_lines}, split_lines={split_lines}, "
            f"total_lines={total_out_lines}, unique_shapes={len(merged)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Extract unique operator shape info with counts")
    parser.add_argument(
        "--input",
        default=None,
        help="Input log file containing [shape info] records (default: gems-config/gems-all.txt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output text file (default: reports/{model_name}/gems_shape_info.txt)",
    )
    parser.add_argument(
        "--marker",
        default=None,
        help="Marker file for split mode (default: results/{model_name}/gems-config-shape/marker.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for split stats mode (default: reports/{model_name}/shape)",
    )
    args = parser.parse_args()

    project_root = get_project_root()

    # Load config for default paths
    config = load_tool_config()
    model_name = config.get('paths', {}).get('model_name', 'default')
    gems_config_dir = config.get('paths', {}).get('gems_config_dir', 'gems-config')
    reports_dir = config.get('paths', {}).get('reports_dir', 'reports')
    results_root = config.get('paths', {}).get('results', 'results')

    default_shape_input = f"{results_root}/{model_name}/gems-config-shape/gems-all.txt"
    default_shape_marker = f"{results_root}/{model_name}/gems-config-shape/marker.txt"
    default_shape_output_dir = f"reports/{model_name}/shape"

    # Determine input path
    if args.input:
        input_path = (project_root / args.input).resolve()
    else:
        shape_input_path = (project_root / default_shape_input).resolve()
        if shape_input_path.exists():
            input_path = shape_input_path
        else:
            input_path = (project_root / gems_config_dir / "gems-all.txt").resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    default_marker_path = (project_root / default_shape_marker).resolve()
    marker_path = (project_root / args.marker).resolve() if args.marker else default_marker_path
    use_split_mode = args.output_dir is not None or args.marker is not None or marker_path.exists()

    if use_split_mode and not args.output:
        output_dir = (
            (project_root / args.output_dir).resolve()
            if args.output_dir
            else (project_root / default_shape_output_dir).resolve()
        )

        if not marker_path.exists():
            raise FileNotFoundError(f"Marker file not found: {marker_path}")

        extract_shape_info_by_marker(input_path, marker_path, output_dir)
        return

    # Legacy single-file mode
    if args.output:
        output_path = (project_root / args.output).resolve()
    else:
        output_path = (project_root / reports_dir / "gems_shape_info.txt").resolve()
    extract_shape_info(input_path, output_path)


if __name__ == "__main__":
    main()
