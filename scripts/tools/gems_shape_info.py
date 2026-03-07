#!/usr/bin/env python3
"""Extract unique operator shape info with counts from FlagGems logs.

Input:
- gems-config/gems-all.txt (or specified input file)

Output:
- reports/{model_name}/gems_shape_info.txt (or specified output file)
"""

import argparse
from pathlib import Path
import re

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


def extract_shape_info(input_path: Path, output_path: Path):
    records = {}

    with input_path.open("r", encoding="utf-8", errors="ignore") as infile:
        for raw_line in infile:
            parsed = parse_line(raw_line)
            if not parsed:
                continue

            key = parsed
            records[key] = records.get(key, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(
        records.items(),
        key=lambda item: (item[0][0], -item[1], item[0][1]),
    )
    with output_path.open("w", encoding="utf-8") as outfile:
        for (op_name, shape_info), count in sorted_items:
            outfile.write(f"{op_name}, [shape info]: {shape_info}, [count]: {count}\n")

    print(f"Done. extracted={len(records)} -> {output_path}")


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
    args = parser.parse_args()

    project_root = get_project_root()

    # Load config for default paths
    config = load_tool_config()
    model_name = config.get('paths', {}).get('model_name', 'default')
    gems_config_dir = config.get('paths', {}).get('gems_config_dir', 'gems-config')
    reports_dir = config.get('paths', {}).get('reports_dir', 'reports')

    # Determine input path
    if args.input:
        input_path = (project_root / args.input).resolve()
    else:
        input_path = (project_root / gems_config_dir / "gems-all.txt").resolve()

    # Determine output path
    if args.output:
        output_path = (project_root / args.output).resolve()
    else:
        output_path = (project_root / reports_dir / "gems_shape_info.txt").resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    extract_shape_info(input_path, output_path)


if __name__ == "__main__":
    main()
