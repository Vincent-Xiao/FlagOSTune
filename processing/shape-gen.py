#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

NumericShape = tuple[int, ...]
MulShape = tuple[str, NumericShape] | tuple[str, NumericShape, NumericShape]
ParsedShape = str | MulShape
ShapeKey = NumericShape | MulShape
GroupedShape = list[object]

OP_SHAPE_DESC = {
    "mul": "kind, lhs_shape, rhs_shape",
    "sparse_attention": "B, M, KV_LEN, TOPK, H, D",
}
OP_EXPECTED_DIMS = {
    "sparse_attention": {6},
    "w8a8_block_fp8_matmul": {4},
}


def normalize_text(text: str) -> str:
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    return re.sub(r"\s+", " ", printable).strip()


def parse_mul_dims(raw_dims: str) -> NumericShape:
    tokens = [token.strip() for token in raw_dims.split(",") if token.strip()]
    try:
        dims = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError(f"Invalid mul dimensions: ({raw_dims})") from error
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Mul dimensions must be positive: {dims}")
    return dims


def shapes_are_broadcastable(lhs: NumericShape, rhs: NumericShape) -> bool:
    for lhs_dim, rhs_dim in zip(reversed(lhs), reversed(rhs)):
        if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:
            return False
    return True


def parse_mul_shape_info(shape_part: str) -> MulShape:
    match = re.search(
        r"A=\((?P<lhs>[^)]*)\),\s*"
        r"B=(?:(?P<scalar>scalar)|\((?P<rhs>[^)]*)\))",
        shape_part,
    )
    if match is None:
        raise ValueError(f"Invalid mul shape info: {shape_part}")

    lhs = parse_mul_dims(match.group("lhs"))
    if match.group("scalar"):
        return "scalar", lhs

    rhs = parse_mul_dims(match.group("rhs"))
    if lhs == rhs:
        return "same", lhs
    if not shapes_are_broadcastable(lhs, rhs):
        raise ValueError(f"Mul shapes cannot be broadcast: {lhs}, {rhs}")
    return "broadcast", lhs, rhs


def parse_shape_line(line: str) -> tuple[str, ParsedShape, int] | None:
    line = line.strip()
    if "[shape info]:" not in line:
        return None

    op_part = line.split("[shape info]:", 1)[0].strip()
    op_part = op_part.rstrip(", ")

    op_name = normalize_text(op_part)
    if "[" in op_name or "]" in op_name:
        return None
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", op_name):
        return None
    if not op_name.startswith("flag_gems.ops."):
        return None

    shape_part = line.split("[shape info]:", 1)[1].strip()
    count = 1
    count_match = re.search(r"\[count\]\s*:\s*(\d+)", line)
    if count_match:
        count = int(count_match.group(1))

    if normalize_op_key(op_name) == "mul":
        return op_name, parse_mul_shape_info(shape_part), count

    left = shape_part.find("[")
    right = shape_part.find("]", left + 1) if left != -1 else -1
    if left == -1 or right == -1:
        return None

    shape_info = normalize_text(shape_part[left : right + 1])
    if not re.match(r"^\[\s*[-\d,\s]+\]$", shape_info):
        return None

    return op_name, shape_info, count


def normalize_op_key(full_op: str) -> str:
    op = full_op.strip()
    marker = ".ops."
    if marker in op:
        op = op.split(marker, 1)[1]

    parts = op.split(".")
    if len(parts) >= 2:
        module_name = parts[0]
        func_name = parts[-1]
        if module_name == "mm" and func_name == "general_mm":
            return "mm"
        return module_name

    return op


def parse_shape(raw_shape: str) -> list[int]:
    shape_inner = raw_shape.strip()[1:-1]
    dims: list[int] = []
    for item in shape_inner.split(","):
        token = item.strip()
        if token == "-":
            dims.append(1)
        else:
            dims.append(int(token))
    return dims


def normalize_shape_for_op(full_op: str, shape: list[int]) -> list[int]:
    op = full_op.strip()

    if op.endswith(".gemv_mm") or op.endswith(".general_mm"):
        # gemv_mm logs use the compact GEMV layout (M, K, N), while the merged
        # YAML config for mm expects the unified matmul layout (B, M, N, K).
        # Some general_mm records can also be emitted in the same compact form.
        # When gemv_mm/general_mm are both folded into `mm`, pad batch=1 and
        # reorder dimensions here so downstream consumers always see B,M,N,K.
        if len(shape) != 3:
            return shape
        m, k, n = shape
        return [1, m, n, k]

    return shape


def get_expected_dims_for_op(op_key: str) -> set[int] | None:
    expected_dims = OP_EXPECTED_DIMS.get(op_key)
    if expected_dims is not None:
        if isinstance(expected_dims, int):
            return {expected_dims}
        return set(expected_dims)

    if "mm" in op_key:
        return {4}

    return None


def collect_sorted_records(input_path: Path) -> list[tuple[str, ParsedShape, int]]:
    records: dict[tuple[str, ParsedShape], int] = {}

    with input_path.open("r", encoding="utf-8", errors="ignore") as infile:
        for raw_line in infile:
            parsed = parse_shape_line(raw_line)
            if not parsed:
                continue
            op_name, shape, count = parsed
            key = (op_name, shape)
            records[key] = records.get(key, 0) + count

    sorted_items = sorted(
        records.items(),
        key=lambda item: (item[0][0], -item[1], item[0][1]),
    )
    return [(op, shape, count) for (op, shape), count in sorted_items]


def shape_key_to_list(shape_key: ShapeKey) -> GroupedShape:
    return [
        list(item) if isinstance(item, tuple) else item
        for item in shape_key
    ]


def build_grouped_shapes(
    records: list[tuple[str, ParsedShape, int]], include_count: bool = True
) -> OrderedDict[str, list[GroupedShape]]:
    grouped_counts: OrderedDict[str, OrderedDict[ShapeKey, int]] = OrderedDict()

    for full_op, raw_shape, count in records:
        op_key = normalize_op_key(full_op)
        if op_key == "mul":
            if isinstance(raw_shape, str):
                raise ValueError(f"Invalid parsed mul shape: {raw_shape}")
            shape_key: ShapeKey = raw_shape
        else:
            if not isinstance(raw_shape, str):
                raise ValueError(f"Invalid numeric shape: {raw_shape}")
            # Normalize operator-specific log layouts before dedup/merge. This
            # is especially important for gemv_mm, because it is merged into `mm`.
            shape_key = tuple(
                normalize_shape_for_op(full_op, parse_shape(raw_shape))
            )
            expected_dims = get_expected_dims_for_op(op_key)
            if expected_dims is not None and len(shape_key) not in expected_dims:
                continue

        if op_key not in grouped_counts:
            grouped_counts[op_key] = OrderedDict()

        if shape_key not in grouped_counts[op_key]:
            grouped_counts[op_key][shape_key] = 0

        grouped_counts[op_key][shape_key] += count

    grouped: OrderedDict[str, list[GroupedShape]] = OrderedDict()
    for op_key, shape_to_count in grouped_counts.items():
        if include_count:
            grouped[op_key] = [
                shape_key_to_list(shape_key) + [count]
                for shape_key, count in shape_to_count.items()
            ]
        else:
            grouped[op_key] = [
                shape_key_to_list(shape_key)
                for shape_key in shape_to_count
            ]

    return grouped


def filter_by_op(
    grouped_shapes: OrderedDict[str, list[GroupedShape]], op_name: str | None
) -> OrderedDict[str, list[GroupedShape]]:
    if not op_name:
        return grouped_shapes

    if op_name not in grouped_shapes:
        raise ValueError(f"Operator '{op_name}' not found in input file")

    filtered: OrderedDict[str, list[GroupedShape]] = OrderedDict()
    filtered[op_name] = grouped_shapes[op_name]
    return filtered


def filter_by_n_eq_1(
    grouped_shapes: OrderedDict[str, list[GroupedShape]], *, is_n_eq_1: bool
) -> OrderedDict[str, list[GroupedShape]]:
    filtered: OrderedDict[str, list[GroupedShape]] = OrderedDict()

    for op_name, shapes in grouped_shapes.items():
        matched_shapes = []
        for shape in shapes:
            if len(shape) < 3:
                continue
            if (shape[2] == 1) == is_n_eq_1:
                matched_shapes.append(shape)

        if matched_shapes:
            filtered[op_name] = matched_shapes

    return filtered


def dump_shapes_yaml(
    grouped_shapes: OrderedDict[str, list[GroupedShape]],
    output_path: Path,
    include_count: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for op_idx, (op_name, shapes) in enumerate(grouped_shapes.items()):
            shape_desc = OP_SHAPE_DESC.get(op_name, "B, M, N, K")
            file.write(f"{op_name}:\n")
            file.write("  shapes:\n")
            for shape in shapes:
                file.write("  - - ")
                file.write(f"{shape[0]}\n")
                for dim in shape[1:]:
                    file.write(f"    - {dim}\n")
            if include_count:
                file.write(f"  shape_desc: {shape_desc}, Count\n")
            else:
                file.write(f"  shape_desc: {shape_desc}\n")
            if op_idx != len(grouped_shapes) - 1:
                file.write("\n")


def print_operator_summary(
    grouped_shapes: OrderedDict[str, list[GroupedShape]],
) -> None:
    print("Operator shape counts:")
    for op_name, shapes in grouped_shapes.items():
        print(f"  - {op_name}: {len(shapes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YAML shape config from gems debug logs")
    parser.add_argument(
        "--model",
        default="qwen3.5",
        help="Model name used to resolve input/output files under FlagTune/shape-config",
    )
    parser.add_argument(
        "--op",
        default=None,
        help="Optional operator name filter (e.g. mm)",
    )
    parser.add_argument(
        "--split1",
        action="store_true",
        help="Also generate split YAML files for shapes with N=1 and N!=1",
    )
    args = parser.parse_args()

    flagtune_dir = Path(__file__).resolve().parent.parent
    shape_config_dir = flagtune_dir / "shape-config"
    input_path = shape_config_dir / f"{args.model}.txt"
    if args.op:
        output_path = shape_config_dir / f"{args.model}_{args.op}.yaml"
        output_count_path = shape_config_dir / f"{args.model}_{args.op}_count.yaml"
    else:
        output_path = shape_config_dir / f"{args.model}.yaml"
        output_count_path = shape_config_dir / f"{args.model}_count.yaml"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = collect_sorted_records(input_path)
    if not records:
        raise ValueError(
            f"No valid shape records parsed from {input_path}. "
            "Please check input log format."
        )

    grouped_shapes_no_count = build_grouped_shapes(records, include_count=False)
    grouped_shapes_no_count = filter_by_op(grouped_shapes_no_count, args.op)
    dump_shapes_yaml(grouped_shapes_no_count, output_path, include_count=False)

    extra_output_paths: list[Path] = []
    if args.split1:
        split_eq_1_path = output_path.with_name(f"{output_path.stem}-N=1.yaml")
        split_ne_1_path = output_path.with_name(f"{output_path.stem}-N!=1.yaml")

        grouped_shapes_n_eq_1 = filter_by_n_eq_1(grouped_shapes_no_count, is_n_eq_1=True)
        grouped_shapes_n_ne_1 = filter_by_n_eq_1(grouped_shapes_no_count, is_n_eq_1=False)

        dump_shapes_yaml(grouped_shapes_n_eq_1, split_eq_1_path, include_count=False)
        dump_shapes_yaml(grouped_shapes_n_ne_1, split_ne_1_path, include_count=False)
        extra_output_paths.extend([split_eq_1_path, split_ne_1_path])

    grouped_shapes_with_count = build_grouped_shapes(records, include_count=True)
    grouped_shapes_with_count = filter_by_op(grouped_shapes_with_count, args.op)
    dump_shapes_yaml(grouped_shapes_with_count, output_count_path, include_count=True)

    print(f"Generated {output_path} from {input_path}")
    for extra_output_path in extra_output_paths:
        print(f"Generated {extra_output_path} from {input_path}")
    print(f"Generated {output_count_path} from {input_path}")
    print_operator_summary(grouped_shapes_no_count)


if __name__ == "__main__":
    main()
