#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DB_PATH = Path("/root/.flaggems/config_cache")
OP_TABLE_PATTERNS = {
    "mm_general": {
        "include_prefixes": ("mm_kernel_general_",),
        "exclude_prefixes": ("mm_kernel_general_host_tma_",),
    },
    "mm_general_tma": {
        "include_prefixes": ("mm_kernel_general_host_tma_",),
        "exclude_prefixes": (),
    },
    "mm_gemv": {
        "include_prefixes": ("gemv_kernel_",),
        "exclude_prefixes": (),
    },
    "w8a8_block_fp8_general": {
        "include_prefixes": ("w8a8_block_fp8_matmul_kernel_",),
        "exclude_prefixes": (),
    },
    "w8a8_block_fp8_general_tma": {
        "include_prefixes": ("w8a8_block_fp8_matmul_sqmma_kernel_",),
        "exclude_prefixes": (),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a narrowed mm expand config from existing FlagGems autotune DB entries."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Autotune sqlite DB directory or file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--yaml",
        dest="input_yaml",
        type=Path,
        required=True,
        help="Source expand yaml path",
    )
    return parser.parse_args()


def resolve_db_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Autotune DB path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Autotune DB path must be a .db file or directory: {path}")

    candidates = sorted(path.glob("TunedConfig*.db"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No TunedConfig*.db found under: {path}")
    raise ValueError(
        f"Found multiple TunedConfig*.db files under {path}, please pass --db with a specific file."
    )


def build_output_paths(input_yaml: Path) -> tuple[Path, Path, Path]:
    narrow_yaml = input_yaml.with_name(f"{input_yaml.stem}_narrow{input_yaml.suffix}")
    remove_yaml = input_yaml.with_name(f"{input_yaml.stem}_remove{input_yaml.suffix}")
    count_yaml = input_yaml.with_name(f"{input_yaml.stem}_count{input_yaml.suffix}")
    return narrow_yaml, remove_yaml, count_yaml


def load_yaml(path: Path) -> OrderedDict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping yaml in {path}, but got {type(data).__name__}")
    return OrderedDict(data)


def dump_yaml(path: Path, data: OrderedDict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        items = list(data.items())
        for idx, (key, value) in enumerate(items):
            file.write(f"{key}:\n")
            if isinstance(value, list):
                for entry in value:
                    write_list_entry(file, entry, indent=2)
            elif isinstance(value, dict):
                write_mapping(file, value, indent=2)
            else:
                file.write(f"  {format_scalar(value)}\n")
            if idx != len(items) - 1:
                file.write("\n")


def write_list_entry(file, entry: Any, indent: int) -> None:
    if not isinstance(entry, dict):
        file.write(" " * indent + f"- {format_scalar(entry)}\n")
        return

    items = list(entry.items())
    if not items:
        file.write(" " * indent + "- {}\n")
        return

    first_key, first_value = items[0]
    if isinstance(first_value, dict):
        file.write(" " * indent + f"- {first_key}:\n")
        write_mapping(file, first_value, indent=indent + 4)
    elif first_value is None:
        file.write(" " * indent + f"- {first_key}:\n")
    elif isinstance(first_value, list):
        if first_value:
            file.write(" " * indent + f"- {first_key}:\n")
            write_list(file, first_value, indent=indent + 4)
        else:
            file.write(" " * indent + f"- {first_key}: []\n")
    else:
        file.write(" " * indent + f"- {first_key}: {format_scalar(first_value)}\n")

    for key, value in items[1:]:
        write_mapping_item(file, key, value, indent=indent + 2)


def write_mapping(file, mapping: dict[str, Any], indent: int) -> None:
    for key, value in mapping.items():
        write_mapping_item(file, key, value, indent)


def write_mapping_item(file, key: str, value: Any, indent: int) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        file.write(f"{prefix}{key}:\n")
        write_mapping(file, value, indent + 2)
        return
    if isinstance(value, list):
        if value:
            file.write(f"{prefix}{key}:\n")
            write_list(file, value, indent + 2)
        else:
            file.write(f"{prefix}{key}: []\n")
        return
    if value is None:
        file.write(f"{prefix}{key}:\n")
        return
    file.write(f"{prefix}{key}: {format_scalar(value)}\n")


def write_list(file, values: list[Any], indent: int) -> None:
    for value in values:
        prefix = " " * indent
        if isinstance(value, dict):
            write_list_entry(file, value, indent)
        elif isinstance(value, list):
            file.write(f"{prefix}-\n")
            write_list(file, value, indent + 2)
        else:
            file.write(f"{prefix}- {format_scalar(value)}\n")


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [name for (name,) in rows]


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return {row[1] for row in rows}


def is_benchmark_table(table_name: str) -> bool:
    return "benchmark" in table_name.lower()


def find_config_tables(conn: sqlite3.Connection, op_name: str) -> list[str]:
    pattern = OP_TABLE_PATTERNS.get(op_name)
    if pattern is None:
        return []

    include_prefixes = pattern["include_prefixes"]
    exclude_prefixes = pattern["exclude_prefixes"]
    matched_tables: list[str] = []
    for table_name in list_tables(conn):
        if not any(table_name.startswith(prefix) for prefix in include_prefixes):
            continue
        if is_benchmark_table(table_name):
            continue
        if any(table_name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        matched_tables.append(table_name)
    return matched_tables


def extract_config_block(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in entries:
        if isinstance(entry, dict) and "config" in entry:
            return entry
    raise ValueError("Unable to find 'config' block in yaml entries.")


def get_param_sources(config_block: dict[str, Any]) -> OrderedDict[str, str]:
    param_map = config_block.get("param_map")
    if not isinstance(param_map, dict):
        raise ValueError("Config block is missing a valid 'param_map'.")

    sources: OrderedDict[str, str] = OrderedDict()
    meta_map = param_map.get("META")
    if not isinstance(meta_map, dict):
        raise ValueError("This script currently expects META to be a mapping.")

    for meta_key, source_name in meta_map.items():
        if not isinstance(meta_key, str) or not isinstance(source_name, str):
            raise ValueError("META param_map entries must be string-to-string mappings.")
        sources[meta_key] = source_name

    for config_key in ("num_stages", "num_warps"):
        source_name = param_map.get(config_key)
        if isinstance(source_name, str):
            sources[config_key] = source_name

    return sources


def query_distinct_values(
    conn: sqlite3.Connection, table_names: list[str], column_name: str
) -> list[Any]:
    values: set[Any] = set()
    for table_name in table_names:
        columns = get_table_columns(conn, table_name)
        if column_name not in columns:
            continue
        rows = conn.execute(
            f'SELECT DISTINCT "{column_name}" FROM "{table_name}" ORDER BY "{column_name}"'
        ).fetchall()
        values.update(row[0] for row in rows if row[0] is not None)
    return sorted(values)


def query_value_counts(
    conn: sqlite3.Connection, table_names: list[str], column_name: str
) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for table_name in table_names:
        columns = get_table_columns(conn, table_name)
        if column_name not in columns:
            continue
        rows = conn.execute(
            f'SELECT "{column_name}", COUNT(*) FROM "{table_name}" '
            f'WHERE "{column_name}" IS NOT NULL GROUP BY "{column_name}"'
        ).fetchall()
        for value, count in rows:
            counts[value] = counts.get(value, 0) + count
    return counts


def ordered_intersection(original_values: list[Any], narrowed_values: list[Any]) -> list[Any]:
    narrowed_set = set(narrowed_values)
    return [value for value in original_values if value in narrowed_set]


def compute_search_space_size(config_block: dict[str, Any]) -> int:
    param_sources = get_param_sources(config_block)
    lengths = []
    for source_name in param_sources.values():
        values = config_block.get(source_name)
        if not isinstance(values, list):
            continue
        lengths.append(len(values))
    return math.prod(lengths) if lengths else 0


def ordered_difference(original_values: list[Any], kept_values: list[Any]) -> list[Any]:
    kept_set = set(kept_values)
    return [value for value in original_values if value not in kept_set]


def narrow_config_block(
    conn: sqlite3.Connection,
    op_name: str,
    config_block: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    table_names = find_config_tables(conn, op_name)
    if not table_names:
        return config_block, {
            "matched": False,
            "tables": [],
            "before": compute_search_space_size(config_block),
            "after": compute_search_space_size(config_block),
            "empty_fields": [],
        }

    narrowed_config = {
        key: list(value) if isinstance(value, list) else value
        for key, value in config_block.items()
    }

    empty_fields: list[str] = []
    param_sources = get_param_sources(config_block)
    for column_name, source_name in param_sources.items():
        original_values = config_block.get(source_name)
        if not isinstance(original_values, list):
            continue

        db_values = query_distinct_values(conn, table_names, column_name)
        narrowed_values = ordered_intersection(original_values, db_values)
        narrowed_config[source_name] = narrowed_values
        if not narrowed_values:
            empty_fields.append(source_name)

    return narrowed_config, {
        "matched": True,
        "tables": table_names,
        "before": compute_search_space_size(config_block),
        "after": compute_search_space_size(narrowed_config),
        "empty_fields": empty_fields,
    }


def build_count_config_block(
    conn: sqlite3.Connection,
    op_name: str,
    config_block: dict[str, Any],
) -> dict[str, Any]:
    table_names = find_config_tables(conn, op_name)
    count_config = {
        key: list(value) if isinstance(value, list) else value
        for key, value in config_block.items()
    }

    for column_name, source_name in get_param_sources(config_block).items():
        original_values = config_block.get(source_name)
        if not isinstance(original_values, list):
            continue

        value_counts = query_value_counts(conn, table_names, column_name)
        count_config[source_name] = [
            OrderedDict([("value", value), ("count", value_counts.get(value, 0))])
            for value in original_values
        ]

    return count_config


def build_narrow_yaml(
    conn: sqlite3.Connection,
    source_yaml: OrderedDict[str, Any],
) -> tuple[OrderedDict[str, Any], dict[str, dict[str, Any]]]:
    output_yaml: OrderedDict[str, Any] = OrderedDict()
    summaries: dict[str, dict[str, Any]] = {}

    for op_name, entries in source_yaml.items():
        if not isinstance(entries, list):
            output_yaml[op_name] = entries
            continue

        config_block = extract_config_block(entries)
        narrowed_config, summary = narrow_config_block(conn, op_name, config_block)
        summaries[op_name] = summary

        new_entries: list[dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict) and "config" in entry:
                new_entries.append(narrowed_config)
            else:
                new_entries.append(entry)
        output_yaml[op_name] = new_entries

    return output_yaml, summaries


def build_count_yaml(
    conn: sqlite3.Connection,
    source_yaml: OrderedDict[str, Any],
) -> OrderedDict[str, Any]:
    count_yaml: OrderedDict[str, Any] = OrderedDict()

    for op_name, entries in source_yaml.items():
        if not isinstance(entries, list):
            count_yaml[op_name] = entries
            continue

        config_block = extract_config_block(entries)
        counted_config = build_count_config_block(conn, op_name, config_block)

        new_entries: list[dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict) and "config" in entry:
                new_entries.append(counted_config)
            else:
                new_entries.append(entry)
        count_yaml[op_name] = new_entries

    return count_yaml


def build_remove_yaml(
    source_yaml: OrderedDict[str, Any],
    narrowed_yaml: OrderedDict[str, Any],
) -> OrderedDict[str, Any]:
    remove_yaml: OrderedDict[str, Any] = OrderedDict()

    for op_name, source_entries in source_yaml.items():
        narrowed_entries = narrowed_yaml.get(op_name)
        if not isinstance(source_entries, list) or not isinstance(narrowed_entries, list):
            remove_yaml[op_name] = source_entries
            continue

        source_config = extract_config_block(source_entries)
        narrowed_config = extract_config_block(narrowed_entries)
        remove_config = {
            key: list(value) if isinstance(value, list) else value
            for key, value in source_config.items()
        }

        for source_name in get_param_sources(source_config).values():
            original_values = source_config.get(source_name)
            kept_values = narrowed_config.get(source_name)
            if not isinstance(original_values, list):
                continue
            if not isinstance(kept_values, list):
                kept_values = [kept_values] if kept_values is not None else []
            remove_config[source_name] = ordered_difference(original_values, kept_values)

        new_entries: list[dict[str, Any]] = []
        for entry in source_entries:
            if isinstance(entry, dict) and "config" in entry:
                new_entries.append(remove_config)
            else:
                new_entries.append(entry)
        remove_yaml[op_name] = new_entries

    return remove_yaml


def print_summary(summaries: dict[str, dict[str, Any]]) -> None:
    print("Narrowed mm expand config summary:")
    for op_name, summary in summaries.items():
        before = summary["before"]
        after = summary["after"]
        tables = summary["tables"]
        empty_fields = summary["empty_fields"]

        if not summary["matched"]:
            print(f"  - {op_name}: no matching config table, kept original search space ({before} configs)")
            continue

        print(f"  - {op_name}: {before} -> {after} configs from {len(tables)} table(s)")
        for table_name in tables:
            print(f"    table: {table_name}")
        if empty_fields:
            print(f"    no overlapping DB values for: {', '.join(empty_fields)}")


def main() -> None:
    args = parse_args()

    db_path = resolve_db_path(args.db)
    if not args.input_yaml.exists():
        raise FileNotFoundError(f"Input yaml not found: {args.input_yaml}")
    output_yaml, remove_yaml, count_yaml = build_output_paths(args.input_yaml)

    source_yaml = load_yaml(args.input_yaml)
    with sqlite3.connect(db_path) as conn:
        narrowed_yaml, summaries = build_narrow_yaml(conn, source_yaml)
        counted_yaml = build_count_yaml(conn, source_yaml)
    removed_yaml = build_remove_yaml(source_yaml, narrowed_yaml)

    dump_yaml(output_yaml, narrowed_yaml)
    dump_yaml(remove_yaml, removed_yaml)
    dump_yaml(count_yaml, counted_yaml)
    print(f"Using autotune DB: {db_path}")
    print_summary(summaries)
    print(f"Wrote narrowed yaml to: {output_yaml}")
    print(f"Wrote removed-values yaml to: {remove_yaml}")
    print(f"Wrote count yaml to: {count_yaml}")


if __name__ == "__main__":
    main()
