#!/usr/bin/env python3
"""Torch profiler analysis for CUDA vs FlagGems.

严格模式：
- op_name 只来自 trace 中的 cpu_op 映射
- 未映射到 cpu_op 的 kernel 统一记为 "unknow"

输入:
- --torch_path: 包含 report-cuda / report-gems-all 的目录
- 未指定时从 tool_config.yaml 读取: results/{model_name}/torch-raw

输出:
- --output_path: 报告输出目录
- 未指定时从 tool_config.yaml 读取: reports/{model_name}/

默认生成:
- perf_analysis_torch.md / perf_analysis_torch.xlsx
"""

from __future__ import annotations

import argparse
import mmap
import json
import math
import numbers
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml
from openpyxl import Workbook

try:
    import ijson  # type: ignore
except ImportError:  # pragma: no cover
    ijson = None


TARGET_SOURCE_PREFIXES = (
    "vllm/model_executor/",
    "vllm/v1/attention/backends/",
    "vllm/distributed/",
)
UNMAPPED_OP_NAME = "unknow"
EVENT_SEPARATOR = b"\n  },\n  {"
EVENT_START_PREFIX = b"  {"
PH_X_MARKER = b'"ph": "X"'
CPU_OP_CAT_MARKER = b'"cat": "cpu_op"'
PYTHON_FUNC_CAT_MARKER = b'"cat": "python_function"'
KERNEL_CAT_MARKER = b'"cat": "kernel"'
NAME_MARKER = b'"name": "'
TID_MARKER = b'"tid": '
TS_MARKER = b'"ts": '
DUR_MARKER = b'"dur": '
EXTERNAL_ID_MARKER = b'"External id": '
NUMERIC_STOP_BYTES = b",}\r\n] "


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_tool_config() -> Optional[Dict[str, Any]]:
    cfg = get_project_root() / "scripts" / "tools" / "tool_config.yaml"
    if not cfg.exists():
        return None
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))


def resolve_path(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def get_default_paths() -> Tuple[Path, Path]:
    root = get_project_root()
    cfg = load_tool_config() or {}
    paths = cfg.get("paths", {})

    model_name = paths.get("model_name", "default")
    results_dir = paths.get("results", "results")

    torch_dir_cfg = paths.get("torch_output_dir")
    if torch_dir_cfg:
        torch_dir = resolve_path(root, torch_dir_cfg)
    else:
        torch_dir = root / results_dir / model_name / "torch-raw"

    reports_dir_cfg = paths.get("reports_dir")
    if reports_dir_cfg:
        out_dir = resolve_path(root, reports_dir_cfg)
    else:
        out_dir = root / "reports" / model_name

    return torch_dir, out_dir


def normalize_kernel_name(kernel_name: str) -> str:
    return re.sub(r"\s+", " ", kernel_name.strip())


def merge_name(existing: Optional[Set[str]], new_name: str) -> Set[str]:
    names = set(existing or set())
    if new_name:
        names.add(new_name)
    return names


def format_op_names(names: Iterable[str]) -> str:
    normalized = sorted({str(name) for name in names if str(name)})
    return ";".join(normalized) if normalized else UNMAPPED_OP_NAME


def format_values(values: Iterable[str]) -> str:
    normalized = sorted({str(value) for value in values if str(value)})
    return ";".join(normalized)


def parse_self_cuda_total_us(txt_path: Path) -> float:
    if not txt_path.exists():
        return 0.0
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Self CUDA time total:\s*([0-9.]+)\s*([num]?s)", text)
    if not m:
        return 0.0

    value = float(m.group(1))
    unit = m.group(2)
    if unit == "s":
        return value * 1_000_000.0
    if unit == "ms":
        return value * 1_000.0
    if unit == "us":
        return value
    if unit == "ns":
        return value / 1_000.0
    return 0.0


def iter_trace_events(trace_path: Path) -> Iterable[Dict[str, Any]]:
    if ijson is not None:
        with trace_path.open("rb") as f:
            yield from ijson.items(
                f,
                "traceEvents.item",
                use_float=True,
                buf_size=1024 * 1024,
            )
        return

    in_trace_events = False
    collecting = False
    braces = 0
    buf: List[str] = []

    with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not in_trace_events:
                if '"traceEvents"' in line:
                    in_trace_events = True
                continue

            stripped = line.lstrip()
            if stripped.startswith("]"):
                break

            if not collecting:
                if stripped.startswith("{"):
                    collecting = True
                    buf = [line]
                    braces = line.count("{") - line.count("}")
                continue

            buf.append(line)
            braces += line.count("{") - line.count("}")
            if braces != 0:
                continue

            raw = "".join(buf).strip()
            if raw.endswith(","):
                raw = raw[:-1]
            collecting = False
            buf = []
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield event


def is_aten_op(op_name: str) -> bool:
    return op_name.startswith("aten::")


def extract_source_file(python_func_name: str) -> str:
    return python_func_name.split(":", 1)[0].strip()


def source_path(source_file: str) -> str:
    return source_file.split("(", 1)[0].strip()


def source_rank(source_file: str, op_name: str) -> Tuple[int, int, int, str]:
    if is_aten_op(op_name):
        return (0, 0, 0, "")
    if not source_file:
        return (5, 0, 0, "")

    path = source_path(source_file)
    if path.startswith(TARGET_SOURCE_PREFIXES):
        return (1, -source_file.count("/"), -len(source_file), source_file)
    if path == "vllm/_custom_ops.py":
        return (6, 0, 0, source_file)
    if path.startswith("vllm/v1/"):
        return (2, -source_file.count("/"), -len(source_file), source_file)
    if path.startswith("vllm/") and not path.startswith("vllm/third_party/"):
        return (3, -source_file.count("/"), -len(source_file), source_file)
    if path.startswith("torch/"):
        return (5, -source_file.count("/"), -len(source_file), source_file)
    return (4, -source_file.count("/"), -len(source_file), source_file)


def pick_best_source_file(source_files: Iterable[str], op_name: str) -> str:
    items = [src for src in source_files if src]
    if not items:
        return ""
    return min(items, key=lambda src: source_rank(src, op_name))


def pick_preferred_source_file(frame_sources: List[str], op_name: str) -> str:
    if is_aten_op(op_name):
        return ""

    py_frames = [src for src in frame_sources if ".py(" in src]
    if not py_frames:
        return ""

    for src in reversed(py_frames):
        if source_path(src).startswith(TARGET_SOURCE_PREFIXES):
            return src

    for src in reversed(py_frames):
        if source_path(src) == "vllm/_custom_ops.py":
            continue
        if src.startswith("vllm/") and not src.startswith("vllm/third_party/"):
            return src

    for src in reversed(py_frames):
        if source_path(src) == "vllm/_custom_ops.py":
            continue
        if not src.startswith("torch/"):
            return src

    return py_frames[-1]


def pick_preferred_source_file_from_active_frames(
    active_frames: List[Tuple[float, str]],
    op_name: str,
) -> str:
    if is_aten_op(op_name):
        return ""

    py_frames: List[str] = []
    for _, src in active_frames:
        if ".py(" in src:
            py_frames.append(src)
    if not py_frames:
        return ""

    for src in reversed(py_frames):
        if source_path(src).startswith(TARGET_SOURCE_PREFIXES):
            return src

    for src in reversed(py_frames):
        if source_path(src) == "vllm/_custom_ops.py":
            continue
        if src.startswith("vllm/") and not src.startswith("vllm/third_party/"):
            return src

    for src in reversed(py_frames):
        if source_path(src) == "vllm/_custom_ops.py":
            continue
        if not src.startswith("torch/"):
            return src

    return py_frames[-1]


def build_cpu_op_source_map(
    python_frames_by_tid: Dict[int, List[Tuple[float, float, str]]],
    cpu_ops_by_tid: Dict[int, List[Tuple[float, str]]],
    cpu_op_names: Dict[str, Set[str]],
) -> Dict[str, str]:
    cpu_op_sources: Dict[str, str] = {}

    for tid, cpu_events in cpu_ops_by_tid.items():
        timeline: List[Tuple[float, int, str, str, float, Optional[str]]] = []
        for start_ts, end_ts, source_file in python_frames_by_tid.get(tid, []):
            timeline.append((start_ts, 0, "start", source_file, end_ts, None))
            timeline.append((end_ts, 2, "end", "", 0.0, None))
        for ts, ext_id in cpu_events:
            timeline.append((ts, 1, "cpu", "", 0.0, ext_id))

        timeline.sort(key=lambda x: (x[0], x[1]))
        active_frames: List[Tuple[float, str]] = []

        for ts, _, kind, source_file, end_ts, ext_id in timeline:
            while active_frames and ts > active_frames[-1][0]:
                active_frames.pop()

            if kind == "start":
                active_frames.append((end_ts, source_file))
                continue
            if kind == "end" or ext_id is None:
                continue

            op_name = format_op_names(cpu_op_names.get(ext_id, set()))
            frame_sources = [src for _, src in active_frames]
            cpu_op_sources[ext_id] = pick_preferred_source_file(frame_sources,
                                                                op_name)

    return cpu_op_sources


def build_cpu_op_source_map_fast(
    python_frames_by_tid: Dict[int, List[Tuple[float, float, str]]],
    cpu_ops_by_tid: Dict[int, List[Tuple[float, str]]],
    cpu_op_names: Dict[str, str],
) -> Dict[str, str]:
    cpu_op_sources: Dict[str, str] = {}

    for tid, cpu_events in cpu_ops_by_tid.items():
        frames = python_frames_by_tid.get(tid, [])
        if not frames or not cpu_events:
            continue

        active_frames: List[Tuple[float, str]] = []
        frame_idx = 0
        frame_count = len(frames)

        for ts, ext_id in cpu_events:
            while frame_idx < frame_count and frames[frame_idx][0] <= ts:
                start_ts, end_ts, source_file = frames[frame_idx]
                frame_idx += 1
                while active_frames and start_ts >= active_frames[-1][0]:
                    active_frames.pop()
                active_frames.append((end_ts, source_file))

            while active_frames and ts > active_frames[-1][0]:
                active_frames.pop()

            cpu_op_sources[ext_id] = pick_preferred_source_file_from_active_frames(
                active_frames, cpu_op_names.get(ext_id, UNMAPPED_OP_NAME))

    return cpu_op_sources


@dataclass(frozen=True)
class KernelStat:
    name: str
    op_name: str
    source_file: str
    calls: int
    total_us: float


@dataclass(frozen=True)
class OpAggStat:
    op_name: str
    source_file: str
    kernel_names: List[str]
    calls: int
    total_us: float


def resolve_op_name(cpu_names: Set[str]) -> str:
    if not cpu_names:
        return UNMAPPED_OP_NAME
    return format_op_names(cpu_names)


def merge_op_name(existing: Optional[str], new_name: str) -> str:
    if not existing:
        return new_name
    if existing == new_name:
        return existing
    return format_op_names((existing, new_name))


def get_cpu_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu)


def trim_event_object(raw: bytes) -> bytes:
    raw = raw.rstrip()
    if raw.endswith(b","):
        raw = raw[:-1].rstrip()
    return raw


def extract_json_string_field(raw: bytes, marker: bytes) -> Optional[str]:
    idx = raw.find(marker)
    if idx < 0:
        return None

    start = idx + len(marker)
    end = start
    escaped = False
    raw_len = len(raw)
    while end < raw_len:
        current = raw[end]
        if escaped:
            escaped = False
        elif current == 92:  # backslash
            escaped = True
        elif current == 34:  # quote
            segment = raw[start:end]
            if 92 not in segment:
                return segment.decode("utf-8", errors="ignore")
            try:
                return json.loads(raw[start - 1:end + 1])
            except json.JSONDecodeError:
                return None
        end += 1
    return None


def extract_numeric_bytes_field(raw: bytes, marker: bytes) -> Optional[bytes]:
    idx = raw.find(marker)
    if idx < 0:
        return None

    start = idx + len(marker)
    raw_len = len(raw)
    while start < raw_len and raw[start] in b" \t":
        start += 1
    if start >= raw_len or raw.startswith(b"null", start):
        return None

    end = start
    while end < raw_len and raw[end] not in NUMERIC_STOP_BYTES:
        end += 1
    if end <= start:
        return None
    return raw[start:end]


def extract_int_field(raw: bytes, marker: bytes) -> Optional[int]:
    value = extract_numeric_bytes_field(raw, marker)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def extract_float_field(raw: bytes, marker: bytes) -> Optional[float]:
    value = extract_numeric_bytes_field(raw, marker)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def align_to_next_event_start(trace_file: Path, offset: int, limit: int) -> int:
    with trace_file.open("rb") as f:
        f.seek(offset)
        if offset > 0:
            f.readline()
        while True:
            pos = f.tell()
            if pos >= limit:
                return limit
            line = f.readline()
            if not line:
                return limit
            if line.startswith(EVENT_START_PREFIX):
                return pos


def iter_event_objects_in_range(
    trace_file: Path,
    start_offset: int,
    end_offset: int,
) -> Iterable[bytes]:
    with trace_file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            cursor = start_offset
            while cursor < end_offset:
                next_sep = mm.find(EVENT_SEPARATOR, cursor, end_offset)
                if next_sep < 0:
                    raw = trim_event_object(mm[cursor:end_offset])
                    if raw:
                        yield raw
                    break

                raw = trim_event_object(mm[cursor:next_sep + 4])
                if raw:
                    yield raw
                cursor = next_sep + 6
        finally:
            mm.close()


def find_trace_event_section_offsets(trace_file: Path) -> Optional[Tuple[int, int, int, int]]:
    with trace_file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            trace_events_pos = mm.find(b'"traceEvents"')
            if trace_events_pos < 0:
                return None

            array_pos = mm.find(b"[", trace_events_pos)
            if array_pos < 0:
                return None

            python_pos = mm.find(b'"cat": "python_function"', array_pos)
            kernel_pos = mm.find(b'"cat": "kernel"', array_pos)
            if python_pos < 0 or kernel_pos < 0:
                return None

            cpu_start = mm.find(b"\n  {", array_pos)
            python_start = mm.rfind(b"\n  {", array_pos, python_pos)
            kernel_start = mm.rfind(b"\n  {", python_pos, kernel_pos)
            if cpu_start < 0 or python_start < 0 or kernel_start < 0:
                return None

            return cpu_start + 1, python_start + 1, kernel_start + 1, mm.size()
        finally:
            mm.close()


def compute_chunk_ranges(
    trace_file: Path,
    start_offset: int,
    end_offset: int,
    workers: int,
) -> List[Tuple[int, int]]:
    if workers <= 1 or end_offset <= start_offset:
        return [(start_offset, end_offset)]

    boundaries = [start_offset]
    for idx in range(1, workers):
        approx = start_offset + ((end_offset - start_offset) * idx) // workers
        aligned = align_to_next_event_start(trace_file, approx, end_offset)
        if aligned > boundaries[-1]:
            boundaries.append(aligned)
    boundaries.append(end_offset)

    ranges: List[Tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right > left:
            ranges.append((left, right))
    return ranges


def parse_trace_file_aggregates(
    trace_file: Path,
) -> Dict[Tuple[str, str, str], Tuple[int, float]]:
    cpu_op_names: Dict[str, str] = {}
    python_frames_by_tid: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
    cpu_ops_by_tid: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    source_needed_tids: Set[int] = set()
    kernel_agg: Dict[Tuple[str, str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])
    cpu_op_sources: Optional[Dict[str, str]] = None

    for event in iter_trace_events(trace_file):
        if event.get("ph") != "X":
            continue

        cat = str(event.get("cat", ""))
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        ext_id = args.get("External id")
        tid = event.get("tid")
        ts = event.get("ts")

        if cat == "python_function" and tid is not None and ts is not None:
            tid_int = int(tid)
            if tid_int not in source_needed_tids:
                continue
            dur = event.get("dur")
            if not isinstance(dur, numbers.Number):
                continue
            source_file = extract_source_file(str(event.get("name", "")))
            if ".py(" not in source_file:
                continue
            python_frames_by_tid[tid_int].append(
                (float(ts), float(ts) + float(dur), source_file))
            continue

        if cat == "cpu_op" and ext_id is not None and tid is not None and ts is not None:
            ext_key = str(ext_id)
            op_name = str(event.get("name", "unknown"))
            cpu_op_names[ext_key] = merge_op_name(cpu_op_names.get(ext_key), op_name)
            if not is_aten_op(op_name):
                tid_int = int(tid)
                cpu_ops_by_tid[tid_int].append((float(ts), ext_key))
                source_needed_tids.add(tid_int)
            continue

        if cat != "kernel":
            continue

        dur = event.get("dur", 0.0)
        if not isinstance(dur, numbers.Number):
            continue

        if cpu_op_sources is None:
            cpu_op_sources = build_cpu_op_source_map_fast(
                python_frames_by_tid,
                cpu_ops_by_tid,
                cpu_op_names,
            )
            python_frames_by_tid.clear()
            cpu_ops_by_tid.clear()
            source_needed_tids.clear()

        kernel_name = normalize_kernel_name(str(event.get("name", "unknown")))
        ext_key = str(ext_id) if ext_id is not None else ""
        op_name = cpu_op_names.get(ext_key, UNMAPPED_OP_NAME)
        source_file = cpu_op_sources.get(ext_key, "") if cpu_op_sources else ""
        key = (op_name, kernel_name, source_file)
        kernel_agg[key][0] += 1.0
        kernel_agg[key][1] += float(dur)

    return {k: (int(v[0]), float(v[1])) for k, v in kernel_agg.items()}


def parse_cpu_section_range(
    trace_file: Path,
    start_offset: int,
    end_offset: int,
) -> Tuple[Dict[str, str], Dict[int, List[Tuple[float, str]]]]:
    cpu_op_names: Dict[str, str] = {}
    cpu_ops_by_tid: Dict[int, List[Tuple[float, str]]] = defaultdict(list)

    for raw in iter_event_objects_in_range(trace_file, start_offset, end_offset):
        if PH_X_MARKER not in raw or CPU_OP_CAT_MARKER not in raw:
            continue

        ext_id = extract_int_field(raw, EXTERNAL_ID_MARKER)
        tid = extract_int_field(raw, TID_MARKER)
        ts = extract_float_field(raw, TS_MARKER)
        if ext_id is None or tid is None or ts is None:
            continue

        op_name = extract_json_string_field(raw, NAME_MARKER)
        if not op_name:
            continue

        ext_key = str(ext_id)
        cpu_op_names[ext_key] = merge_op_name(cpu_op_names.get(ext_key), op_name)
        if not is_aten_op(op_name):
            cpu_ops_by_tid[tid].append((ts, ext_key))

    return cpu_op_names, dict(cpu_ops_by_tid)


def parse_python_section_range(
    trace_file: Path,
    start_offset: int,
    end_offset: int,
    needed_tids: Set[int],
) -> Dict[int, List[Tuple[float, float, str]]]:
    python_frames_by_tid: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)

    for raw in iter_event_objects_in_range(trace_file, start_offset, end_offset):
        if PH_X_MARKER not in raw or PYTHON_FUNC_CAT_MARKER not in raw:
            continue

        tid = extract_int_field(raw, TID_MARKER)
        ts = extract_float_field(raw, TS_MARKER)
        dur = extract_float_field(raw, DUR_MARKER)
        if tid is None or ts is None or dur is None:
            continue

        if tid not in needed_tids:
            continue

        source_name = extract_json_string_field(raw, NAME_MARKER)
        if not source_name:
            continue

        source_file = extract_source_file(source_name)
        if ".py(" not in source_file:
            continue
        python_frames_by_tid[tid].append((ts, ts + dur, source_file))

    return dict(python_frames_by_tid)


def parse_kernel_section_range(
    trace_file: Path,
    start_offset: int,
    end_offset: int,
) -> Dict[Tuple[str, str], Tuple[int, float]]:
    kernel_agg: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])

    for raw in iter_event_objects_in_range(trace_file, start_offset, end_offset):
        if PH_X_MARKER not in raw or KERNEL_CAT_MARKER not in raw:
            continue

        dur = extract_float_field(raw, DUR_MARKER)
        if dur is None:
            continue

        kernel_name_raw = extract_json_string_field(raw, NAME_MARKER)
        if not kernel_name_raw:
            continue

        ext_id = extract_int_field(raw, EXTERNAL_ID_MARKER)
        ext_key = str(ext_id) if ext_id is not None else ""
        kernel_name = normalize_kernel_name(kernel_name_raw)
        key = (ext_key, kernel_name)
        kernel_agg[key][0] += 1.0
        kernel_agg[key][1] += dur

    return {k: (int(v[0]), float(v[1])) for k, v in kernel_agg.items()}


def parse_cpu_section_range_from_args(
    args: Tuple[Path, int, int],
) -> Tuple[Dict[str, str], Dict[int, List[Tuple[float, str]]]]:
    trace_file, start_offset, end_offset = args
    return parse_cpu_section_range(trace_file, start_offset, end_offset)


def parse_python_section_range_from_args(
    args: Tuple[Path, int, int, Set[int]],
) -> Dict[int, List[Tuple[float, float, str]]]:
    trace_file, start_offset, end_offset, needed_tids = args
    return parse_python_section_range(trace_file, start_offset, end_offset, needed_tids)


def parse_kernel_section_range_from_args(
    args: Tuple[Path, int, int],
) -> Dict[Tuple[str, str], Tuple[int, float]]:
    trace_file, start_offset, end_offset = args
    return parse_kernel_section_range(trace_file, start_offset, end_offset)


def parse_trace_file_aggregates_parallel(
    trace_file: Path,
    workers: int,
) -> Dict[Tuple[str, str, str], Tuple[int, float]]:
    offsets = find_trace_event_section_offsets(trace_file)
    if offsets is None:
        return parse_trace_file_aggregates(trace_file)

    cpu_start, python_start, kernel_start, file_end = offsets
    workers = max(1, workers)

    cpu_ranges = compute_chunk_ranges(trace_file, cpu_start, python_start, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        cpu_results = list(
            executor.map(parse_cpu_section_range_from_args,
                         ((trace_file, left, right) for left, right in cpu_ranges),
                         chunksize=1))

        cpu_op_names: Dict[str, str] = {}
        cpu_ops_by_tid: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
        for partial_names, partial_ops in cpu_results:
            for ext_key, name in partial_names.items():
                cpu_op_names[ext_key] = merge_op_name(cpu_op_names.get(ext_key), name)
            for tid, items in partial_ops.items():
                cpu_ops_by_tid[tid].extend(items)

        needed_tids = set(cpu_ops_by_tid.keys())
        python_ranges = compute_chunk_ranges(trace_file, python_start, kernel_start, workers)
        python_results = list(
            executor.map(parse_python_section_range_from_args,
                         ((trace_file, left, right, needed_tids)
                          for left, right in python_ranges),
                         chunksize=1))

        python_frames_by_tid: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
        for partial_frames in python_results:
            for tid, items in partial_frames.items():
                python_frames_by_tid[tid].extend(items)

        cpu_op_sources = build_cpu_op_source_map_fast(python_frames_by_tid, cpu_ops_by_tid,
                                                      cpu_op_names)

        kernel_ranges = compute_chunk_ranges(trace_file, kernel_start, file_end, workers)
        kernel_results = list(
            executor.map(parse_kernel_section_range_from_args,
                         ((trace_file, left, right) for left, right in kernel_ranges),
                         chunksize=1))

    kernel_agg: Dict[Tuple[str, str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])
    for partial_kernel in kernel_results:
        for (ext_key, kernel_name), (calls, total_us) in partial_kernel.items():
            op_name = cpu_op_names.get(ext_key, UNMAPPED_OP_NAME)
            source_file = cpu_op_sources.get(ext_key, "")
            key = (op_name, kernel_name, source_file)
            kernel_agg[key][0] += float(calls)
            kernel_agg[key][1] += total_us

    return {k: (int(v[0]), float(v[1])) for k, v in kernel_agg.items()}


def get_default_workers(num_files: int) -> int:
    return get_cpu_workers() if num_files > 0 else 1


def choose_file_parallelism(
    total_workers: int,
    trace_files: List[Path],
) -> int:
    file_count = len(trace_files)
    if total_workers <= 1 or file_count <= 1:
        return 1

    avg_file_size = sum(trace_file.stat().st_size for trace_file in trace_files) / file_count
    target_per_file_workers = 32 if avg_file_size >= 2 * 1024**3 else 16

    upper = min(total_workers, file_count)
    for candidate in range(upper, 0, -1):
        if total_workers % candidate == 0 and total_workers // candidate >= target_per_file_workers:
            return candidate

    for candidate in range(upper, 0, -1):
        if total_workers % candidate == 0:
            return candidate
    return 1


def parse_trace_file_with_workers(
    args: Tuple[Path, int],
) -> Dict[Tuple[str, str, str], Tuple[int, float]]:
    trace_file, workers = args
    if workers <= 1:
        return parse_trace_file_aggregates(trace_file)
    return parse_trace_file_aggregates_parallel(trace_file, workers)


def parse_profile_dir(report_dir: Path, workers: int = 1) -> Tuple[List[KernelStat], float]:
    trace_files = sorted(report_dir.glob("*.pt.trace.json"))
    if not trace_files:
        raise SystemExit(f"Missing trace file in {report_dir}")
    file_count = len(trace_files)
    workers = max(1, min(workers, get_cpu_workers()))

    txt_total_us = 0.0
    for txt in sorted(report_dir.glob("profiler_out_*.txt")):
        txt_total_us += parse_self_cuda_total_us(txt)

    kernel_agg: Dict[Tuple[str, str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])

    if file_count == 1 and workers > 1:
        parsed_results = [parse_trace_file_aggregates_parallel(trace_files[0], workers)]
    elif workers <= 1:
        parsed_results = [parse_trace_file_aggregates(trace_file) for trace_file in trace_files]
    else:
        file_parallelism = choose_file_parallelism(workers, trace_files)
        per_file_workers = max(1, workers // file_parallelism)
        with ThreadPoolExecutor(max_workers=file_parallelism) as executor:
            parsed_results = list(
                executor.map(
                    parse_trace_file_with_workers,
                    ((trace_file, per_file_workers) for trace_file in trace_files),
                ))

    for file_kernel_agg in parsed_results:
        for key, (calls, total_us) in file_kernel_agg.items():
            kernel_agg[key][0] += float(calls)
            kernel_agg[key][1] += total_us

    kernel_stats = [
        KernelStat(name=kernel_name,
                   op_name=op_name,
                   source_file=source_file,
                   calls=int(vals[0]),
                   total_us=vals[1])
        for (op_name, kernel_name, source_file), vals in kernel_agg.items()
    ]
    kernel_stats.sort(key=lambda x: x.total_us, reverse=True)

    kernel_total_us = sum(x.total_us for x in kernel_stats)
    total_ref_us = kernel_total_us if kernel_total_us > 0 else txt_total_us
    return kernel_stats, total_ref_us


def parse_profile_dirs_in_parallel(
    cuda_dir: Path,
    gems_dir: Path,
    workers: int,
) -> Tuple[Tuple[List[KernelStat], float], Tuple[List[KernelStat], float]]:
    cuda_trace_count = len(list(cuda_dir.glob("*.pt.trace.json")))
    gems_trace_count = len(list(gems_dir.glob("*.pt.trace.json")))
    per_dir_workers = workers
    if cuda_trace_count == 1 and gems_trace_count == 1 and workers > 1:
        per_dir_workers = max(1, workers // 2)

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_cuda = executor.submit(parse_profile_dir, cuda_dir, per_dir_workers)
        future_gems = executor.submit(parse_profile_dir, gems_dir, per_dir_workers)
        return future_cuda.result(), future_gems.result()


def us_to_ms(us: float) -> float:
    return us / 1000.0


def fmt_ms(us: float) -> str:
    return f"{us_to_ms(us):.3f}"


def fmt_us(us: float) -> str:
    return f"{us:.3f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_speedup(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:.3f}"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def aggregate_by_op(stats: List[KernelStat]) -> List[OpAggStat]:
    agg: Dict[str, Dict[str, Any]] = {}
    for item in stats:
        if item.op_name not in agg:
            agg[item.op_name] = {
                "source_files": set(),
                "kernel_names": set(),
                "calls": 0,
                "total_us": 0.0,
            }
        agg[item.op_name]["source_files"].add(item.source_file)
        agg[item.op_name]["kernel_names"].add(item.name)
        agg[item.op_name]["calls"] += item.calls
        agg[item.op_name]["total_us"] += item.total_us

    out: List[OpAggStat] = []
    for op_name, values in agg.items():
        out.append(
            OpAggStat(
                op_name=op_name,
                source_file=pick_best_source_file(values["source_files"], op_name),
                kernel_names=sorted(values["kernel_names"]),
                calls=int(values["calls"]),
                total_us=float(values["total_us"]),
            ))

    out.sort(key=lambda x: x.total_us, reverse=True)
    return out


def speedup(cuda_total_us: float, gems_total_us: float) -> float:
    if gems_total_us <= 0:
        return math.inf if cuda_total_us > 0 else 1.0
    return cuda_total_us / gems_total_us


def build_compare_rows(
    cuda_stats: List[KernelStat],
    gems_stats: List[KernelStat],
    cuda_total_ref_us: float,
    gems_total_ref_us: float,
    sort_by: str,
) -> List[List[str]]:
    cuda_agg = aggregate_by_op(cuda_stats)
    gems_agg = aggregate_by_op(gems_stats)
    cuda_map = {x.op_name: x for x in cuda_agg}
    gems_map = {x.op_name: x for x in gems_agg}

    all_ops = set(cuda_map) | set(gems_map)

    def cuda_total(op: str) -> float:
        return cuda_map[op].total_us if op in cuda_map else 0.0

    def sp(op: str) -> float:
        c = cuda_total(op)
        g = gems_map[op].total_us if op in gems_map else 0.0
        return speedup(c, g)

    if sort_by == "cuda_total_desc":
        ordered = sorted(all_ops, key=lambda x: (cuda_total(x), sp(x)), reverse=True)
    elif sort_by == "speedup_desc":
        ordered = sorted(all_ops, key=lambda x: (math.isinf(sp(x)), sp(x), cuda_total(x)), reverse=True)
    else:
        raise ValueError(f"Unsupported sort mode: {sort_by}")

    rows: List[List[str]] = []
    for op in ordered:
        c = cuda_map.get(op)
        g = gems_map.get(op)
        c_calls = c.calls if c else 0
        g_calls = g.calls if g else 0
        if c_calls == 0 or g_calls == 0:
            continue

        c_total = c.total_us if c else 0.0
        g_total = g.total_us if g else 0.0
        c_pct = c_total / cuda_total_ref_us if cuda_total_ref_us > 0 else 0.0
        g_pct = g_total / gems_total_ref_us if gems_total_ref_us > 0 else 0.0

        kernel_names: Set[str] = set()
        if c:
            kernel_names.update(c.kernel_names)
        if g:
            kernel_names.update(g.kernel_names)

        source_files = []
        if c:
            source_files.append(c.source_file)
        if g:
            source_files.append(g.source_file)

        rows.append([
            pick_best_source_file(source_files, op),
            op,
            md_escape(format_values(kernel_names)),
            str(c_calls),
            fmt_ms(c_total),
            fmt_pct(c_pct),
            str(g_calls),
            fmt_ms(g_total),
            fmt_pct(g_pct),
            fmt_speedup(speedup(c_total, g_total)),
        ])

    return rows


def write_excel_tables(path: Path, tables: List[Dict[str, Any]]) -> None:
    wb = Workbook()
    default_ws = wb.active
    wb.remove(default_ws)

    for table in tables:
        ws = wb.create_sheet(title=str(table["sheet_name"])[:31])
        ws.append(table["headers"])
        for row in table["rows"]:
            ws.append(row)

    wb.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlagGems vs CUDA torch profiler 性能对比分析")
    parser.add_argument(
        "--torch_path",
        "--input_path",
        dest="torch_path",
        type=str,
        default=None,
        help="包含 report-cuda 和 report-gems-all 的目录路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="报告输出目录路径（会自动创建）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行解析 trace 文件的进程数，默认使用当前 CPU 最大核心数；单个超大 trace 也会并行切分解析",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("cuda", "gems", "compare"),
        default="cuda",
        help="报告模式：cuda（仅 CUDA）、gems（仅 FlagGems）、compare（CUDA 与 FlagGems 对比）",
    )
    return parser.parse_args()


def build_mode_rows(stats: List[OpAggStat], total_ref_us: float) -> List[List[str]]:
    rows: List[List[str]] = []
    for row in stats:
        avg_us = row.total_us / row.calls if row.calls > 0 else 0.0
        rows.append([
            row.source_file,
            row.op_name,
            md_escape(format_values(row.kernel_names)),
            str(row.calls),
            fmt_ms(row.total_us),
            fmt_us(avg_us),
            fmt_pct(row.total_us / total_ref_us if total_ref_us > 0 else 0.0),
        ])
    return rows


def main() -> None:
    args = parse_args()
    root = get_project_root()
    default_torch_dir, default_output_dir = get_default_paths()

    torch_dir = resolve_path(root, args.torch_path) if args.torch_path else default_torch_dir
    output_dir = resolve_path(root, args.output_path) if args.output_path else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_dir = torch_dir / "report-cuda"
    gems_dir = torch_dir / "report-gems-all"
    if args.mode in ("cuda", "compare") and not cuda_dir.exists():
        raise SystemExit(f"Missing CUDA report directory: {cuda_dir}")
    if args.mode in ("gems", "compare") and not gems_dir.exists():
        raise SystemExit(f"Missing FlagGems report directory: {gems_dir}")

    out_md = output_dir / "perf_analysis_torch.md"
    out_xlsx = output_dir / "perf_analysis_torch.xlsx"

    trace_file_count = 0
    if args.mode in ("cuda", "compare"):
        trace_file_count += len(list(cuda_dir.glob("*.pt.trace.json")))
    if args.mode in ("gems", "compare"):
        trace_file_count += len(list(gems_dir.glob("*.pt.trace.json")))
    workers = args.workers if args.workers is not None else get_default_workers(trace_file_count)

    cuda_stats: List[KernelStat] = []
    gems_stats: List[KernelStat] = []
    cuda_total_ref_us = 0.0
    gems_total_ref_us = 0.0
    cuda_agg: List[OpAggStat] = []
    gems_agg: List[OpAggStat] = []

    if args.mode == "compare":
        (cuda_stats, cuda_total_ref_us), (gems_stats,
                                          gems_total_ref_us) = parse_profile_dirs_in_parallel(
                                              cuda_dir, gems_dir, workers=workers)
        cuda_agg = aggregate_by_op(cuda_stats)
        gems_agg = aggregate_by_op(gems_stats)
    elif args.mode == "cuda":
        cuda_stats, cuda_total_ref_us = parse_profile_dir(cuda_dir, workers=workers)
        cuda_agg = aggregate_by_op(cuda_stats)
    elif args.mode == "gems":
        gems_stats, gems_total_ref_us = parse_profile_dir(gems_dir, workers=workers)
        gems_agg = aggregate_by_op(gems_stats)

    report: List[str] = []
    excel_tables: List[Dict[str, Any]] = []

    cuda_headers = ["source file", "op_name", "kernel_name", "调用次数", "总时间(ms)", "平均时间(us)", "占比"]
    gems_headers = ["source file", "op_name", "kernel_name", "调用次数", "总时间(ms)", "平均时间(us)", "占比"]
    if args.mode in ("cuda", "compare"):
        cuda_rows = build_mode_rows(cuda_agg, cuda_total_ref_us)
        report.append("## CUDA kernel（按总时间排序）")
        report.append("")
        report.append(md_table(cuda_headers, cuda_rows))
        report.append("")
        excel_tables.append({"sheet_name": "CUDA按总时间", "headers": cuda_headers, "rows": cuda_rows})

    if args.mode in ("gems", "compare"):
        gems_rows = build_mode_rows(gems_agg, gems_total_ref_us)
        report.append("## FlagGems kernel（按总时间排序）")
        report.append("")
        report.append(md_table(gems_headers, gems_rows))
        report.append("")
        excel_tables.append({"sheet_name": "FlagGems按总时间", "headers": gems_headers, "rows": gems_rows})

    if args.mode == "compare":
        compare_headers = [
            "source file",
            "op_name",
            "kernel_name",
            "CUDA调用次数",
            "CUDA总时间(ms)",
            "CUDA占比",
            "FlagGems调用次数",
            "FlagGems总时间(ms)",
            "FlagGems占比",
            "加速比(CUDA/FlagGems)",
        ]

        compare_cuda_rows = build_compare_rows(
            cuda_stats,
            gems_stats,
            cuda_total_ref_us,
            gems_total_ref_us,
            sort_by="cuda_total_desc",
        )
        compare_speed_rows = build_compare_rows(
            cuda_stats,
            gems_stats,
            cuda_total_ref_us,
            gems_total_ref_us,
            sort_by="speedup_desc",
        )

        report.append("## CUDA 和 FlagGems kernel 对比（按 CUDA 总时间排序）")
        report.append("")
        report.append(md_table(compare_headers, compare_cuda_rows))
        report.append("")

        report.append("## CUDA 和 FlagGems kernel 对比（按加速比从高到低）")
        report.append("")
        report.append(md_table(compare_headers, compare_speed_rows))
        report.append("")

        excel_tables.extend([
            {"sheet_name": "对比按CUDA总时间", "headers": compare_headers, "rows": compare_cuda_rows},
            {"sheet_name": "对比按加速比", "headers": compare_headers, "rows": compare_speed_rows},
        ])

    out_md.write_text("\n".join(report), encoding="utf-8")
    write_excel_tables(out_xlsx, excel_tables)

    for stale_file in (
        output_dir / "shape_analysis_torch.md",
        output_dir / "shape_analysis_torch.xlsx",
    ):
        if stale_file.exists():
            stale_file.unlink()

    print(f"Generated: {out_md}")
    print(f"Generated: {out_xlsx}")


if __name__ == "__main__":
    main()
