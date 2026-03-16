#!/usr/bin/env python3
"""Torch profiler analysis for CUDA vs FlagGems.

Input (default):
- results/{model_name}/torch-raw/report-cuda/*.pt.trace.json
- results/{model_name}/torch-raw/report-gems-all/*.pt.trace.json
- results/{model_name}/torch-raw/report-*/profiler_out_*.txt (optional)

Output (default):
- reports/{model_name}/perf_analysis_torch.md
- reports/{model_name}/perf_analysis_torch.xlsx
- reports/{model_name}/shape_analysis_torch.md
- reports/{model_name}/shape_analysis_torch.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from openpyxl import Workbook


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
    name = kernel_name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def infer_aten_op(kernel_name: str) -> str:
    lower = kernel_name.lower()

    if lower.startswith("aten::"):
        return kernel_name
    if lower.startswith("triton_poi_fused_"):
        return "triton::fused_pointwise"
    if lower.startswith("triton_per_fused_"):
        return "triton::fused_persistent"
    if lower.startswith("triton_red_fused_"):
        return "triton::fused_reduction"
    if lower.startswith("nvjet_tst_") or "mm_kernel" in lower or "gemm" in lower or "marlin" in lower:
        return "aten::mm"
    if "fused_moe_kernel" in lower:
        return "vllm::fused_moe"
    if "fused_recurrent_gated_delta_rule_fwd_kernel" in lower:
        return "vllm::fused_recurrent_gated_delta_rule_fwd"
    if "topkgating" in lower:
        return "vllm::topk_gating"
    if "moe_align_block_size" in lower:
        return "vllm::moe_align_block_size"
    if "reshape_and_cache" in lower:
        return "vllm::reshape_and_cache"
    if "prepare_varlen_num_blocks" in lower:
        return "vllm::prepare_varlen_num_blocks"
    if "softmax" in lower:
        return "aten::softmax"
    if "copy" in lower:
        return "aten::copy_"
    if "sum" in lower:
        return "aten::sum"
    if "mul" in lower:
        return "aten::mul"
    if "exp" in lower:
        return "aten::exp"
    if "argmax" in lower:
        return "aten::argmax"
    if "index_select" in lower or "indexselect" in lower:
        return "aten::index_select"
    if "embedding" in lower:
        return "aten::embedding"
    return "unknown"


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
    """Iterate trace events without loading the whole JSON into memory."""
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

            if braces == 0:
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


@dataclass(frozen=True)
class KernelStat:
    name: str
    aten_op: str
    calls: int
    total_us: float


@dataclass(frozen=True)
class AtenAggStat:
    aten_op: str
    kernel_names: List[str]
    calls: int
    total_us: float


@dataclass(frozen=True)
class MmKernelShapeStat:
    framework_op: str
    exec_op: str
    input_shape: str
    calls: int
    total_us: float


def format_input_dims(dims: Any) -> str:
    if not isinstance(dims, list):
        return "N/A"
    parts: List[str] = []
    for item in dims:
        if isinstance(item, list) and item:
            parts.append("x".join(str(v) for v in item))
    return ", ".join(parts) if parts else "N/A"


def parse_trace_stats(trace_file: Path) -> Tuple[Dict[str, Tuple[str, str]], List[Tuple[str, int, float]]]:
    """Return (external_id -> (cpu_op, input_shape), kernel_events).

    kernel_events item: (kernel_name, external_id, dur_us)
    """
    cpu_info: Dict[str, Tuple[str, str]] = {}
    kernel_events: List[Tuple[str, int, float]] = []

    for event in iter_trace_events(trace_file):
        cat = str(event.get("cat", ""))
        if event.get("ph") != "X":
            continue

        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        ext_id = args.get("External id")

        if cat == "cpu_op" and ext_id is not None:
            name = str(event.get("name", "unknown"))
            shape = format_input_dims(args.get("Input Dims"))
            cpu_info[str(ext_id)] = (name, shape)
            continue

        if cat == "kernel":
            dur = event.get("dur", 0.0)
            if not isinstance(dur, (int, float)):
                continue
            name = normalize_kernel_name(str(event.get("name", "unknown")))
            ext = int(ext_id) if isinstance(ext_id, int) else -1
            kernel_events.append((name, ext, float(dur)))

    return cpu_info, kernel_events


def parse_profile_dir(report_dir: Path) -> Tuple[List[KernelStat], List[MmKernelShapeStat], float]:
    trace_files = sorted(report_dir.glob("*.pt.trace.json"))
    if not trace_files:
        raise SystemExit(f"Missing trace file in {report_dir}")

    txt_total_us = 0.0
    for txt in sorted(report_dir.glob("profiler_out_*.txt")):
        txt_total_us += parse_self_cuda_total_us(txt)

    kernel_agg: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])
    mm_shape_agg: Dict[Tuple[str, str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])

    for trace_file in trace_files:
        cpu_info, kernel_events = parse_trace_stats(trace_file)

        for kernel_name, ext_id, dur_us in kernel_events:
            cpu_name = None
            input_shape = "N/A"
            if ext_id >= 0:
                pair = cpu_info.get(str(ext_id))
                if pair:
                    cpu_name, input_shape = pair

            if cpu_name and cpu_name.startswith("aten::"):
                framework_op = cpu_name
            else:
                framework_op = infer_aten_op(kernel_name)

            kk = (framework_op, kernel_name)
            kernel_agg[kk][0] += 1.0
            kernel_agg[kk][1] += dur_us

            if framework_op == "aten::mm":
                sk = (framework_op, kernel_name, input_shape)
                mm_shape_agg[sk][0] += 1.0
                mm_shape_agg[sk][1] += dur_us

    kernel_stats = [
        KernelStat(name=kname, aten_op=op, calls=int(vals[0]), total_us=vals[1])
        for (op, kname), vals in kernel_agg.items()
    ]
    kernel_stats.sort(key=lambda x: x.total_us, reverse=True)

    mm_stats = [
        MmKernelShapeStat(
            framework_op=op,
            exec_op=kname,
            input_shape=shape,
            calls=int(vals[0]),
            total_us=vals[1],
        )
        for (op, kname, shape), vals in mm_shape_agg.items()
    ]
    mm_stats.sort(key=lambda x: x.total_us, reverse=True)

    kernel_total_us = sum(x.total_us for x in kernel_stats)
    total_ref_us = kernel_total_us if kernel_total_us > 0 else txt_total_us
    return kernel_stats, mm_stats, total_ref_us


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


def aggregate_by_aten(stats: List[KernelStat]) -> List[AtenAggStat]:
    agg: Dict[str, Dict[str, Any]] = {}
    for item in stats:
        if item.aten_op not in agg:
            agg[item.aten_op] = {"names": set(), "calls": 0, "total_us": 0.0}
        agg[item.aten_op]["names"].add(item.name)
        agg[item.aten_op]["calls"] += item.calls
        agg[item.aten_op]["total_us"] += item.total_us

    out: List[AtenAggStat] = []
    for op, values in agg.items():
        out.append(
            AtenAggStat(
                aten_op=op,
                kernel_names=sorted(values["names"]),
                calls=int(values["calls"]),
                total_us=float(values["total_us"]),
            )
        )

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
    cuda_agg = aggregate_by_aten(cuda_stats)
    gems_agg = aggregate_by_aten(gems_stats)
    cuda_map = {x.aten_op: x for x in cuda_agg}
    gems_map = {x.aten_op: x for x in gems_agg}

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

        names = set()
        if c:
            names.update(c.kernel_names)
        if g:
            names.update(g.kernel_names)

        rows.append(
            [
                op,
                md_escape(", ".join(sorted(names))),
                str(c_calls),
                fmt_ms(c_total),
                fmt_pct(c_pct),
                str(g_calls),
                fmt_ms(g_total),
                fmt_pct(g_pct),
                fmt_speedup(speedup(c_total, g_total)),
            ]
        )

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
    parser = argparse.ArgumentParser(description="Analyze torch profiler outputs for CUDA and FlagGems")
    parser.add_argument("--torch_path", type=str, default=None, help="Path containing report-cuda/report-gems-all")
    parser.add_argument("--output_path", type=str, default=None, help="Output report directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_project_root()
    default_torch_dir, default_output_dir = get_default_paths()

    torch_dir = resolve_path(root, args.torch_path) if args.torch_path else default_torch_dir
    output_dir = resolve_path(root, args.output_path) if args.output_path else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_dir = torch_dir / "report-cuda"
    gems_dir = torch_dir / "report-gems-all"
    if not cuda_dir.exists() or not gems_dir.exists():
        raise SystemExit(f"Missing report directories under {torch_dir}")

    out_md = output_dir / "perf_analysis_torch.md"
    out_xlsx = output_dir / "perf_analysis_torch.xlsx"
    out_shape_md = output_dir / "shape_analysis_torch.md"
    out_shape_xlsx = output_dir / "shape_analysis_torch.xlsx"

    cuda_stats, cuda_mm_stats, cuda_total_ref_us = parse_profile_dir(cuda_dir)
    gems_stats, gems_mm_stats, gems_total_ref_us = parse_profile_dir(gems_dir)

    cuda_agg = aggregate_by_aten(cuda_stats)
    gems_agg = aggregate_by_aten(gems_stats)

    report: List[str] = []
    report.append("# Torch Profiler CUDA vs FlagGems Performance Analysis")
    report.append("")

    cuda_headers = ["框架算子名", "执行算子名", "调用次数", "总时间(ms)", "平均时间(us)", "占比"]
    cuda_rows: List[List[str]] = []
    for row in cuda_agg:
        avg_us = row.total_us / row.calls if row.calls > 0 else 0.0
        cuda_rows.append(
            [
                row.aten_op,
                md_escape(", ".join(row.kernel_names)),
                str(row.calls),
                fmt_ms(row.total_us),
                fmt_us(avg_us),
                fmt_pct(row.total_us / cuda_total_ref_us if cuda_total_ref_us > 0 else 0.0),
            ]
        )

    gems_headers = ["框架算子名", "执行算子名", "调用次数", "总时间(ms)", "平均时间(us)", "占比"]
    gems_rows: List[List[str]] = []
    for row in gems_agg:
        avg_us = row.total_us / row.calls if row.calls > 0 else 0.0
        gems_rows.append(
            [
                row.aten_op,
                md_escape(", ".join(row.kernel_names)),
                str(row.calls),
                fmt_ms(row.total_us),
                fmt_us(avg_us),
                fmt_pct(row.total_us / gems_total_ref_us if gems_total_ref_us > 0 else 0.0),
            ]
        )

    compare_headers = [
        "框架算子名",
        "执行算子名",
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

    report.append("## CUDA kernel (sorted by total time)")
    report.append("")
    report.append(md_table(cuda_headers, cuda_rows))
    report.append("")

    report.append("## FlagGems kernel (sorted by total time)")
    report.append("")
    report.append(md_table(gems_headers, gems_rows))
    report.append("")

    report.append("## CUDA vs FlagGems (sorted by CUDA total time)")
    report.append("")
    report.append(md_table(compare_headers, compare_cuda_rows))
    report.append("")

    report.append("## CUDA vs FlagGems (sorted by speedup)")
    report.append("")
    report.append(md_table(compare_headers, compare_speed_rows))
    report.append("")

    out_md.write_text("\n".join(report), encoding="utf-8")
    write_excel_tables(
        out_xlsx,
        [
            {"sheet_name": "CUDA按总时间", "headers": cuda_headers, "rows": cuda_rows},
            {"sheet_name": "FlagGems按总时间", "headers": gems_headers, "rows": gems_rows},
            {"sheet_name": "对比按CUDA总时间", "headers": compare_headers, "rows": compare_cuda_rows},
            {"sheet_name": "对比按加速比", "headers": compare_headers, "rows": compare_speed_rows},
        ],
    )

    shape_report: List[str] = []
    shape_report.append("# Torch Profiler aten::mm Shape Analysis")
    shape_report.append("")

    cuda_shape_headers = ["框架算子名", "执行算子名", "输入 shape", "CUDA调用次数", "CUDA总时间(ms)", "CUDA占比"]
    cuda_shape_rows: List[List[str]] = []
    for row in cuda_mm_stats:
        cuda_shape_rows.append(
            [
                row.framework_op,
                md_escape(row.exec_op),
                row.input_shape,
                str(row.calls),
                fmt_ms(row.total_us),
                fmt_pct(row.total_us / cuda_total_ref_us if cuda_total_ref_us > 0 else 0.0),
            ]
        )

    gems_shape_headers = ["框架算子名", "执行算子名", "输入 shape", "FlagGems调用次数", "FlagGems总时间(ms)", "FlagGems占比"]
    gems_shape_rows: List[List[str]] = []
    for row in gems_mm_stats:
        gems_shape_rows.append(
            [
                row.framework_op,
                md_escape(row.exec_op),
                row.input_shape,
                str(row.calls),
                fmt_ms(row.total_us),
                fmt_pct(row.total_us / gems_total_ref_us if gems_total_ref_us > 0 else 0.0),
            ]
        )

    shape_report.append("## CUDA aten::mm kernels (sorted by total time)")
    shape_report.append("")
    shape_report.append(md_table(cuda_shape_headers, cuda_shape_rows))
    shape_report.append("")

    shape_report.append("## FlagGems aten::mm kernels (sorted by total time)")
    shape_report.append("")
    shape_report.append(md_table(gems_shape_headers, gems_shape_rows))
    shape_report.append("")

    out_shape_md.write_text("\n".join(shape_report), encoding="utf-8")
    write_excel_tables(
        out_shape_xlsx,
        [
            {"sheet_name": "CUDA_mm_shape", "headers": cuda_shape_headers, "rows": cuda_shape_rows},
            {"sheet_name": "FlagGems_mm_shape", "headers": gems_shape_headers, "rows": gems_shape_rows},
        ],
    )

    print(f"Generated: {out_md}")
    print(f"Generated: {out_xlsx}")
    print(f"Generated: {out_shape_md}")
    print(f"Generated: {out_shape_xlsx}")


if __name__ == "__main__":
    main()
