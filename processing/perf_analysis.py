#!/usr/bin/env python3
"""FlagGems vs CUDA kernel性能对比分析

输入:
- nsys-cuda/report-cuda.sqlite
- nsys-gems/report-gems-all.sqlite

输出:
- reports/perf_analysis.md
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from openpyxl import Workbook


@dataclass(frozen=True)
class KernelStat:
    name: str
    aten_op: str
    calls: int
    total_ns: int
    avg_ns: float


@dataclass(frozen=True)
class AtenAggStat:
    aten_op: str
    kernel_names: List[str]
    calls: int
    total_ns: int
    avg_ns: float


@dataclass(frozen=True)
class MmKernelShapeStat:
    framework_op: str
    exec_op: str
    input_shape: str
    calls: int
    total_ns: int


def normalize_kernel_name(kernel_name: str) -> str:
    name = kernel_name

    # nvjet_tst_128x48_64x9_4x1_v_bz_splitK_TNT -> nvjet_tst_v_bz_splitK_TNT
    if name.startswith("nvjet_tst_"):
        tail = name[len("nvjet_tst_"):]
        parts = tail.split("_")
        while parts and re.fullmatch(r"\d+x\d+", parts[0]):
            parts.pop(0)
        if parts:
            return "nvjet_tst_" + "_".join(parts)

    # sm90_xmma_..._tilesize128x128x64_... -> sm90_xmma_..._tilesize_...
    name = re.sub(r"tilesize\d+x\d+x\d+", "tilesize", name)
    return name


def infer_aten_op(kernel_name: str) -> str:
    lower = kernel_name.lower()

    # 高优先级精确规则（先匹配）
    if lower.startswith("nvjet_tst_"):
        return "aten::mm"
    if lower.startswith("triton_poi_fused_"):
        return "triton::fused_pointwise"
    if lower.startswith("triton_per_fused_"):
        return "triton::fused_persistent"
    if lower.startswith("triton_red_fused_"):
        return "triton::fused_reduction"
    if lower == "fused_moe_kernel":
        return "vllm::fused_moe"
    if lower == "topkgating":
        return "vllm::topk_gating"
    if lower == "moe_align_block_size_kernel":
        return "vllm::moe_align_block_size"
    if lower == "reshape_and_cache_flash_kernel":
        return "vllm::reshape_and_cache"
    if lower == "prepare_varlen_num_blocks_kernel":
        return "vllm::prepare_varlen_num_blocks"
    if lower == "fused_recurrent_gated_delta_rule_fwd_kernel":
        return "vllm::fused_recurrent_gated_delta_rule_fwd"
    if lower == "fused_gdn_gating_kernel":
        return "vllm::fused_gdn_gating"
    if lower == "device_kernel":
        return "vllm::device_kernel"
    if lower == "sweep":
        return "vllm::sweep"
    if lower.startswith("sum_dim_kernel"):
        return "aten::sum"
    if lower.startswith("fill_scalar_func_kernel"):
        return "aten::fill_"
    if lower.startswith("sigmoid_forward"):
        return "aten::sigmoid"
    if lower.startswith("_index_put_jit_function"):
        return "aten::index_put"
    if lower.startswith("_index_jit_function"):
        return "aten::index"
    if lower == "devicesegmentedradixsortkernel":
        return "aten::sort"
    if lower == "dot_kernel":
        return "aten::dot"
    if lower == "rotary_kernel":
        return "vllm::rotary_embedding"
    if lower == "nonzero_kernel":
        return "aten::nonzero"
    if lower == "rand_kernel":
        return "aten::rand"
    if lower == "linspace_kernel":
        return "aten::linspace"
    if lower.startswith("zeros_kernel"):
        return "aten::zeros"
    if lower.startswith("ones_kernel"):
        return "aten::ones"
    if lower.startswith("arange_func"):
        return "aten::arange"
    if lower == "kernel2":
        return "custom::kernel2"
    if lower == "tensor_kernel_scan_innermost_dim":
        return "aten::scan"
    if lower == "l2norm_fwd_kernel2":
        return "aten::linalg_vector_norm"
    if lower == "fill_reverse_indices_kernel":
        return "aten::unique"
    if lower == "indexselectsmallindex":
        return "aten::index_select"
    if lower == "lamport_initialize_kernel":
        return "custom::lamport_initialize"
    if lower == "compute_global_hist_kernel":
        return "aten::histogram"
    if lower.startswith("le_func_"):
        return "aten::le"
    if lower.startswith("lt_func_"):
        return "aten::lt"
    if lower.startswith("gt_func_"):
        return "aten::gt"
    if lower.startswith("full_func_scalar"):
        return "aten::full"
    if lower.startswith("bitwise_not_func"):
        return "aten::bitwise_not"
    if lower.startswith("sum_kernel"):
        return "aten::sum"
    if lower.startswith("reciprocal_func"):
        return "aten::reciprocal"

    rules = [
        ("all_reduce", "aten::all_reduce"),
        ("allgather", "aten::all_gather"),
        ("broadcast", "aten::broadcast"),
        ("flash_fwd", "aten::scaled_dot_product_attention"),
        ("fmha", "aten::scaled_dot_product_attention"),
        ("mm_kernel", "aten::mm"),
        ("addmm", "aten::addmm"),
        ("gemm", "aten::mm"),
        ("gemv", "aten::mv"),
        ("bmm", "aten::bmm"),
        ("layer_norm", "aten::layer_norm"),
        ("rms_norm", "aten::rms_norm"),
        ("softmax", "aten::softmax"),
        ("silu", "aten::silu"),
        ("gelu", "aten::gelu"),
        ("relu", "aten::relu"),
        ("tanh", "aten::tanh"),
        ("exp", "aten::exp"),
        ("sin", "aten::sin"),
        ("cos", "aten::cos"),
        ("pow", "aten::pow"),
        ("copy", "aten::copy_"),
        ("scatter", "aten::scatter"),
        ("gather", "aten::gather"),
        ("embedding", "aten::embedding"),
        ("where", "aten::where"),
        ("masked_fill", "aten::masked_fill"),
        ("reduce", "aten::reduce"),
        ("max_kernel", "aten::max"),
        ("min_kernel", "aten::min"),
        ("conv", "aten::conv2d"),
        ("pad", "aten::pad"),
        ("elementwise", "aten::elementwise"),
    ]

    for pattern, aten in rules:
        if pattern in lower:
            return aten

    # 避免子串误匹配（例如 multimem -> mul, scatter -> cat）
    if "catarray" in lower or lower.startswith("cat") or "_cat_" in lower:
        return "aten::cat"
    if re.search(r"(^|_)mul(_|$)", lower):
        return "aten::mul"
    if re.search(r"(^|_)add(_|$)", lower):
        return "aten::add"
    if re.search(r"(^|_)sub(_|$)", lower):
        return "aten::sub"
    if re.search(r"(^|_)div(_|$)", lower):
        return "aten::div"

    if re.search(r"\bmm\b", lower):
        return "aten::mm"
    return "unknown"


def query_kernel_stats(conn: sqlite3.Connection) -> List[KernelStat]:
    rows = conn.execute(
        """
        SELECT COALESCE(s.value, CAST(k.shortName AS TEXT)) AS kernel_name,
               COUNT(*) AS calls,
               SUM(k.end - k.start) AS total_ns,
               AVG(k.end - k.start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON s.id = k.shortName
        GROUP BY k.shortName
        ORDER BY total_ns DESC
        """
    ).fetchall()

    agg: Dict[str, Dict[str, float]] = {}
    for name, calls, total_ns, _avg_ns in rows:
        raw_name = str(name)
        norm_name = normalize_kernel_name(raw_name)
        if norm_name not in agg:
            agg[norm_name] = {"calls": 0.0, "total_ns": 0.0}
        agg[norm_name]["calls"] += float(calls or 0)
        agg[norm_name]["total_ns"] += float(total_ns or 0)

    out: List[KernelStat] = []
    for norm_name, values in agg.items():
        calls = int(values["calls"])
        total_ns = int(values["total_ns"])
        avg_ns = (total_ns / calls) if calls > 0 else 0.0
        out.append(
            KernelStat(
                name=norm_name,
                aten_op=infer_aten_op(norm_name),
                calls=calls,
                total_ns=total_ns,
                avg_ns=avg_ns,
            )
        )

    out.sort(key=lambda x: x.total_ns, reverse=True)
    return out


def extract_input_shape(kernel_name: str) -> str:
    lower = kernel_name.lower()

    tiles = re.search(r"tilesize(\d+x\d+x\d+)", lower)
    if tiles:
        return tiles.group(1)

    parts = kernel_name.split("_")
    shape_parts: List[str] = []
    for part in parts:
        if re.fullmatch(r"\d+x\d+(x\d+)?", part):
            shape_parts.append(part)

    if shape_parts:
        return ", ".join(shape_parts)

    return "N/A"


def query_mm_shape_stats(conn: sqlite3.Connection) -> List[MmKernelShapeStat]:
    rows = conn.execute(
        """
        SELECT COALESCE(s.value, CAST(k.shortName AS TEXT)) AS kernel_name,
               COUNT(*) AS calls,
               SUM(k.end - k.start) AS total_ns,
               k.gridX,
               k.gridY,
               k.gridZ,
               k.blockX,
               k.blockY,
               k.blockZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON s.id = k.shortName
        GROUP BY k.shortName, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
        ORDER BY total_ns DESC
        """
    ).fetchall()

    agg: Dict[str, Dict[str, object]] = {}
    for name, calls, total_ns, gx, gy, gz, bx, by, bz in rows:
        exec_name = str(name)
        normalized = normalize_kernel_name(exec_name)
        framework_op = infer_aten_op(normalized)
        if framework_op != "aten::mm":
            continue

        if exec_name not in agg:
            agg[exec_name] = {
                "framework_op": framework_op,
                "calls": 0,
                "total_ns": 0,
                "shape": extract_input_shape(exec_name),
                "launch_cfg": {},
            }

        agg[exec_name]["calls"] = int(agg[exec_name]["calls"]) + int(calls or 0)
        agg[exec_name]["total_ns"] = int(agg[exec_name]["total_ns"]) + int(total_ns or 0)

        cfg = f"grid({int(gx)},{int(gy)},{int(gz)}) block({int(bx)},{int(by)},{int(bz)})"
        launch_cfg = agg[exec_name]["launch_cfg"]
        if isinstance(launch_cfg, dict):
            launch_cfg[cfg] = int(launch_cfg.get(cfg, 0)) + int(calls or 0)

    out: List[MmKernelShapeStat] = []
    for exec_name, values in agg.items():
        shape_value = str(values["shape"])
        if shape_value == "N/A":
            launch_cfg = values["launch_cfg"]
            if isinstance(launch_cfg, dict) and launch_cfg:
                top_cfg = sorted(launch_cfg.items(), key=lambda x: x[1], reverse=True)[:2]
                shape_value = "; ".join(cfg for cfg, _ in top_cfg)

        out.append(
            MmKernelShapeStat(
                framework_op=str(values["framework_op"]),
                exec_op=exec_name,
                input_shape=shape_value,
                calls=int(values["calls"]),
                total_ns=int(values["total_ns"]),
            )
        )

    out.sort(key=lambda x: x.total_ns, reverse=True)
    return out


def ns_to_ms(ns: float) -> float:
    return ns / 1e6


def ns_to_us(ns: float) -> float:
    return ns / 1e3


def fmt_ms(ns: float) -> str:
    return f"{ns_to_ms(ns):.3f}"


def fmt_us(ns: float) -> str:
    return f"{ns_to_us(ns):.3f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def write_excel_tables(excel_path: Path, tables: List[Dict[str, object]]) -> None:
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    for table in tables:
        sheet_name = str(table["sheet_name"])[:31]
        headers = table["headers"]
        rows = table["rows"]

        sheet = workbook.create_sheet(title=sheet_name)
        sheet.append(headers)
        for row in rows:
            sheet.append(row)

    workbook.save(excel_path)


def speedup(cuda_total_ns: int, gems_total_ns: int) -> float:
    if gems_total_ns <= 0:
        return math.inf if cuda_total_ns > 0 else 1.0
    return cuda_total_ns / gems_total_ns


def fmt_speedup(v: float) -> str:
    if math.isinf(v):
        return "∞"
    return f"{v:.3f}"


def aggregate_by_aten(stats: List[KernelStat]) -> List[AtenAggStat]:
    agg: Dict[str, Dict[str, object]] = {}
    for item in stats:
        if item.aten_op not in agg:
            agg[item.aten_op] = {"names": set(), "calls": 0, "total_ns": 0}
        names = agg[item.aten_op]["names"]
        if isinstance(names, set):
            names.add(item.name)
        agg[item.aten_op]["calls"] = int(agg[item.aten_op]["calls"]) + item.calls
        agg[item.aten_op]["total_ns"] = int(agg[item.aten_op]["total_ns"]) + item.total_ns

    out: List[AtenAggStat] = []
    for aten_op, values in agg.items():
        names_obj = values["names"]
        names_sorted = sorted(names_obj) if isinstance(names_obj, set) else []
        calls = int(values["calls"])
        total_ns = int(values["total_ns"])
        avg_ns = (total_ns / calls) if calls > 0 else 0.0
        out.append(
            AtenAggStat(
                aten_op=aten_op,
                kernel_names=names_sorted,
                calls=calls,
                total_ns=total_ns,
                avg_ns=avg_ns,
            )
        )

    out.sort(key=lambda x: x.total_ns, reverse=True)
    return out


def build_compare_rows(
    cuda_stats: List[KernelStat],
    gems_stats: List[KernelStat],
    sort_by: str,
) -> List[List[str]]:
    cuda_aten_stats = aggregate_by_aten(cuda_stats)
    gems_aten_stats = aggregate_by_aten(gems_stats)

    cuda_map: Dict[str, AtenAggStat] = {k.aten_op: k for k in cuda_aten_stats}
    gems_map: Dict[str, AtenAggStat] = {k.aten_op: k for k in gems_aten_stats}

    cuda_total_ns = sum(k.total_ns for k in cuda_stats)
    gems_total_ns = sum(k.total_ns for k in gems_stats)

    atens = set(cuda_map.keys()) | set(gems_map.keys())

    def cuda_total(aten: str) -> int:
        return cuda_map[aten].total_ns if aten in cuda_map else 0

    def by_speedup(aten: str) -> float:
        c = cuda_total(aten)
        g = gems_map[aten].total_ns if aten in gems_map else 0
        return speedup(c, g)

    if sort_by == "cuda_total_desc":
        ordered = sorted(atens, key=lambda a: (cuda_total(a), by_speedup(a)), reverse=True)
    elif sort_by == "speedup_desc":
        ordered = sorted(
            atens,
            key=lambda a: (math.isinf(by_speedup(a)), by_speedup(a), cuda_total(a)),
            reverse=True,
        )
    else:
        raise ValueError(f"unsupported sort mode: {sort_by}")

    rows: List[List[str]] = []
    for aten in ordered:
        c = cuda_map.get(aten)
        g = gems_map.get(aten)
        c_calls = c.calls if c else 0
        c_total = c.total_ns if c else 0
        c_pct = (c_total / cuda_total_ns) if cuda_total_ns else 0.0

        g_calls = g.calls if g else 0
        g_total = g.total_ns if g else 0
        g_pct = (g_total / gems_total_ns) if gems_total_ns else 0.0

        # 仅统计 CUDA 和 FlagGems 都有调用的kernel
        if c_calls == 0 or g_calls == 0:
            continue

        name_set: Set[str] = set()
        if c:
            name_set.update(c.kernel_names)
        if g:
            name_set.update(g.kernel_names)
        merged_names = ", ".join(sorted(name_set))

        aten_name = c.aten_op if c else (g.aten_op if g else "unknown")
        sp = speedup(c_total, g_total)

        rows.append(
            [
                aten_name,
                md_escape(merged_names),
                str(c_calls),
                fmt_ms(c_total),
                fmt_pct(c_pct),
                str(g_calls),
                fmt_ms(g_total),
                fmt_pct(g_pct),
                fmt_speedup(sp),
            ]
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cuda_db = root / "nsys-cuda" / "report-cuda.sqlite"
    gems_db = root / "nsys-gems" / "report-gems-all.sqlite"
    out_md = root / "reports" / "perf_analysis.md"
    out_xlsx = root / "reports" / "perf_analysis.xlsx"
    out_shape_md = root / "processing" / "shape_analysis.md"
    out_shape_xlsx = root / "reports" / "shape_analysis.xlsx"

    if not cuda_db.exists() or not gems_db.exists():
        raise SystemExit(f"缺少sqlite文件: {cuda_db} 或 {gems_db}")

    with sqlite3.connect(str(cuda_db)) as cuda_conn, sqlite3.connect(str(gems_db)) as gems_conn:
        cuda_stats = query_kernel_stats(cuda_conn)
        gems_stats = query_kernel_stats(gems_conn)
        cuda_mm_shape_stats = query_mm_shape_stats(cuda_conn)
        gems_mm_shape_stats = query_mm_shape_stats(gems_conn)

    cuda_aten_stats = aggregate_by_aten(cuda_stats)
    gems_aten_stats = aggregate_by_aten(gems_stats)

    cuda_total_ns = sum(k.total_ns for k in cuda_stats)
    gems_total_ns = sum(k.total_ns for k in gems_stats)

    report: List[str] = []
    report.append("# FlagGems 和 CUDA 算子库性能对比分析")
    report.append("")

    report.append("## CUDA kernel（按总时间排序）")
    report.append("")
    cuda_headers = ["框架算子名", "执行算子名", "调用次数", "总时间(ms)", "平均时间(μs)", "占比"]
    cuda_rows: List[List[str]] = []
    for k in cuda_aten_stats:
        cuda_rows.append(
            [
                k.aten_op,
                md_escape(", ".join(k.kernel_names)),
                str(k.calls),
                fmt_ms(k.total_ns),
                fmt_us(k.avg_ns),
                fmt_pct(k.total_ns / cuda_total_ns if cuda_total_ns else 0.0),
            ]
        )
    report.append(md_table(cuda_headers, cuda_rows))
    report.append("")

    report.append("## FlagGems kernel（按总时间排序）")
    report.append("")
    gems_headers = ["框架算子名", "执行算子名", "调用次数", "总时间(ms)", "平均时间(μs)", "占比"]
    gems_rows: List[List[str]] = []
    for k in gems_aten_stats:
        gems_rows.append(
            [
                k.aten_op,
                md_escape(", ".join(k.kernel_names)),
                str(k.calls),
                fmt_ms(k.total_ns),
                fmt_us(k.avg_ns),
                fmt_pct(k.total_ns / gems_total_ns if gems_total_ns else 0.0),
            ]
        )
    report.append(md_table(gems_headers, gems_rows))
    report.append("")

    report.append("## CUDA 和 FlagGems kernel 对比（按 CUDA 总时间排序）")
    report.append("")
    compare_cuda_rows = build_compare_rows(cuda_stats, gems_stats, sort_by="cuda_total_desc")
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
    report.append(
        md_table(
            compare_headers,
            compare_cuda_rows,
        )
    )
    report.append("")

    report.append("## CUDA 和 FlagGems kernel 对比（按加速比从高到低）")
    report.append("")
    compare_speedup_rows = build_compare_rows(cuda_stats, gems_stats, sort_by="speedup_desc")
    report.append(
        md_table(
            compare_headers,
            compare_speedup_rows,
        )
    )
    report.append("")

    write_excel_tables(
        out_xlsx,
        [
            {"sheet_name": "CUDA按总时间", "headers": cuda_headers, "rows": cuda_rows},
            {"sheet_name": "FlagGems按总时间", "headers": gems_headers, "rows": gems_rows},
            {"sheet_name": "对比按CUDA总时间", "headers": compare_headers, "rows": compare_cuda_rows},
            {"sheet_name": "对比按加速比", "headers": compare_headers, "rows": compare_speedup_rows},
        ],
    )

    out_md.write_text("\n".join(report), encoding="utf-8")

    cuda_mm_rows: List[List[str]] = []
    for row in cuda_mm_shape_stats:
        cuda_mm_rows.append(
            [
                row.framework_op,
                md_escape(row.exec_op),
                row.input_shape,
                str(row.calls),
                fmt_ms(row.total_ns),
                fmt_pct(row.total_ns / cuda_total_ns if cuda_total_ns else 0.0),
            ]
        )

    gems_mm_rows: List[List[str]] = []
    for row in gems_mm_shape_stats:
        gems_mm_rows.append(
            [
                row.framework_op,
                md_escape(row.exec_op),
                row.input_shape,
                str(row.calls),
                fmt_ms(row.total_ns),
                fmt_pct(row.total_ns / gems_total_ns if gems_total_ns else 0.0),
            ]
        )

    shape_report: List[str] = []
    shape_report.append("# CUDA 和 FlagGems 的 aten::mm 输入 Shape 分析")
    shape_report.append("")
    shape_report.append("## CUDA aten::mm 算子（按总时间排序）")
    shape_report.append("")
    shape_report.append(
        md_table(
            ["框架算子名", "执行算子名", "输入 shape", "CUDA调用次数", "CUDA总时间(ms)", "CUDA占比"],
            cuda_mm_rows,
        )
    )
    shape_report.append("")

    shape_report.append("## FlagGems aten::mm 算子（按总时间排序）")
    shape_report.append("")
    shape_report.append(
        md_table(
            ["框架算子名", "执行算子名", "输入 shape", "FlagGems调用次数", "FlagGems总时间(ms)", "FlagGems占比"],
            gems_mm_rows,
        )
    )
    shape_report.append("")

    out_shape_md.write_text("\n".join(shape_report), encoding="utf-8")

    shape_cuda_headers = ["框架算子名", "执行算子名", "输入 shape", "CUDA调用次数", "CUDA总时间(ms)", "CUDA占比"]
    shape_gems_headers = ["框架算子名", "执行算子名", "输入 shape", "FlagGems调用次数", "FlagGems总时间(ms)", "FlagGems占比"]
    write_excel_tables(
        out_shape_xlsx,
        [
            {"sheet_name": "CUDA_mm_shape", "headers": shape_cuda_headers, "rows": cuda_mm_rows},
            {"sheet_name": "FlagGems_mm_shape", "headers": shape_gems_headers, "rows": gems_mm_rows},
        ],
    )

    print(f"已生成报告: {out_md}")
    print(f"已生成Excel: {out_xlsx}")
    print(f"已生成Shape分析: {out_shape_md}")
    print(f"已生成Shape Excel: {out_shape_xlsx}")
    print(f"CUDA kernels: {len(cuda_stats)}, FlagGems kernels: {len(gems_stats)}")


if __name__ == "__main__":
    main()
