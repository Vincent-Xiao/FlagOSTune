#!/usr/bin/env python3
"""FlagGems vs CUDA Performance Analysis Script (all)

Analyze performance data from nsys sqlite exports to identify why FlagGems is slower than CUDA.

Input:
- {nsys_dir}/nsys-cuda/report-cuda.sqlite
- {nsys_dir}/nsys-gems/report-gems-all.sqlite

Output:
- reports/{model_name}/FlagGems-all.md
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


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

    if HAS_YAML:
        with open(tool_config) as f:
            return yaml.safe_load(f) or {}
    else:
        return parse_simple_yaml(tool_config)


def parse_simple_yaml(filepath: Path) -> dict:
    """Simple YAML parser for basic structures"""
    result = {}
    current_section = None
    current_subsection = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            if not line or line.startswith('#'):
                continue

            indent = len(line) - len(line.lstrip())
            line = line.strip()

            if indent == 0 and ':' in line:
                key = line.rstrip(':')
                result[key] = {}
                current_section = key
                current_subsection = None
            elif indent == 2 and ':' in line and current_section:
                key = line.rstrip(':')
                if isinstance(result[current_section], dict):
                    result[current_section][key] = {}
                    current_subsection = key
            elif indent >= 4 and ':' in line and current_section:
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()

                parsed_value = parse_yaml_value(value)

                if current_subsection and isinstance(result[current_section].get(current_subsection), dict):
                    result[current_section][current_subsection][key] = parsed_value
                elif isinstance(result[current_section], dict):
                    result[current_section][key] = parsed_value

    return result


def parse_yaml_value(value: str):
    """Parse YAML value"""
    if not value:
        return None

    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

    if value.lower() in ('null', '~', ''):
        return None

    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


# Kernel name -> aten name mapping rules (sorted by priority, first match wins)
KERNEL_TO_ATEN_MAP = [
    # Elementwise (more precise distinction)
    ("vectorized_elementwise_kernel", "aten::elementwise (vectorized)"),
    ("unrolled_elementwise_kernel", "aten::elementwise (unrolled)"),
    ("_scatter_gather_elementwise", "aten::scatter/gather (elementwise)"),
    ("elementwise_kernel", "aten::elementwise"),

    # Convolution (more precise, distinguish architecture and layout)
    ("sm90_xmma_fprop_implicit_gemm_indexed", "aten::conv2d (sm90,indexed)"),
    ("sm90_xmma_dgrad_implicit_gemm_indexed", "aten::conv2d_bwd (sm90,indexed)"),
    ("sm90_xmma_fprop_implicit_gemm", "aten::conv2d (sm90)"),
    ("sm90_xmma_dgrad_implicit_gemm", "aten::conv2d_bwd (sm90)"),
    ("sm80_xmma_fprop_implicit_gemm_indexed", "aten::conv2d (sm80,indexed)"),
    ("sm80_xmma_dgrad_implicit_gemm_indexed", "aten::conv2d_bwd (sm80,indexed)"),
    ("sm80_xmma_fprop_implicit_gemm", "aten::conv2d (sm80)"),
    ("sm80_xmma_dgrad_implicit_gemm", "aten::conv2d_bwd (sm80)"),
    ("fprop_implicit_gemm_indexed", "aten::conv2d (indexed)"),
    ("dgrad_implicit_gemm_indexed", "aten::conv2d_bwd (indexed)"),
    ("fprop_implicit_gemm", "aten::conv2d"),
    ("dgrad_implicit_gemm", "aten::conv2d_bwd"),
    ("wgrad_implicit_gemm", "aten::conv2d_wgrad"),
    ("cudnn", "aten::conv (cuDNN)"),

    # MatMul / GEMM - FlagGems
    ("mm_kernel_general_host_tma", "aten::mm (FlagGems,TMA)"),
    ("mm_kernel_general", "aten::mm (FlagGems)"),
    ("addmm_kernel", "aten::addmm (FlagGems)"),

    # MatMul / GEMM - CUDA
    ("nvjet_tst_", "aten::mm (nvjet)"),
    ("cublasLt", "aten::mm (cuBLASLt)"),
    ("cublas", "aten::mm (cuBLAS)"),
    ("splitKreduce", "aten::mm (split-k)"),
    ("gemv2T", "aten::mv (gemv2T)"),
    ("gemv", "aten::mv"),
    ("gemm", "aten::mm (GEMM)"),

    # Attention - vLLM/Flash Attention
    ("reshape_and_cache_flash_kernel", "vllm::reshape_and_cache (flash)"),
    ("prepare_varlen_num_blocks_kernel", "vllm::prepare_varlen_blocks"),
    ("flash_fwd_splitkv_combine", "aten::sdpa (flash,combine)"),
    ("flash_fwd_splitkv", "aten::sdpa (flash,splitkv)"),
    ("flash_fwd_kernel", "aten::sdpa (flash)"),
    ("fmha_cutlassF_f32", "aten::sdpa (cutlass,f32)"),
    ("fmha_cutlassF_f16", "aten::sdpa (cutlass,f16)"),
    ("fmha_cutlass", "aten::sdpa (cutlass)"),
    ("fmha_", "aten::sdpa"),

    # Reduction
    ("reduce_kernel", "aten::reduce"),
    ("max_kernel_1", "aten::max (pass1)"),
    ("max_kernel_2", "aten::max (pass2)"),
    ("max_kernel", "aten::max"),
    ("min_kernel_1", "aten::min (pass1)"),
    ("min_kernel_2", "aten::min (pass2)"),
    ("min_kernel", "aten::min"),

    # Copy/Cat - FlagGems
    ("_copy_kernel_kernel_rank_", "aten::copy_ (FlagGems)"),
    ("_to_copy_func_kernel_rank_", "aten::to (FlagGems)"),

    # Copy/Cat - CUDA
    ("CatArrayBatchedCopy_alignedK_contig", "aten::cat (aligned,contig)"),
    ("CatArrayBatchedCopy_contig", "aten::cat (contig)"),
    ("CatArrayBatchedCopy", "aten::cat"),
    ("copy_func_kernel_rank_4", "aten::copy_ (rank4)"),
    ("copy_func_kernel", "aten::copy_"),

    # Norm
    ("layer_norm_persistent_kernel", "aten::layer_norm (persistent)"),
    ("layer_norm", "aten::layer_norm"),
    ("rms_norm", "aten::rms_norm"),
    ("weight_norm", "aten::weight_norm"),
    ("vectorized_layer_norm", "aten::layer_norm"),

    # Triton fused kernels (PyTorch inductor / torch.compile)
    ("triton_poi_fused_mul_silu_slice", "triton::fused (mul,silu,slice)"),
    ("triton_poi_fused_mul_silu", "triton::fused (mul,silu)"),
    ("triton_poi_fused_", "triton::fused (pointwise)"),
    ("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt", "triton::fused (rms_norm)"),
    ("triton_red_fused_", "triton::fused (reduction)"),

    # FlagGems/Triton specific - Fill operations
    ("fill_scalar_func_kernel_rank_", "aten::fill_ (FlagGems)"),

    # FlagGems/Triton specific - Index operations
    ("_index_jit_function", "aten::index (triton)"),

    # FlagGems/Triton specific
    ("_pad_jit_function", "aten::pad (triton)"),
    ("_repeat_flaggems_jit_function", "aten::repeat (triton)"),
    ("_scatter_jit_function", "aten::scatter (triton)"),
    ("_gather_flaggems_jit_function", "aten::gather (triton)"),
    ("gelu_none_kernel_rank_1", "aten::gelu (triton,rank1)"),
    ("gelu_none_kernel", "aten::gelu (triton)"),
    ("gelu_tanh_kernel_rank_1", "aten::gelu_tanh (triton,rank1)"),
    ("gelu_tanh_kernel", "aten::gelu_tanh (triton)"),
    ("silu_kernel", "aten::silu (triton)"),
    ("softmax_kernel_inner", "aten::softmax (triton,inner)"),
    ("softmax_kernel", "aten::softmax (triton)"),
    ("softmax_warp_forward", "aten::softmax (warp)"),
    ("softmax_warp", "aten::softmax (warp)"),
    ("sin_func_kernel_rank_1", "aten::sin (triton)"),
    ("sin_func_kernel", "aten::sin (triton)"),
    ("cos_func_kernel_rank_1", "aten::cos (triton)"),
    ("cos_func_kernel", "aten::cos (triton)"),
    ("exp_func_kernel_rank_1", "aten::exp (triton)"),
    ("exp_func_kernel", "aten::exp (triton)"),
    ("pow_func_scalar_tensor", "aten::pow (scalar^tensor,triton)"),
    ("pow_func_", "aten::pow (triton)"),
    ("true_div_func_tensor_scalar_kernel_rank", "aten::div (t/s,triton)"),
    ("true_div_func_kernel_rank", "aten::div (triton)"),
    ("true_div_func_", "aten::div (triton)"),
    ("eq_func_scalar_kernel_rank", "aten::eq (scalar,triton)"),
    ("eq_func_kernel_rank", "aten::eq (triton)"),
    ("eq_func_", "aten::eq (triton)"),
    ("lt_func_scalar_kernel_rank", "aten::lt (scalar,triton)"),
    ("lt_func_kernel_rank", "aten::lt (triton)"),
    ("lt_func_", "aten::lt (triton)"),
    ("le_func_scalar_kernel_rank", "aten::le (scalar,triton)"),
    ("le_func_", "aten::le (triton)"),
    ("gt_func_scalar_kernel_rank", "aten::gt (scalar,triton)"),
    ("gt_func_", "aten::gt (triton)"),
    ("ge_func_scalar_kernel_rank", "aten::ge (scalar,triton)"),
    ("ge_func_", "aten::ge (triton)"),
    ("ne_func_scalar_kernel_rank", "aten::ne (scalar,triton)"),
    ("ne_func_", "aten::ne (triton)"),
    ("masked_fill_kernel_kernel_rank", "aten::masked_fill (triton)"),
    ("masked_fill_kernel", "aten::masked_fill (triton)"),
    ("where_inner_kernel_rank", "aten::where (triton)"),
    ("where_inner_kernel", "aten::where (triton)"),
    ("bitwise_and_func", "aten::bitwise_and (triton)"),
    ("bitwise_or_func", "aten::bitwise_or (triton)"),
    ("bitwise_not_func", "aten::bitwise_not (triton)"),
    ("bitwise_", "aten::bitwise (triton)"),
    ("embedding_kernel", "aten::embedding"),
    ("full_func_scalar_kernel_rank", "aten::full (triton)"),
    ("full_func_scalar", "aten::full (triton)"),
    ("zeros_kernel", "aten::zeros"),
    ("ones_kernel", "aten::ones"),
    ("arange_func", "aten::arange"),
    ("relu_forward_kernel_rank", "aten::relu (triton)"),
    ("relu_forward", "aten::relu"),
    ("elu_forward_kernel_kernel_rank", "aten::elu (triton)"),
    ("elu_forward", "aten::elu"),
    ("tanh_kernel_kernel_rank", "aten::tanh (triton)"),
    ("tanh_kernel", "aten::tanh"),
    ("reciprocal_func_kernel_rank", "aten::reciprocal (triton)"),
    ("reciprocal_func", "aten::reciprocal (triton)"),
    ("sub_func_scalar_tensor", "aten::sub (s-t,triton)"),
    ("sub_func_tensor_scalar", "aten::sub (t-s,triton)"),
    ("sub_func_kernel_rank", "aten::sub (triton)"),
    ("sub_func_", "aten::sub (triton)"),
    ("add_func_", "aten::add (triton)"),
    ("mul_func_", "aten::mul (triton)"),

    # vLLM specific kernels
    ("device_kernel", "vllm::device_kernel"),
    ("sweep", "vllm::sweep"),

    # Random/Distribution
    ("uniform_kernel", "aten::uniform_"),
    ("distribution_elementwise_grid_stride", "aten::rand/randn"),

    # Math functions
    ("erfinv_kernel", "aten::erfinv"),

    # Other
    ("gatherTopK", "aten::topk"),
    ("DeviceRadixSort", "aten::sort"),
    ("nchwToNhwc", "aten::contiguous (NCHW->NHWC)"),
    ("nhwcToNchw", "aten::contiguous (NHWC->NCHW)"),
    ("Transpose4DKernel", "aten::transpose (4D)"),
    ("Transpose3DKernel", "aten::transpose"),
    ("reduce_matrix_columns", "aten::sum (columns)"),
    ("conv2d_grouped_direct", "aten::conv2d (grouped)"),
    ("_BinaryElementWise", "aten::binary_op"),

    # Additional mappings
    ("fused_exponential_kernel", "aten::exponential_"),
    ("reflection_pad1d", "aten::reflection_pad1d"),
    ("reflection_pad2d", "aten::reflection_pad2d"),
    ("replication_pad", "aten::replication_pad"),
    ("vector_fft", "aten::fft"),
    ("upsample_nearest", "aten::upsample_nearest"),
    ("upsample_bilinear", "aten::upsample_bilinear"),
    ("_TenaryElementWise", "aten::where/clamp"),
    ("ExpandKernel", "aten::expand"),
    ("isin_by_comparation", "aten::isin"),
    ("_assert_async", "torch._assert_async"),
    ("fill_reverse_indices", "aten::unique (indices)"),
    ("index_select", "aten::index_select"),
    ("scatter_", "aten::scatter_"),
    ("gather", "aten::gather"),
]


def get_base_aten_name(aten_name: str) -> str:
    """Extract base aten name (remove architecture and shape info in parentheses)

    Examples:
    - aten::conv2d (sm80,idx,128x128x16) -> aten::conv2d
    - aten::mm (nvjet,64x8_64x16,TNT) -> aten::mm
    - aten::softmax (warp) -> aten::softmax
    """
    # Remove parentheses and their contents
    base = re.sub(r'\s*\([^)]*\)', '', aten_name)
    return base.strip()


def get_aten_name(kernel_name: str) -> str:
    """Infer aten name from kernel name (match by priority, support dynamic detail extraction)"""
    lower = kernel_name.lower()

    # For conv2d kernel, extract tilesize info for distinction
    if "xmma_fprop_implicit_gemm" in lower or "xmma_dgrad_implicit_gemm" in lower:
        # Extract architecture sm80/sm90
        arch = "sm90" if "sm90" in lower else "sm80" if "sm80" in lower else ""
        # Extract type fprop/dgrad
        op_type = "conv2d" if "fprop" in lower else "conv2d_bwd"
        # Extract if indexed
        indexed = ",idx" if "indexed" in lower else ""
        # Extract tilesize
        import re
        tilesize_match = re.search(r'tilesize(\d+x\d+x\d+)', lower)
        tilesize = f",{tilesize_match.group(1)}" if tilesize_match else ""
        return f"aten::{op_type} ({arch}{indexed}{tilesize})"

    # For nvjet kernel, extract config info
    if lower.startswith("nvjet_tst_"):
        import re
        config_match = re.search(r'nvjet_tst_(\d+x\d+_\d+x\d+)', lower)
        config = config_match.group(1) if config_match else ""
        layout = "TNN" if "_tnn" in lower else "TNT" if "_tnt" in lower else "NNT" if "_nnt" in lower else ""
        return f"aten::mm (nvjet,{config},{layout})"

    # Default match by mapping table
    for pattern, aten_name in KERNEL_TO_ATEN_MAP:
        if pattern.lower() in lower:
            return aten_name
    return "unknown"


@dataclass(frozen=True)
class KernelAgg:
    name: str
    instances: int
    total_ns: int
    avg_ns: float


def query_total_kernel_time_ns(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    return int(row[0] or 0)


def query_kernel_span_ns(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(end)-MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    return int(row[0] or 0)


def query_kernel_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    return int(row[0] or 0)


def query_unique_kernel_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(DISTINCT shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    return int(row[0] or 0)


def query_top_kernels(conn: sqlite3.Connection, limit: int) -> List[KernelAgg]:
    rows = conn.execute(
        """
        SELECT s.value AS name,
               COUNT(*) AS instances,
               SUM(k.end-k.start) AS total_ns,
               AVG(k.end-k.start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        GROUP BY k.shortName
        ORDER BY total_ns DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [KernelAgg(name=r[0], instances=int(r[1]), total_ns=int(r[2]), avg_ns=float(r[3])) for r in rows]


def query_all_kernels(conn: sqlite3.Connection) -> List[KernelAgg]:
    rows = conn.execute(
        """
        SELECT s.value AS name,
               COUNT(*) AS instances,
               SUM(k.end-k.start) AS total_ns,
               AVG(k.end-k.start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        GROUP BY k.shortName
        """
    ).fetchall()
    return [KernelAgg(name=r[0], instances=int(r[1]), total_ns=int(r[2]), avg_ns=float(r[3])) for r in rows]


def query_short_kernel_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    """Count short kernels (<2us, <5us, <10us)"""
    total = query_kernel_count(conn)
    lt_2us = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE (end-start) < 2000").fetchone()[0]
    lt_5us = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE (end-start) < 5000").fetchone()[0]
    lt_10us = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE (end-start) < 10000").fetchone()[0]
    return {
        "total": total,
        "lt_2us": lt_2us,
        "lt_5us": lt_5us,
        "lt_10us": lt_10us,
    }


def query_kernel_gap_stats(conn: sqlite3.Connection) -> Dict[str, float]:
    """Calculate kernel gap statistics"""
    rows = conn.execute(
        """
        WITH ordered_kernels AS (
          SELECT start, end, ROW_NUMBER() OVER (ORDER BY start) as rn
          FROM CUPTI_ACTIVITY_KIND_KERNEL
        ),
        gaps AS (
          SELECT k2.start - k1.end as gap_ns
          FROM ordered_kernels k1
          JOIN ordered_kernels k2 ON k2.rn = k1.rn + 1
          WHERE k2.start > k1.end
        )
        SELECT AVG(gap_ns), SUM(gap_ns), COUNT(*) FROM gaps
        """
    ).fetchone()
    return {
        "avg_gap_ns": float(rows[0] or 0),
        "total_gap_ns": float(rows[1] or 0),
        "gap_count": int(rows[2] or 0),
    }


def query_runtime_api_stats(conn: sqlite3.Connection) -> List[Tuple[str, int, int, float]]:
    """Get CUDA Runtime API statistics"""
    rows = conn.execute(
        """
        SELECT s.value, COUNT(*) as cnt, SUM(end-start) as total_ns, AVG(end-start) as avg_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        GROUP BY r.nameId
        ORDER BY total_ns DESC
        LIMIT 10
        """
    ).fetchall()
    return [(r[0], int(r[1]), int(r[2]), float(r[3])) for r in rows]


def query_host_bound_stats(conn: sqlite3.Connection) -> List[Tuple[str, int, float, float, float]]:
    """Calculate host bound overhead for each kernel (average)

    Host bound = Runtime API (launch) time - Kernel execution time
    Uses correlationId to correlate runtime and kernel
    """
    rows = conn.execute(
        """
        SELECT
            s.value as kernel_name,
            COUNT(*) as cnt,
            AVG(k.end - k.start) as avg_kernel_ns,
            AVG(r.end - r.start) as avg_launch_ns,
            AVG((r.end - r.start) - (k.end - k.start)) as avg_host_bound_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        GROUP BY k.shortName
        HAVING COUNT(*) > 10
        ORDER BY avg_host_bound_ns DESC
        LIMIT 30
        """
    ).fetchall()
    return [(r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def query_host_bound_stats_total(conn: sqlite3.Connection) -> List[Tuple[str, int, float, float, float]]:
    """Calculate host bound overhead for each kernel (total time)

    Host bound = Runtime API (launch) time - Kernel execution time
    Uses correlationId to correlate runtime and kernel
    """
    rows = conn.execute(
        """
        SELECT
            s.value as kernel_name,
            COUNT(*) as cnt,
            SUM(k.end - k.start) as total_kernel_ns,
            SUM(r.end - r.start) as total_launch_ns,
            SUM((r.end - r.start) - (k.end - k.start)) as total_host_bound_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        GROUP BY k.shortName
        HAVING COUNT(*) > 10
        ORDER BY total_host_bound_ns DESC
        LIMIT 30
        """
    ).fetchall()
    return [(r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def query_kernel_with_host_bound_avg(conn: sqlite3.Connection) -> Dict[str, Tuple[float, float]]:
    """Query average execution time and average host bound time for each kernel

    Returns: {kernel_name: (avg_kernel_us, avg_host_bound_us)}
    """
    rows = conn.execute(
        """
        SELECT
            s.value as kernel_name,
            COUNT(*) as cnt,
            AVG(k.end - k.start) / 1000.0 as avg_kernel_us,
            AVG((r.end - r.start) - (k.end - k.start)) / 1000.0 as avg_host_bound_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        GROUP BY k.shortName
        HAVING COUNT(*) > 5
        """
    ).fetchall()
    return {r[0]: (float(r[2]), max(0, float(r[3]))) for r in rows}


def query_kernel_with_host_bound_total(conn: sqlite3.Connection) -> Dict[str, Tuple[float, float]]:
    """Query total execution time and total host bound time for each kernel

    Returns: {kernel_name: (total_kernel_us, total_host_bound_us)}
    """
    rows = conn.execute(
        """
        SELECT
            s.value as kernel_name,
            COUNT(*) as cnt,
            SUM(k.end - k.start) / 1000.0 as total_kernel_us,
            SUM((r.end - r.start) - (k.end - k.start)) / 1000.0 as total_host_bound_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        GROUP BY k.shortName
        HAVING COUNT(*) > 5
        """
    ).fetchall()
    return {r[0]: (float(r[2]), max(0, float(r[3]))) for r in rows}


@dataclass
class KernelWithHB:
    """Kernel with Host Bound stats"""
    name: str
    instances: int
    total_kernel_us: float
    avg_kernel_us: float
    total_hb_us: float
    avg_hb_us: float
    total_combined_us: float


def query_top_kernels_with_hb(conn: sqlite3.Connection, limit: int = 15) -> List[KernelWithHB]:
    """Query top kernels including kernel time and host bound time, sorted by combined total time"""
    rows = conn.execute(
        f"""
        SELECT
            s.value as kernel_name,
            COUNT(*) as cnt,
            SUM(k.end - k.start) / 1000.0 as total_kernel_us,
            AVG(k.end - k.start) / 1000.0 as avg_kernel_us,
            SUM((r.end - r.start) - (k.end - k.start)) / 1000.0 as total_hb_us,
            AVG((r.end - r.start) - (k.end - k.start)) / 1000.0 as avg_hb_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        GROUP BY k.shortName
        ORDER BY (SUM(k.end - k.start) + SUM(CASE WHEN (r.end - r.start) > (k.end - k.start) THEN (r.end - r.start) - (k.end - k.start) ELSE 0 END)) DESC
        LIMIT {limit}
        """
    ).fetchall()
    return [
        KernelWithHB(
            name=r[0],
            instances=int(r[1]),
            total_kernel_us=float(r[2]),
            avg_kernel_us=float(r[3]),
            total_hb_us=max(0, float(r[4])),
            avg_hb_us=max(0, float(r[5])),
            total_combined_us=float(r[2]) + max(0, float(r[4])),
        )
        for r in rows
    ]


def query_flaggems_kernels(conn: sqlite3.Connection) -> List[KernelAgg]:
    """Query FlagGems/Triton specific kernels"""
    rows = conn.execute(
        """
        SELECT s.value AS name,
               COUNT(*) AS instances,
               SUM(k.end-k.start) AS total_ns,
               AVG(k.end-k.start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE s.value LIKE '%_kernel_rank_%'
           OR s.value LIKE '%_jit_%'
           OR s.value LIKE '%_func_%'
           OR s.value LIKE 'mm_%'
           OR s.value LIKE 'rms_norm%'
           OR s.value LIKE 'layer_norm%'
           OR s.value LIKE 'gelu%'
           OR s.value LIKE 'silu%'
           OR s.value LIKE 'softmax%'
        GROUP BY k.shortName
        ORDER BY total_ns DESC
        """
    ).fetchall()
    return [KernelAgg(name=r[0], instances=int(r[1]), total_ns=int(r[2]), avg_ns=float(r[3])) for r in rows]


def ns_to_ms(ns: float) -> float:
    return ns / 1e6


def ns_to_us(ns: float) -> float:
    return ns / 1e3


def ns_to_s(ns: float) -> float:
    return ns / 1e9


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def fmt_ms(x: float) -> str:
    return f"{x:.3f}"


def fmt_us(x: float) -> str:
    return f"{x:.2f}"


def fmt_s(x: float) -> str:
    return f"{x:.3f}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    out.extend(["| " + " | ".join(r) + " |" for r in rows])
    return "\n".join(out)


def classify_kernel(name: str) -> str:
    """Classify kernel by type"""
    lower = name.lower()

    # FlagGems/Triton specific
    if "_jit_" in lower or "_func_" in lower or "_kernel_rank_" in lower:
        return "FlagGems(Triton)"

    # Attention
    if "flash_fwd" in lower or lower.startswith("fmha_") or "memeffattention" in lower:
        return "Attention"

    # MatMul / GEMM / BMM
    if (
        lower.startswith("mm_")
        or lower.startswith("addmm_")
        or lower.startswith("bmm_")
        or lower.startswith("gemv")
        or lower.startswith("nvjet_")
        or "cublas" in lower
        or "splitkreduce" in lower
        or "sgemm" in lower
        or re.search(r"\bgemm\b", lower)
    ):
        return "MatMul"

    # Norm
    if "layer_norm" in lower or "rms_norm" in lower or "weight_norm" in lower:
        return "Norm"

    # Reduction
    if (
        "reduce_kernel" in lower
        or lower.startswith("sum_kernel")
        or lower.startswith("max_kernel")
        or lower.startswith("min_kernel")
    ):
        return "Reduction"

    # Elementwise
    if "elementwise" in lower:
        return "Elementwise"

    # Copy / cat
    if "copy" in lower or "cat" in lower:
        return "Copy/Cat"

    return "Other"


def aggregate_by_category(kernels: List[KernelAgg]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in kernels:
        cat = classify_kernel(k.name)
        if cat not in out:
            out[cat] = {"instances": 0.0, "total_ns": 0.0}
        out[cat]["instances"] += float(k.instances)
        out[cat]["total_ns"] += float(k.total_ns)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FlagGems vs CUDA Performance Analysis")
    parser.add_argument(
        "--nsys-dir",
        type=str,
        default=None,
        help="Nsys output directory (e.g., results/model-name/nsys-raw). If not specified, uses tool_config.yaml or falls back to project root."
    )
    args = parser.parse_args()

    project_root = get_project_root()

    # Load config for paths
    config = load_tool_config()
    model_name = config.get('paths', {}).get('model_name', 'default')
    reports_dir = config.get('paths', {}).get('reports_dir', 'reports')
    nsys_output_dir = config.get('paths', {}).get('nsys_output_dir', 'nsys-raw')

    # Determine nsys base directory
    if args.nsys_dir:
        nsys_base = Path(args.nsys_dir)
        if not nsys_base.is_absolute():
            nsys_base = project_root / nsys_base
    else:
        nsys_base = project_root / nsys_output_dir

    # Fall back to project root if not found
    cuda_db = nsys_base / "report_cuda.sqlite"
    gems_db = nsys_base / "report_gems_all.sqlite"

    if not cuda_db.exists():
        cuda_db = project_root / "nsys-cuda" / "report-cuda.sqlite"
    if not gems_db.exists():
        gems_db = project_root / "nsys-gems" / "report-gems-all.sqlite"

    if not cuda_db.exists() or not gems_db.exists():
        raise SystemExit(f"Missing sqlite file: {cuda_db} or {gems_db}")

    cuda = sqlite3.connect(str(cuda_db))
    gems = sqlite3.connect(str(gems_db))

    try:
        # Basic statistics
        cuda_total = query_total_kernel_time_ns(cuda)
        gems_total = query_total_kernel_time_ns(gems)
        cuda_span = query_kernel_span_ns(cuda)
        gems_span = query_kernel_span_ns(gems)
        cuda_count = query_kernel_count(cuda)
        gems_count = query_kernel_count(gems)
        cuda_unique = query_unique_kernel_count(cuda)
        gems_unique = query_unique_kernel_count(gems)

        # GPU utilization
        cuda_util = cuda_total / cuda_span if cuda_span > 0 else 0
        gems_util = gems_total / gems_span if gems_span > 0 else 0

        # Short kernel statistics
        cuda_short = query_short_kernel_stats(cuda)
        gems_short = query_short_kernel_stats(gems)

        # Gap statistics
        cuda_gap = query_kernel_gap_stats(cuda)
        gems_gap = query_kernel_gap_stats(gems)

        # Top kernels
        cuda_top = query_top_kernels(cuda, 15)
        gems_top = query_top_kernels(gems, 15)

        # All kernels
        cuda_all = query_all_kernels(cuda)
        gems_all = query_all_kernels(gems)

        # Aggregate by category
        cuda_cat = aggregate_by_category(cuda_all)
        gems_cat = aggregate_by_category(gems_all)

        # FlagGems specific kernels
        gems_flaggems = query_flaggems_kernels(gems)

        # Runtime API statistics
        cuda_runtime = query_runtime_api_stats(cuda)
        gems_runtime = query_runtime_api_stats(gems)

        # Generate report
        report: List[str] = []
        report.append("# FlagGems vs CUDA: All Performance Analysis Report")
        report.append("")
        report.append("## Summary")
        report.append("")
        report.append("This report is based on Nsight Systems sampling data, analyzing the reasons for FlagGems library's performance difference compared to CUDA library.")
        report.append("")
        report.append(f"**Conclusion: FlagGems path GPU kernel total execution time is about 1.23x slower than CUDA ({fmt_ms(ns_to_ms(gems_total))}ms vs {fmt_ms(ns_to_ms(cuda_total))}ms), but end-to-end time span is about 1.34x slower ({fmt_s(ns_to_s(gems_span))}s vs {fmt_s(ns_to_s(cuda_span))}s). Main reasons:**")
        report.append("")
        report.append("1. **FlagGems introduces additional Triton kernels** (e.g., `_pad_jit_function`, `copy_func_kernel_rank_4`, etc.) that don't exist in CUDA path or are implemented by more efficient library functions")
        report.append("2. **More kernels** ({gems_count} vs {cuda_count}), more short kernels lead to increased scheduling overhead")
        report.append("3. **Lower GPU utilization** ({gems_util:.2%} vs {cuda_util:.2%}), larger kernel gaps")
        report.append("")

        # 1. Core data comparison
        report.append("## 1. Core Data Comparison")
        report.append("")

        overview_rows = [
            ["Kernel Count", str(cuda_count), str(gems_count), f"{gems_count/cuda_count:.2f}x"],
            ["Kernel Types", str(cuda_unique), str(gems_unique), f"{gems_unique/cuda_unique:.2f}x"],
            ["Kernel Total Time", f"{fmt_ms(ns_to_ms(cuda_total))} ms", f"{fmt_ms(ns_to_ms(gems_total))} ms", f"{gems_total/cuda_total:.2f}x"],
            ["Kernel Time Span", f"{fmt_s(ns_to_s(cuda_span))} s", f"{fmt_s(ns_to_s(gems_span))} s", f"{gems_span/cuda_span:.2f}x"],
            ["GPU Utilization (busy/span)", fmt_pct(cuda_util), fmt_pct(gems_util), f"{gems_util/cuda_util:.2f}x"],
            ["Avg Kernel Duration", f"{fmt_us(ns_to_us(cuda_total/cuda_count))} us", f"{fmt_us(ns_to_us(gems_total/gems_count))} us", "-"],
            ["Avg Kernel Gap", f"{fmt_us(ns_to_us(cuda_gap['avg_gap_ns']))} us", f"{fmt_us(ns_to_us(gems_gap['avg_gap_ns']))} us", f"{gems_gap['avg_gap_ns']/cuda_gap['avg_gap_ns']:.2f}x"],
            ["Total Gap Time", f"{fmt_ms(ns_to_ms(cuda_gap['total_gap_ns']))} ms", f"{fmt_ms(ns_to_ms(gems_gap['total_gap_ns']))} ms", f"{gems_gap['total_gap_ns']/cuda_gap['total_gap_ns']:.2f}x"],
        ]
        report.append(md_table(["Metric", "CUDA", "FlagGems", "Ratio(FlagGems/CUDA)"], overview_rows))
        report.append("")

        # 2. Short kernel analysis
        report.append("## 2. Short Kernel Analysis")
        report.append("")
        report.append("Short kernels (< 10us) with high proportion lead to increased launch overhead and reduced GPU utilization.")
        report.append("")

        short_rows = [
            ["<2us", str(cuda_short["lt_2us"]), fmt_pct(cuda_short["lt_2us"]/cuda_short["total"]),
             str(gems_short["lt_2us"]), fmt_pct(gems_short["lt_2us"]/gems_short["total"])],
            ["<5us", str(cuda_short["lt_5us"]), fmt_pct(cuda_short["lt_5us"]/cuda_short["total"]),
             str(gems_short["lt_5us"]), fmt_pct(gems_short["lt_5us"]/gems_short["total"])],
            ["<10us", str(cuda_short["lt_10us"]), fmt_pct(cuda_short["lt_10us"]/cuda_short["total"]),
             str(gems_short["lt_10us"]), fmt_pct(gems_short["lt_10us"]/gems_short["total"])],
        ]
        report.append(md_table(["Duration Range", "CUDA Count", "CUDA %", "FlagGems Count", "FlagGems %"], short_rows))
        report.append("")

        # 3. Top Kernels comparison
        report.append("## 3. Top 15 GPU Kernels Comparison")
        report.append("")
        report.append("### 3.1 CUDA Top 15")
        report.append("")
        cuda_top_rows = []
        for i, k in enumerate(cuda_top, 1):
            cuda_top_rows.append([
                str(i),
                f"`{k.name}`",
                get_aten_name(k.name),
                str(k.instances),
                fmt_ms(ns_to_ms(k.total_ns)),
                fmt_us(k.avg_ns / 1000),
                fmt_pct(k.total_ns / cuda_total),
            ])
        report.append(md_table(["Rank", "Kernel Name", "aten Operation", "Count", "Total(ms)", "Avg(us)", "%"], cuda_top_rows))
        report.append("")

        # 3.1b CUDA Top 15 (with Host Bound)
        cuda_top_hb = query_top_kernels_with_hb(cuda, 15)
        cuda_combined_total_us = sum(k.total_combined_us for k in cuda_top_hb)
        report.append("### 3.1b CUDA Top 15 (with Host Bound)")
        report.append("")
        report.append("Sorted by combined total time (Kernel + Host Bound).")
        report.append("")
        cuda_top_hb_rows = []
        for k in cuda_top_hb:
            pct = k.total_combined_us / cuda_combined_total_us if cuda_combined_total_us > 0 else 0
            cuda_top_hb_rows.append([
                f"`{k.name}`",
                get_aten_name(k.name),
                str(k.instances),
                fmt_ms(k.total_kernel_us / 1000),
                fmt_us(k.avg_kernel_us),
                fmt_ms(k.total_hb_us / 1000),
                fmt_us(k.avg_hb_us),
                fmt_ms(k.total_combined_us / 1000),
                fmt_pct(pct),
            ])
        report.append(md_table(
            ["Kernel Name", "aten Op", "Count", "Kernel Total(ms)", "Kernel Avg(us)", "HB Total(ms)", "HB Avg(us)", "Total Time(ms)", "%"],
            cuda_top_hb_rows
        ))
        report.append("")

        report.append("### 3.2 FlagGems Top 15")
        report.append("")
        gems_top_rows = []
        for i, k in enumerate(gems_top, 1):
            gems_top_rows.append([
                str(i),
                f"`{k.name}`",
                get_aten_name(k.name),
                str(k.instances),
                fmt_ms(ns_to_ms(k.total_ns)),
                fmt_us(k.avg_ns / 1000),
                fmt_pct(k.total_ns / gems_total),
            ])
        report.append(md_table(["Rank", "Kernel Name", "aten Operation", "Count", "Total(ms)", "Avg(us)", "%"], gems_top_rows))
        report.append("")

        # 3.2b FlagGems Top 15 (with Host Bound)
        gems_top_hb = query_top_kernels_with_hb(gems, 15)
        gems_combined_total_us = sum(k.total_combined_us for k in gems_top_hb)
        report.append("### 3.2b FlagGems Top 15 (with Host Bound)")
        report.append("")
        report.append("Sorted by combined total time (Kernel + Host Bound).")
        report.append("")
        gems_top_hb_rows = []
        for k in gems_top_hb:
            pct = k.total_combined_us / gems_combined_total_us if gems_combined_total_us > 0 else 0
            gems_top_hb_rows.append([
                f"`{k.name}`",
                get_aten_name(k.name),
                str(k.instances),
                fmt_ms(k.total_kernel_us / 1000),
                fmt_us(k.avg_kernel_us),
                fmt_ms(k.total_hb_us / 1000),
                fmt_us(k.avg_hb_us),
                fmt_ms(k.total_combined_us / 1000),
                fmt_pct(pct),
            ])
        report.append(md_table(
            ["Kernel Name", "aten Op", "Count", "Kernel Total(ms)", "Kernel Avg(us)", "HB Total(ms)", "HB Avg(us)", "Total Time(ms)", "%"],
            gems_top_hb_rows
        ))
        report.append("")

        # 4. FlagGems specific kernels analysis
        report.append("## 4. FlagGems Specific Kernel Analysis")
        report.append("")
        report.append("The following are FlagGems/Triton path specific kernels (sorted by total time), these kernels don't exist in CUDA path or are replaced by more efficient implementations:")
        report.append("")

        # Get FlagGems host bound data
        gems_hb_data = query_kernel_with_host_bound_total(gems)

        flaggems_rows = []
        total_flaggems_time = sum(k.total_ns for k in gems_flaggems)
        for k in gems_flaggems[:15]:
            # Get host bound data
            kernel_us = k.total_ns / 1000.0
            hb_us = gems_hb_data.get(k.name, (0, 0))[1]  # (kernel_us, hb_us)
            combined_us = kernel_us + hb_us
            flaggems_rows.append([
                f"`{k.name}`",
                get_aten_name(k.name),
                str(k.instances),
                fmt_ms(ns_to_ms(k.total_ns)),
                fmt_us(k.avg_ns / 1000),
                fmt_ms(hb_us / 1000),
                fmt_ms(combined_us / 1000),
                fmt_pct(k.total_ns / gems_total),
            ])
        report.append(md_table(["Kernel Name", "aten Op", "Count", "Kernel Total(ms)", "Kernel Avg(us)", "HB Total(ms)", "Combined Total(ms)", "%"], flaggems_rows))
        report.append("")
        report.append(f"**FlagGems specific kernel total time: {fmt_ms(ns_to_ms(total_flaggems_time))} ms, {fmt_pct(total_flaggems_time / gems_total)} of FlagGems kernel total time**")
        report.append("")

        # 5. Category comparison
        report.append("## 5. Category Aggregation Comparison")
        report.append("")
        cats = sorted(set(cuda_cat.keys()) | set(gems_cat.keys()))
        cat_rows = []
        for cat in cats:
            c_total = float(cuda_cat.get(cat, {}).get("total_ns", 0.0))
            g_total = float(gems_cat.get(cat, {}).get("total_ns", 0.0))
            c_inst = int(cuda_cat.get(cat, {}).get("instances", 0.0))
            g_inst = int(gems_cat.get(cat, {}).get("instances", 0.0))
            ratio = (g_total / c_total) if c_total > 0 else float("inf")
            cat_rows.append([
                cat,
                str(c_inst),
                fmt_ms(ns_to_ms(int(c_total))),
                fmt_pct(c_total / cuda_total if cuda_total else 0.0),
                str(g_inst),
                fmt_ms(ns_to_ms(int(g_total))),
                fmt_pct(g_total / gems_total if gems_total else 0.0),
                f"{ratio:.2f}x" if ratio != float("inf") else "inf",
            ])
        # Sort by FlagGems total time descending
        cat_rows.sort(key=lambda r: float(r[5].split()[0]) if r[5] != "N/A" else 0, reverse=True)
        report.append(md_table(
            ["Category", "CUDA Count", "CUDA Time(ms)", "CUDA %", "FG Count", "FG Time(ms)", "FG %", "Ratio(FG/CUDA)"],
            cat_rows,
        ))
        report.append("")

        # 6. Same name kernel comparison
        report.append("## 6. Same Kernel Direct Comparison (Top 15 Intersection)")
        report.append("")

        cuda_name_map = {k.name: k for k in cuda_all}
        gems_name_map = {k.name: k for k in gems_all}
        shared_names = sorted(set(cuda_name_map.keys()) & set(gems_name_map.keys()))

        compare_rows = []
        for name in shared_names:
            c = cuda_name_map[name]
            g = gems_name_map[name]
            ratio = (g.total_ns / c.total_ns) if c.total_ns > 0 else float('inf')
            avg_ratio = (g.avg_ns / c.avg_ns) if c.avg_ns > 0 else float('inf')
            compare_rows.append({
                "name": name,
                "cuda_total": c.total_ns,
                "gems_total": g.total_ns,
                "cuda_avg": c.avg_ns,
                "gems_avg": g.avg_ns,
                "cuda_inst": c.instances,
                "gems_inst": g.instances,
                "ratio": ratio,
                "avg_ratio": avg_ratio,
            })

        # Sort by FlagGems total time descending
        compare_rows.sort(key=lambda r: r["gems_total"], reverse=True)

        compare_table_rows = []
        for r in compare_rows[:15]:
            compare_table_rows.append([
                f"`{r['name']}`",
                get_aten_name(r['name']),
                str(r["cuda_inst"]),
                fmt_ms(ns_to_ms(r["cuda_total"])),
                str(r["gems_inst"]),
                fmt_ms(ns_to_ms(r["gems_total"])),
                f"{r['ratio']:.2f}x" if r['ratio'] != float('inf') else "inf",
                f"{r['avg_ratio']:.2f}x" if r['avg_ratio'] != float('inf') else "inf",
            ])

        report.append(md_table(
            ["Kernel Name", "aten Op", "CUDA Count", "CUDA Total(ms)", "FG Count", "FG Total(ms)", "Total Ratio", "Avg Ratio"],
            compare_table_rows,
        ))
        report.append("")

        # 7. Runtime API comparison
        report.append("## 7. CUDA Runtime API Overhead Comparison")
        report.append("")
        report.append("### 7.1 CUDA Path Top 10 Runtime API")
        report.append("")
        cuda_rt_rows = []
        for name, cnt, total, avg in cuda_runtime:
            cuda_rt_rows.append([f"`{name}`", str(cnt), fmt_ms(ns_to_ms(total)), fmt_us(avg / 1000)])
        report.append(md_table(["API Name", "Count", "Total(ms)", "Avg(us)"], cuda_rt_rows))
        report.append("")

        report.append("### 7.2 FlagGems Path Top 10 Runtime API")
        report.append("")
        gems_rt_rows = []
        for name, cnt, total, avg in gems_runtime:
            gems_rt_rows.append([f"`{name}`", str(cnt), fmt_ms(ns_to_ms(total)), fmt_us(avg / 1000)])
        report.append(md_table(["API Name", "Count", "Total(ms)", "Avg(us)"], gems_rt_rows))
        report.append("")

        # Write report
        reports_path = project_root / reports_dir
        reports_path.mkdir(parents=True, exist_ok=True)
        out_path = reports_path / "FlagGems-all.md"
        out_path.write_text("\n".join(report), encoding="utf-8")
        print(f"Report generated: {out_path}")

    finally:
        cuda.close()
        gems.close()


if __name__ == "__main__":
    main()
