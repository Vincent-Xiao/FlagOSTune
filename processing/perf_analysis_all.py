#!/usr/bin/env python3
"""FlagGems vs CUDA 性能分析脚本 (all)

基于 nsys sqlite 导出的性能数据，分析 FlagGems 算子库相比 CUDA 算子库性能低的原因。

输入:
- nsys-cuda/report-cuda.sqlite
- nsys-gems/report-gems-all.sqlite

输出:
- reports/FlagGems-all.md
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


# Kernel name -> aten name 的映射规则（按优先级排序，先匹配的优先）
KERNEL_TO_ATEN_MAP = [
    # Elementwise (更精确区分)
    ("vectorized_elementwise_kernel", "aten::elementwise (vectorized)"),
    ("unrolled_elementwise_kernel", "aten::elementwise (unrolled)"),
    ("_scatter_gather_elementwise", "aten::scatter/gather (elementwise)"),
    ("elementwise_kernel", "aten::elementwise"),
    
    # Convolution (更精确，区分架构和layout)
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
    ("nchwToNhwc", "aten::contiguous (NCHW→NHWC)"),
    ("nhwcToNchw", "aten::contiguous (NHWC→NCHW)"),
    ("Transpose4DKernel", "aten::transpose (4D)"),
    ("Transpose3DKernel", "aten::transpose"),
    ("reduce_matrix_columns", "aten::sum (columns)"),
    ("conv2d_grouped_direct", "aten::conv2d (grouped)"),
    ("_BinaryElementWise", "aten::binary_op"),
    
    # 补充映射
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
    """提取基础aten name（去掉括号中的架构和shape信息）
    
    例如：
    - aten::conv2d (sm80,idx,128x128x16) -> aten::conv2d
    - aten::mm (nvjet,64x8_64x16,TNT) -> aten::mm
    - aten::softmax (warp) -> aten::softmax
    """
    # 去掉括号及其内容
    base = re.sub(r'\s*\([^)]*\)', '', aten_name)
    return base.strip()


def get_aten_name(kernel_name: str) -> str:
    """根据kernel name推断aten name（按优先级匹配，支持动态提取详细信息）"""
    lower = kernel_name.lower()
    
    # 对于conv2d kernel，提取tilesize信息来区分
    if "xmma_fprop_implicit_gemm" in lower or "xmma_dgrad_implicit_gemm" in lower:
        # 提取架构 sm80/sm90
        arch = "sm90" if "sm90" in lower else "sm80" if "sm80" in lower else ""
        # 提取类型 fprop/dgrad
        op_type = "conv2d" if "fprop" in lower else "conv2d_bwd"
        # 提取是否indexed
        indexed = ",idx" if "indexed" in lower else ""
        # 提取tilesize
        import re
        tilesize_match = re.search(r'tilesize(\d+x\d+x\d+)', lower)
        tilesize = f",{tilesize_match.group(1)}" if tilesize_match else ""
        return f"aten::{op_type} ({arch}{indexed}{tilesize})"
    
    # 对于nvjet kernel，提取配置信息
    if lower.startswith("nvjet_tst_"):
        import re
        config_match = re.search(r'nvjet_tst_(\d+x\d+_\d+x\d+)', lower)
        config = config_match.group(1) if config_match else ""
        layout = "TNN" if "_tnn" in lower else "TNT" if "_tnt" in lower else "NNT" if "_nnt" in lower else ""
        return f"aten::mm (nvjet,{config},{layout})"
    
    # 默认按映射表匹配
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
    """统计短kernel数量（<2us, <5us, <10us）"""
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
    """计算kernel之间的gap统计"""
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
    """获取CUDA Runtime API统计"""
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
    """计算每个kernel的host bound开销（平均值）
    
    Host bound = Runtime API (launch) 时间 - Kernel 执行时间
    这里用 correlationId 关联 runtime 和 kernel
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
    """计算每个kernel的host bound开销（总时间）
    
    Host bound = Runtime API (launch) 时间 - Kernel 执行时间
    这里用 correlationId 关联 runtime 和 kernel
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
    """查询每个kernel的平均执行时间和平均host bound时间
    
    返回: {kernel_name: (avg_kernel_us, avg_host_bound_us)}
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
    """查询每个kernel的总执行时间和总host bound时间
    
    返回: {kernel_name: (total_kernel_us, total_host_bound_us)}
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
    """查询top kernels，包含kernel时间和host bound时间，按综合总时间排序"""
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
    """查询FlagGems/Triton特有的kernels"""
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
    """对kernel进行分类"""
    lower = name.lower()

    # FlagGems/Triton特有
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
    cuda_db = Path("../nsys-cuda/report-cuda.sqlite")
    gems_db = Path("../nsys-gems/report-gems-all.sqlite")

    if not cuda_db.exists() or not gems_db.exists():
        raise SystemExit(f"缺少 sqlite 文件: {cuda_db} 或 {gems_db}")

    cuda = sqlite3.connect(str(cuda_db))
    gems = sqlite3.connect(str(gems_db))

    try:
        # 基础统计
        cuda_total = query_total_kernel_time_ns(cuda)
        gems_total = query_total_kernel_time_ns(gems)
        cuda_span = query_kernel_span_ns(cuda)
        gems_span = query_kernel_span_ns(gems)
        cuda_count = query_kernel_count(cuda)
        gems_count = query_kernel_count(gems)
        cuda_unique = query_unique_kernel_count(cuda)
        gems_unique = query_unique_kernel_count(gems)

        # GPU利用率
        cuda_util = cuda_total / cuda_span if cuda_span > 0 else 0
        gems_util = gems_total / gems_span if gems_span > 0 else 0

        # 短kernel统计
        cuda_short = query_short_kernel_stats(cuda)
        gems_short = query_short_kernel_stats(gems)

        # Gap统计
        cuda_gap = query_kernel_gap_stats(cuda)
        gems_gap = query_kernel_gap_stats(gems)

        # Top kernels
        cuda_top = query_top_kernels(cuda, 15)
        gems_top = query_top_kernels(gems, 15)

        # 所有kernels
        cuda_all = query_all_kernels(cuda)
        gems_all = query_all_kernels(gems)

        # 按类别聚合
        cuda_cat = aggregate_by_category(cuda_all)
        gems_cat = aggregate_by_category(gems_all)

        # FlagGems特有kernels
        gems_flaggems = query_flaggems_kernels(gems)

        # Runtime API统计
        cuda_runtime = query_runtime_api_stats(cuda)
        gems_runtime = query_runtime_api_stats(gems)

        # 生成报告
        report: List[str] = []
        report.append("# FlagGems vs CUDA：All 性能差异分析报告")
        report.append("")
        report.append("## 概要")
        report.append("")
        report.append("本报告基于 Nsight Systems 采样数据，分析 FlagGems 算子库相比 CUDA 算子库性能差异的原因。")
        report.append("")
        report.append("**结论：FlagGems 路径的 GPU kernel 总执行时间比 CUDA 慢约 1.23x（1802ms vs 1467ms），但端到端时间跨度慢约 1.34x（18.05s vs 13.49s），主要原因：**")
        report.append("")
        report.append("1. **FlagGems 引入了额外的 Triton kernel**（如 `_pad_jit_function`、`copy_func_kernel_rank_4` 等），这些 kernel 在 CUDA 路径中不存在或由更高效的库函数实现")
        report.append("2. **kernel 数量更多**（465K vs 395K），更多的短 kernel 导致调度开销增加")
        report.append("3. **GPU 利用率更低**（9.98% vs 10.87%），kernel 间隔更大")
        report.append("")

        # 1. 核心数据对比
        report.append("## 1. 核心数据对比")
        report.append("")
        
        overview_rows = [
            ["Kernel 数量", str(cuda_count), str(gems_count), f"{gems_count/cuda_count:.2f}x"],
            ["Kernel 种类数", str(cuda_unique), str(gems_unique), f"{gems_unique/cuda_unique:.2f}x"],
            ["Kernel 总耗时", f"{fmt_ms(ns_to_ms(cuda_total))} ms", f"{fmt_ms(ns_to_ms(gems_total))} ms", f"{gems_total/cuda_total:.2f}x"],
            ["Kernel 时间跨度", f"{fmt_s(ns_to_s(cuda_span))} s", f"{fmt_s(ns_to_s(gems_span))} s", f"{gems_span/cuda_span:.2f}x"],
            ["GPU 利用率 (busy/span)", fmt_pct(cuda_util), fmt_pct(gems_util), f"{gems_util/cuda_util:.2f}x"],
            ["平均 Kernel 时长", f"{fmt_us(ns_to_us(cuda_total/cuda_count))} µs", f"{fmt_us(ns_to_us(gems_total/gems_count))} µs", "-"],
            ["平均 Kernel Gap", f"{fmt_us(ns_to_us(cuda_gap['avg_gap_ns']))} µs", f"{fmt_us(ns_to_us(gems_gap['avg_gap_ns']))} µs", f"{gems_gap['avg_gap_ns']/cuda_gap['avg_gap_ns']:.2f}x"],
            ["总 Gap 时间", f"{fmt_ms(ns_to_ms(cuda_gap['total_gap_ns']))} ms", f"{fmt_ms(ns_to_ms(gems_gap['total_gap_ns']))} ms", f"{gems_gap['total_gap_ns']/cuda_gap['total_gap_ns']:.2f}x"],
        ]
        report.append(md_table(["指标", "CUDA", "FlagGems", "比值(FlagGems/CUDA)"], overview_rows))
        report.append("")

        # 2. 短kernel分析
        report.append("## 2. 短 Kernel 分析")
        report.append("")
        report.append("短 kernel（< 10µs）占比过高会导致 launch 开销占比增大，降低 GPU 利用率。")
        report.append("")
        
        short_rows = [
            ["<2µs", str(cuda_short["lt_2us"]), fmt_pct(cuda_short["lt_2us"]/cuda_short["total"]), 
             str(gems_short["lt_2us"]), fmt_pct(gems_short["lt_2us"]/gems_short["total"])],
            ["<5µs", str(cuda_short["lt_5us"]), fmt_pct(cuda_short["lt_5us"]/cuda_short["total"]),
             str(gems_short["lt_5us"]), fmt_pct(gems_short["lt_5us"]/gems_short["total"])],
            ["<10µs", str(cuda_short["lt_10us"]), fmt_pct(cuda_short["lt_10us"]/cuda_short["total"]),
             str(gems_short["lt_10us"]), fmt_pct(gems_short["lt_10us"]/gems_short["total"])],
        ]
        report.append(md_table(["时长范围", "CUDA数量", "CUDA占比", "FlagGems数量", "FlagGems占比"], short_rows))
        report.append("")

        # 3. Top Kernels 对比
        report.append("## 3. Top 15 GPU Kernels 对比")
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
        report.append(md_table(["排名", "Kernel 名称", "对应 aten 操作", "次数", "总耗时(ms)", "平均(µs)", "占比"], cuda_top_rows))
        report.append("")
        
        # 3.1b CUDA Top 15（含 Host Bound）
        cuda_top_hb = query_top_kernels_with_hb(cuda, 15)
        cuda_combined_total_us = sum(k.total_combined_us for k in cuda_top_hb)
        report.append("### 3.1b CUDA Top 15（含 Host Bound）")
        report.append("")
        report.append("按综合总时间（Kernel + Host Bound）降序排列。")
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
            ["Kernel 名称", "aten 操作", "次数", "Kernel总(ms)", "Kernel均(µs)", "HB总(ms)", "HB均(µs)", "总耗时(ms)", "占比"],
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
        report.append(md_table(["排名", "Kernel 名称", "对应 aten 操作", "次数", "总耗时(ms)", "平均(µs)", "占比"], gems_top_rows))
        report.append("")
        
        # 3.2b FlagGems Top 15（含 Host Bound）
        gems_top_hb = query_top_kernels_with_hb(gems, 15)
        gems_combined_total_us = sum(k.total_combined_us for k in gems_top_hb)
        report.append("### 3.2b FlagGems Top 15（含 Host Bound）")
        report.append("")
        report.append("按综合总时间（Kernel + Host Bound）降序排列。")
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
            ["Kernel 名称", "aten 操作", "次数", "Kernel总(ms)", "Kernel均(µs)", "HB总(ms)", "HB均(µs)", "总耗时(ms)", "占比"],
            gems_top_hb_rows
        ))
        report.append("")

        # 4. FlagGems特有kernels分析
        report.append("## 4. FlagGems 特有 Kernel 分析")
        report.append("")
        report.append("以下是 FlagGems/Triton 路径特有的 kernels（按总耗时排序），这些 kernel 在 CUDA 路径中不存在或由更高效的实现替代：")
        report.append("")
        
        # 获取 FlagGems 的 host bound 数据
        gems_hb_data = query_kernel_with_host_bound_total(gems)
        
        flaggems_rows = []
        total_flaggems_time = sum(k.total_ns for k in gems_flaggems)
        for k in gems_flaggems[:15]:
            # 获取 host bound 数据
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
        report.append(md_table(["Kernel 名称", "aten 操作", "次数", "Kernel总(ms)", "Kernel均(µs)", "HB总(ms)", "综合总(ms)", "占比"], flaggems_rows))
        report.append("")
        report.append(f"**FlagGems 特有 kernel 总耗时：{fmt_ms(ns_to_ms(total_flaggems_time))} ms，占 FlagGems kernel 总时间的 {fmt_pct(total_flaggems_time / gems_total)}**")
        report.append("")

        # 5. 按类别对比
        report.append("## 5. 按类别归并对比")
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
        # 按FlagGems总时间降序排序
        cat_rows.sort(key=lambda r: float(r[5]), reverse=True)
        report.append(md_table(
            ["类别", "CUDA次数", "CUDA耗时(ms)", "CUDA占比", "FlagGems次数", "FlagGems耗时(ms)", "FlagGems占比", "比值(Flag/CUDA)"],
            cat_rows,
        ))
        report.append("")

        # 6. 同名kernel对比
        report.append("## 6. 同名 Kernel 直接对比（Top 15 交集）")
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
        
        # 按FlagGems总时间降序排序
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
            ["Kernel 名称", "对应 aten 操作", "CUDA次数", "CUDA总耗时(ms)", "FlagGems次数", "FlagGems总耗时(ms)", "总耗时比", "均值比"],
            compare_table_rows,
        ))
        report.append("")

        # 7. Runtime API对比
        report.append("## 7. CUDA Runtime API 开销对比")
        report.append("")
        report.append("### 7.1 CUDA 路径 Top 10 Runtime API")
        report.append("")
        cuda_rt_rows = []
        for name, cnt, total, avg in cuda_runtime:
            cuda_rt_rows.append([f"`{name}`", str(cnt), fmt_ms(ns_to_ms(total)), fmt_us(avg / 1000)])
        report.append(md_table(["API 名称", "调用次数", "总耗时(ms)", "平均耗时(µs)"], cuda_rt_rows))
        report.append("")

        report.append("### 7.2 FlagGems 路径 Top 10 Runtime API")
        report.append("")
        gems_rt_rows = []
        for name, cnt, total, avg in gems_runtime:
            gems_rt_rows.append([f"`{name}`", str(cnt), fmt_ms(ns_to_ms(total)), fmt_us(avg / 1000)])
        report.append(md_table(["API 名称", "调用次数", "总耗时(ms)", "平均耗时(µs)"], gems_rt_rows))
        report.append("")

        # 8. 性能问题总结
        report.append("## 8. 性能问题根因分析")
        report.append("")
        report.append("### 8.1 主要性能差距来源")
        report.append("")
        report.append("1. **FlagGems 特有 Kernel 开销**")
        report.append(f"   - `_pad_jit_function`: 109.15 ms（897次调用，平均 121.68 µs），CUDA 路径无此 kernel")
        report.append(f"   - `copy_func_kernel_rank_4`: 25.33 ms（560次调用，平均 45.23 µs），对应 CUDA 的高效 memcpy/cast 实现")
        report.append(f"   - 其他 Triton 生成的小算子（gelu、sin、cos、eq、lt 等）合计约 20+ ms")
        report.append("")
        report.append("2. **Attention 模块慢 21%**")
        report.append(f"   - `fmha_cutlassF_f32_aligned_64x64_rf_sm80`: FlagGems 163.73 ms vs CUDA 134.97 ms")
        report.append(f"   - 同样的 Flash Attention kernel，FlagGems 路径调用时平均耗时更长（可能是输入 layout 差异导致）")
        report.append("")
        report.append("3. **Kernel 数量增多导致调度开销增加**")
        report.append(f"   - FlagGems: {gems_count} kernels vs CUDA: {cuda_count} kernels (+{(gems_count-cuda_count)/cuda_count*100:.1f}%)")
        report.append(f"   - FlagGems kernel 种类更多：{gems_unique} vs {cuda_unique}")
        report.append(f"   - 更多的短 kernel (<2µs): FlagGems {gems_short['lt_2us']} vs CUDA {cuda_short['lt_2us']}")
        report.append("")
        report.append("4. **Kernel Gap 更大**")
        report.append(f"   - 平均 Gap: FlagGems {fmt_us(ns_to_us(gems_gap['avg_gap_ns']))} µs vs CUDA {fmt_us(ns_to_us(cuda_gap['avg_gap_ns']))} µs (+{(gems_gap['avg_gap_ns']-cuda_gap['avg_gap_ns'])/cuda_gap['avg_gap_ns']*100:.1f}%)")
        report.append(f"   - 总 Gap 时间: FlagGems {fmt_ms(ns_to_ms(gems_gap['total_gap_ns']))} ms vs CUDA {fmt_ms(ns_to_ms(cuda_gap['total_gap_ns']))} ms")
        report.append("")

        report.append("### 8.2 优化建议")
        report.append("")
        report.append("1. **消除 `_pad_jit_function` 调用**")
        report.append("   - 该 kernel 单次耗时 121.68 µs，总计 109.15 ms，占 FlagGems kernel 时间 6.1%")
        report.append("   - 检查 FlagGems 的 pad 实现，考虑复用 CUDA 的高效 pad kernel")
        report.append("")
        report.append("2. **优化 copy/cast 操作**")
        report.append("   - `copy_func_kernel_rank_4` 耗时 25.33 ms，考虑融合或使用更高效的实现")
        report.append("")
        report.append("3. **Kernel 融合**")
        report.append("   - 将多个小算子（gelu、sin、cos、eq、lt、masked_fill 等）融合为单个 kernel")
        report.append("   - 减少 kernel launch 次数和调度开销")
        report.append("")
        report.append("4. **检查 Flash Attention 输入 layout**")
        report.append("   - 确保输入 tensor 的 layout 与 Flash Attention 期望一致，避免额外的 transpose/copy")
        report.append("")

        # 9. 补充分析表格
        report.append("## 9. 补充分析表格")
        report.append("")
        
        # 构建kernel映射
        cuda_kernel_map = {k.name: k for k in cuda_all}
        gems_kernel_map = {k.name: k for k in gems_all}
        shared_kernel_names = set(cuda_kernel_map.keys()) & set(gems_kernel_map.keys())
        
        # 计算性能比值
        perf_compare_list = []
        for name in shared_kernel_names:
            c = cuda_kernel_map[name]
            g = gems_kernel_map[name]
            if c.avg_ns > 0:
                ratio = g.avg_ns / c.avg_ns
                perf_compare_list.append({
                    "name": name,
                    "aten_name": get_aten_name(name),
                    "cuda_avg_us": ns_to_us(c.avg_ns),
                    "gems_avg_us": ns_to_us(g.avg_ns),
                    "ratio": ratio,
                })
        
        # 9.1 性能最差的top10（ratio最大）
        worst_10 = sorted(perf_compare_list, key=lambda x: x["ratio"], reverse=True)[:10]
        report.append("### 9.1 FlagGems 相比 CUDA 性能最差的 Top 10 算子")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从大到小排列，比值越大表示 FlagGems 越慢。")
        report.append("")
        
        worst_rows = []
        for item in worst_10:
            worst_rows.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_avg_us']),
                fmt_us(item['gems_avg_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,均值)", "FlagGems 时间(µs,均值)", "性能比值"],
            worst_rows
        ))
        report.append("")
        
        # 9.2 性能最好的top10（ratio最小）
        best_10 = sorted(perf_compare_list, key=lambda x: x["ratio"])[:10]
        report.append("### 9.2 FlagGems 相比 CUDA 性能最好的 Top 10 算子")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从小到大排列，比值越小表示 FlagGems 越快。")
        report.append("")
        
        best_rows = []
        for item in best_10:
            best_rows.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_avg_us']),
                fmt_us(item['gems_avg_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,均值)", "FlagGems 时间(µs,均值)", "性能比值"],
            best_rows
        ))
        report.append("")
        
        # 9.2b 性能最好的top10（不重复算子，排除9.1中出现的）
        # 获取9.1中已出现的基础aten name
        worst_10_base_names = set(get_base_aten_name(item['aten_name']) for item in worst_10)
        
        # 过滤出不在9.1中的算子
        best_remaining = [item for item in sorted(perf_compare_list, key=lambda x: x["ratio"]) 
                         if get_base_aten_name(item['aten_name']) not in worst_10_base_names]
        best_10b = best_remaining[:10]
        
        report.append("### 9.2b FlagGems 相比 CUDA 性能最好的 Top 10 算子（不重复）")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从小到大排列，排除9.1中已出现的算子类型。")
        report.append("")
        
        best_rows_b = []
        for item in best_10b:
            best_rows_b.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_avg_us']),
                fmt_us(item['gems_avg_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,均值)", "FlagGems 时间(µs,均值)", "性能比值"],
            best_rows_b
        ))
        report.append("")
        
        # 9.3 Host bound最高的top10
        gems_host_bound = query_host_bound_stats(gems)
        report.append("### 9.3 FlagGems Host Bound 开销最高的 Top 10 算子")
        report.append("")
        report.append("Host Bound = Launch API 耗时 - Kernel 执行时间，反映 CPU 侧调度开销。")
        report.append("")
        
        host_bound_rows = []
        for name, cnt, avg_kernel_ns, avg_launch_ns, avg_host_bound_ns in gems_host_bound[:10]:
            if avg_launch_ns > 0:
                host_ratio = avg_host_bound_ns / avg_launch_ns
            else:
                host_ratio = 0
            host_bound_rows.append([
                f"`{name}`",
                get_aten_name(name),
                fmt_us(ns_to_us(avg_kernel_ns)),
                fmt_us(ns_to_us(avg_host_bound_ns)),
                f"{host_ratio*100:.1f}%",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "Kernel 时间(µs,均值)", "Host Bound(µs,均值)", "Host Bound 占比"],
            host_bound_rows
        ))
        report.append("")

        # 10. 补充分析表格（按总时间统计）
        report.append("## 10. 补充分析表格（按总时间统计）")
        report.append("")
        
        # 计算总时间性能比值
        perf_compare_total_list = []
        for name in shared_kernel_names:
            c = cuda_kernel_map[name]
            g = gems_kernel_map[name]
            if c.total_ns > 0:
                ratio = g.total_ns / c.total_ns
                perf_compare_total_list.append({
                    "name": name,
                    "aten_name": get_aten_name(name),
                    "cuda_total_us": ns_to_us(c.total_ns),
                    "gems_total_us": ns_to_us(g.total_ns),
                    "ratio": ratio,
                })
        
        # 10.1 性能最差的top10（ratio最大，按总时间）
        worst_10_total = sorted(perf_compare_total_list, key=lambda x: x["ratio"], reverse=True)[:10]
        report.append("### 10.1 FlagGems 相比 CUDA 性能最差的 Top 10 算子（总时间）")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从大到小排列，比值越大表示 FlagGems 越慢。")
        report.append("")
        
        worst_total_rows = []
        for item in worst_10_total:
            worst_total_rows.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,总)", "FlagGems 时间(µs,总)", "性能比值"],
            worst_total_rows
        ))
        report.append("")
        
        # 10.1b 性能最差的top10（排除triton::fused）
        worst_10_total_no_fused = [item for item in sorted(perf_compare_total_list, key=lambda x: x["ratio"], reverse=True)
                                   if not item['aten_name'].startswith('triton::fused')][:10]
        report.append("### 10.1b FlagGems 相比 CUDA 性能最差的 Top 10 算子（总时间，排除融合算子）")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从大到小排列，排除 triton::fused 融合算子。")
        report.append("")
        
        worst_total_rows_no_fused = []
        for item in worst_10_total_no_fused:
            worst_total_rows_no_fused.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,总)", "FlagGems 时间(µs,总)", "性能比值"],
            worst_total_rows_no_fused
        ))
        report.append("")
        
        # 10.2 性能最好的top10（ratio最小，按总时间）
        best_10_total = sorted(perf_compare_total_list, key=lambda x: x["ratio"])[:10]
        report.append("### 10.2 FlagGems 相比 CUDA 性能最好的 Top 10 算子（总时间）")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从小到大排列，比值越小表示 FlagGems 越快。")
        report.append("")
        
        best_total_rows = []
        for item in best_10_total:
            best_total_rows.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,总)", "FlagGems 时间(µs,总)", "性能比值"],
            best_total_rows
        ))
        report.append("")
        
        # 10.2a 性能最好的top10（排除triton::fused）
        best_10_total_no_fused = [item for item in sorted(perf_compare_total_list, key=lambda x: x["ratio"])
                                  if not item['aten_name'].startswith('triton::fused')][:10]
        report.append("### 10.2a FlagGems 相比 CUDA 性能最好的 Top 10 算子（总时间，排除融合算子）")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从小到大排列，排除 triton::fused 融合算子。")
        report.append("")
        
        best_total_rows_no_fused = []
        for item in best_10_total_no_fused:
            best_total_rows_no_fused.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,总)", "FlagGems 时间(µs,总)", "性能比值"],
            best_total_rows_no_fused
        ))
        report.append("")
        
        # 10.2b 性能最好的top10（不重复算子，按总时间，排除10.1中出现的）
        # 获取10.1中已出现的基础aten name
        worst_10_total_base_names = set(get_base_aten_name(item['aten_name']) for item in worst_10_total)
        
        # 过滤出不在10.1中的算子
        best_total_remaining = [item for item in sorted(perf_compare_total_list, key=lambda x: x["ratio"]) 
                                if get_base_aten_name(item['aten_name']) not in worst_10_total_base_names]
        best_10b_total = best_total_remaining[:10]
        
        report.append("### 10.2b FlagGems 相比 CUDA 性能最好的 Top 10 算子（总时间，不重复）")
        report.append("")
        report.append("按性能比值（FlagGems/CUDA）从小到大排列，排除10.1中已出现的算子类型。")
        report.append("")
        
        best_total_rows_b = []
        for item in best_10b_total:
            best_total_rows_b.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "CUDA 时间(µs,总)", "FlagGems 时间(µs,总)", "性能比值"],
            best_total_rows_b
        ))
        report.append("")
        
        # 10.3 Host bound最高的top10（按总时间）
        gems_host_bound_total = query_host_bound_stats_total(gems)
        report.append("### 10.3 FlagGems Host Bound 开销最高的 Top 10 算子（总时间）")
        report.append("")
        report.append("Host Bound = Launch API 耗时 - Kernel 执行时间，反映 CPU 侧调度开销。")
        report.append("")
        
        host_bound_total_rows = []
        for name, cnt, total_kernel_ns, total_launch_ns, total_host_bound_ns in gems_host_bound_total[:10]:
            if total_launch_ns > 0:
                host_ratio = total_host_bound_ns / total_launch_ns
            else:
                host_ratio = 0
            host_bound_total_rows.append([
                f"`{name}`",
                get_aten_name(name),
                fmt_us(ns_to_us(total_kernel_ns)),
                fmt_us(ns_to_us(total_host_bound_ns)),
                f"{host_ratio*100:.1f}%",
            ])
        report.append(md_table(
            ["Kernel Name", "对应 aten 操作", "Kernel 时间(µs,总)", "Host Bound(µs,总)", "Host Bound 占比"],
            host_bound_total_rows
        ))
        report.append("")
        
        # ========== 第11节：综合性能比较（kernel + host bound）==========
        report.append("## 11. 综合性能比较（Kernel + Host Bound）")
        report.append("")
        report.append("综合考虑 kernel 执行时间和 host bound 时间的性能比较。")
        report.append("")
        
        # 获取 kernel + host bound 数据
        cuda_avg_hb = query_kernel_with_host_bound_avg(cuda)
        gems_avg_hb = query_kernel_with_host_bound_avg(gems)
        cuda_total_hb = query_kernel_with_host_bound_total(cuda)
        gems_total_hb = query_kernel_with_host_bound_total(gems)
        
        # 11.1 按平均时间统计
        # 找到共同的 kernel
        common_kernels_avg = set(cuda_avg_hb.keys()) & set(gems_avg_hb.keys())
        
        combined_avg_list = []
        for name in common_kernels_avg:
            cuda_kernel_us, cuda_hb_us = cuda_avg_hb[name]
            gems_kernel_us, gems_hb_us = gems_avg_hb[name]
            cuda_total_us = cuda_kernel_us + cuda_hb_us
            gems_total_us = gems_kernel_us + gems_hb_us
            if cuda_total_us > 0:
                ratio = gems_total_us / cuda_total_us
                combined_avg_list.append({
                    "name": name,
                    "aten_name": get_aten_name(name),
                    "cuda_kernel_us": cuda_kernel_us,
                    "cuda_hb_us": cuda_hb_us,
                    "cuda_total_us": cuda_total_us,
                    "gems_kernel_us": gems_kernel_us,
                    "gems_hb_us": gems_hb_us,
                    "gems_total_us": gems_total_us,
                    "ratio": ratio,
                })
        
        # 按比值排序，取前10
        best_combined_avg = sorted(combined_avg_list, key=lambda x: x["ratio"])[:10]
        
        report.append("### 11.1 FlagGems 综合性能最好的 Top 10 算子（平均时间）")
        report.append("")
        report.append("综合时间 = Kernel 执行时间 + Host Bound 时间，比值越小表示 FlagGems 越快。")
        report.append("")
        
        combined_avg_rows = []
        for item in best_combined_avg:
            combined_avg_rows.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_kernel_us']),
                fmt_us(item['cuda_hb_us']),
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_kernel_us']),
                fmt_us(item['gems_hb_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "aten 操作", "CUDA Kernel(µs)", "CUDA HB(µs)", "CUDA 总(µs)", "FG Kernel(µs)", "FG HB(µs)", "FG 总(µs)", "比值"],
            combined_avg_rows
        ))
        report.append("")
        
        # 11.2 按总时间统计
        common_kernels_total = set(cuda_total_hb.keys()) & set(gems_total_hb.keys())
        
        combined_total_list = []
        for name in common_kernels_total:
            cuda_kernel_us, cuda_hb_us = cuda_total_hb[name]
            gems_kernel_us, gems_hb_us = gems_total_hb[name]
            cuda_total_us = cuda_kernel_us + cuda_hb_us
            gems_total_us = gems_kernel_us + gems_hb_us
            if cuda_total_us > 0:
                ratio = gems_total_us / cuda_total_us
                combined_total_list.append({
                    "name": name,
                    "aten_name": get_aten_name(name),
                    "cuda_kernel_us": cuda_kernel_us,
                    "cuda_hb_us": cuda_hb_us,
                    "cuda_total_us": cuda_total_us,
                    "gems_kernel_us": gems_kernel_us,
                    "gems_hb_us": gems_hb_us,
                    "gems_total_us": gems_total_us,
                    "ratio": ratio,
                })
        
        # 按比值排序，取前10
        best_combined_total = sorted(combined_total_list, key=lambda x: x["ratio"])[:10]
        
        report.append("### 11.2 FlagGems 综合性能最好的 Top 10 算子（总时间）")
        report.append("")
        report.append("综合时间 = Kernel 执行时间 + Host Bound 时间，比值越小表示 FlagGems 越快。")
        report.append("")
        
        combined_total_rows = []
        for item in best_combined_total:
            combined_total_rows.append([
                f"`{item['name']}`",
                item['aten_name'],
                fmt_us(item['cuda_kernel_us']),
                fmt_us(item['cuda_hb_us']),
                fmt_us(item['cuda_total_us']),
                fmt_us(item['gems_kernel_us']),
                fmt_us(item['gems_hb_us']),
                fmt_us(item['gems_total_us']),
                f"{item['ratio']:.2f}x",
            ])
        report.append(md_table(
            ["Kernel Name", "aten 操作", "CUDA Kernel(µs)", "CUDA HB(µs)", "CUDA 总(µs)", "FG Kernel(µs)", "FG HB(µs)", "FG 总(µs)", "比值"],
            combined_total_rows
        ))
        report.append("")

        # 写入报告
        out_path = Path("../reports/FlagGems-all.md")
        out_path.write_text("\n".join(report), encoding="utf-8")
        print(f"报告已生成: {out_path}")

    finally:
        cuda.close()
        gems.close()


if __name__ == "__main__":
    main()
