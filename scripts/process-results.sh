#!/usr/bin/env bash
#
# process-results.sh - 处理基准测试结果
#
# 用法:
#   ./process-results.sh                    # 处理所有结果
#   ./process-results.sh --workflow optimized  # 仅处理 optimized 工作流结果
#   ./process-results.sh --workflow benchmark  # 仅处理 benchmark 工作流结果
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROCESSING_DIR="${PROJECT_ROOT}/processing"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

WORKFLOW=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow)
            WORKFLOW="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
用法: $0 [--workflow TYPE]

处理基准测试结果

选项:
  --workflow TYPE   工作流类型: optimized, benchmark, nsys

示例:
  $0                          # 处理所有结果
  $0 --workflow optimized     # 处理 optimized 工作流结果
EOF
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"

# 检查 processing 目录
if [[ ! -d "$PROCESSING_DIR" ]]; then
    log_warn "processing 目录不存在: $PROCESSING_DIR"
    log_info "请先复制处理脚本到 processing/ 目录"
    exit 1
fi

# 检查 bench_stat.py
if [[ ! -f "${PROCESSING_DIR}/bench_stat.py" ]]; then
    log_warn "bench_stat.py 不存在"
    exit 1
fi

log_info "处理基准测试结果..."

case "${WORKFLOW:-all}" in
    optimized)
        log_info "运行 optimized 工作流结果处理..."
        python3 "${PROCESSING_DIR}/bench_stat.py" --f optimized 2>/dev/null || \
            python3 "${PROCESSING_DIR}/bench_stat.py" optimized 2>/dev/null || \
            log_warn "处理失败，请检查 bench_stat.py 参数"
        ;;
    benchmark|full)
        log_info "运行 benchmark 工作流结果处理..."
        python3 "${PROCESSING_DIR}/bench_stat.py" 2>/dev/null || \
            log_warn "处理失败，请检查 bench_stat.py"
        ;;
    nsys)
        log_info "运行 nsys 工作流结果处理..."
        if [[ -f "${PROCESSING_DIR}/perf_analysis_all.py" ]]; then
            python3 "${PROCESSING_DIR}/perf_analysis_all.py"
        else
            log_warn "perf_analysis_all.py 不存在"
        fi
        ;;
    all|*)
        log_info "运行所有结果处理..."
        python3 "${PROCESSING_DIR}/bench_stat.py" 2>/dev/null || true
        ;;
esac

log_info "完成! 结果保存在 reports/ 目录"
