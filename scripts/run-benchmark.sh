#!/usr/bin/env bash
#
# run-benchmark.sh - Benchmark 工作流入口
#
# 用于完整场景的标准基准测试
#
# 用法:
#   ./run-benchmark.sh --mode cuda --device 0           # CUDA 模式
#   ./run-benchmark.sh --mode gems --device 0           # Gems 模式
#   ./run-benchmark.sh --mode gems --device 0 --batch   # Gems 模式 (逐算子)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
MODE="cuda"
DEVICE=0
GEMS_MODE="all"
BATCH=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gems-mode)
            GEMS_MODE="$2"
            shift 2
            ;;
        --batch)
            BATCH=true
            shift
            ;;
        -h|--help)
            cat << EOF
用法: $0 [选项]

Benchmark 工作流 - 完整场景基准测试

选项:
  --mode cuda|gems    运行模式 (默认: cuda)
  --device N          GPU 设备 ID (默认: 0)
  --gems-mode MODE    FlagGems 模式 (默认: all)
  --batch             批量模式，逐算子运行

示例:
  $0 --mode cuda --device 0              # CUDA 基准测试
  $0 --mode gems --device 0              # Gems 基准测试 (所有算子)
  $0 --mode gems --device 0 --batch      # Gems 基准测试 (逐算子)

完成后运行:
  ./scripts/process-results.sh           # 处理结果
EOF
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 构建参数 (不使用 optimized，运行完整场景)
ARGS="--mode $MODE --device $DEVICE --gems-mode $GEMS_MODE"

if [[ "$BATCH" == "true" ]]; then
    ARGS="$ARGS --batch"
fi

# 调用通用工作流
exec "${SCRIPT_DIR}/run-workflow.sh" $ARGS
