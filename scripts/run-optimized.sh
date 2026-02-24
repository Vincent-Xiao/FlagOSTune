#!/usr/bin/env bash
#
# run-optimized.sh - Optimized 工作流入口
#
# 用于优化场景的基准测试
#
# 用法:
#   ./run-optimized.sh --mode cuda --device 0           # CUDA 模式
#   ./run-optimized.sh --mode gems --device 0           # Gems 模式 (all 算子)
#   ./run-optimized.sh --mode gems --device 0 --batch   # Gems 模式 (逐算子)
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

Optimized 工作流 - 优化场景基准测试

选项:
  --mode cuda|gems    运行模式 (默认: cuda)
  --device N          GPU 设备 ID (默认: 0)
  --gems-mode MODE    FlagGems 模式 (默认: all)
  --batch             批量模式，逐算子运行

示例:
  $0 --mode cuda --device 0              # CUDA 基准测试
  $0 --mode gems --device 0              # Gems 基准测试 (所有算子)
  $0 --mode gems --device 0 --batch      # Gems 基准测试 (逐算子)
  $0 --mode gems --device 0 --gems-mode layer_norm  # 单个算子测试
EOF
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 构建参数
ARGS="--mode $MODE --device $DEVICE --gems-mode $GEMS_MODE --optimized"

if [[ "$BATCH" == "true" ]]; then
    ARGS="$ARGS --batch"
fi

# 调用通用工作流
exec "${SCRIPT_DIR}/run-workflow.sh" $ARGS
