#!/usr/bin/env bash
#
# restore-vllm.sh - 还原 gpu_model_runner.py 到原始状态
#
# 用法:
#   ./restore-vllm.sh           # 还原补丁
#   ./restore-vllm.sh --status  # 检查状态
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_CONFIG="${SCRIPT_DIR}/tools/tool_config.yaml"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查状态
check_status() {
    local gpu_runner="$1"
    local backup_file="${gpu_runner}.bak"

    echo "=== vLLM 补丁状态 ==="
    echo ""

    if [[ -f "$backup_file" ]]; then
        echo -e "备份文件: ${GREEN}存在${NC}"
        echo "  路径: $backup_file"
    else
        echo -e "备份文件: ${YELLOW}不存在${NC}"
    fi

    if [[ -f "$gpu_runner" ]]; then
        if grep -q "USE_FLAGOS" "$gpu_runner" 2>/dev/null; then
            echo -e "补丁状态: ${GREEN}已应用${NC}"
        else
            echo -e "补丁状态: ${YELLOW}未应用${NC}"
        fi
    else
        echo -e "目标文件: ${RED}不存在${NC}"
    fi
}

# 主函数
main() {
    local CHECK_STATUS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --status)
                CHECK_STATUS=true
                shift
                ;;
            -h|--help)
                echo "用法: $0 [--status]"
                echo "  --status  检查补丁状态"
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done

    # 从 tool_config.yaml 获取路径
    local gpu_runner=""
    if [[ -f "$TOOL_CONFIG" ]]; then
        gpu_runner=$(yq '.vllm.gpu_model_runner' "$TOOL_CONFIG" 2>/dev/null)
    fi

    # 如果配置中没有，尝试自动检测
    if [[ -z "$gpu_runner" || "$gpu_runner" == "null" || ! -f "$gpu_runner" ]]; then
        local site_packages
        site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
        gpu_runner="${site_packages}/vllm/v1/worker/gpu_model_runner.py"
    fi

    if [[ "$CHECK_STATUS" == "true" ]]; then
        check_status "$gpu_runner"
        exit 0
    fi

    # 执行还原
    local backup_file="${gpu_runner}.bak"

    if [[ ! -f "$backup_file" ]]; then
        log_error "备份文件不存在: $backup_file"
        log_info "无法还原，可能文件从未被修改过"
        exit 1
    fi

    cp "$backup_file" "$gpu_runner"
    log_info "已还原: $gpu_runner"

    rm -f "$backup_file"
    log_info "已删除备份: $backup_file"
}

main "$@"
