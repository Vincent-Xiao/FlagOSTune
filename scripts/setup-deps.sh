#!/usr/bin/env bash
#
# setup-deps.sh - 安装 FlagTune 所需依赖
#
# 用法:
#   ./scripts/setup-deps.sh           # 安装所有依赖
#   ./scripts/setup-deps.sh --check   # 仅检查依赖状态
#

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 依赖列表
APT_DEPS=(
    "sudo"
    "jq"
    "tmux"
    "curl"
)

PIP_DEPS=(
)

# 可选依赖
OPTIONAL_DEPS=(
    "nsys:nsys:NVIDIA Nsight Systems (用于性能分析)"
)

CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            cat << EOF
用法: $0 [--check]

安装 FlagTune 所需依赖

选项:
  --check    仅检查依赖状态，不安装

依赖列表:
  APT:  ${APT_DEPS[*]}
  PIP:  ${PIP_DEPS[*]}

可选依赖:
  - nsys (NVIDIA Nsight Systems)
EOF
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 检查命令是否存在
check_command() {
    local cmd="$1"
    command -v "$cmd" &>/dev/null
}

# 检查 Python 包
check_pip_package() {
    local pkg="$1"
    python3 -c "import ${pkg}" &>/dev/null
}

# 检查 yq (特殊处理，因为它有多种安装方式)
check_yq() {
    if check_command yq; then
        return 0
    fi
    # 检查是否通过 pip 安装
    python3 -c "import yaml; print('ok')" &>/dev/null && \
        python3 -m yq --version &>/dev/null && return 0
    return 1
}

# 安装 yq
install_yq() {
    log_step "安装 yq..."

    local yq_url="https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64"

    log_info "下载 yq 二进制文件..."
    if curl -fsSL "$yq_url" -o /tmp/yq; then
        chmod +x /tmp/yq
        sudo mv /tmp/yq /usr/local/bin/yq
        log_info "yq 已安装到 /usr/local/bin/yq"
        return 0
    fi

    log_warn "yq 安装失败，将使用内置的简单 YAML 解析器"
    return 1
}

# 检查所有依赖状态
check_dependencies() {
    echo ""
    echo "=== 依赖状态检查 ==="
    echo ""

    local missing_apt=()
    local missing_pip=()
    local all_ok=true

    # 检查 APT 依赖
    echo "系统依赖:"
    for dep in "${APT_DEPS[@]}"; do
        if check_command "$dep"; then
            echo -e "  ${GREEN}✓${NC} $dep"
        else
            echo -e "  ${RED}✗${NC} $dep (缺失)"
            missing_apt+=("$dep")
            all_ok=false
        fi
    done

    # 检查 yq
    echo ""
    echo "工具:"
    if check_yq; then
        echo -e "  ${GREEN}✓${NC} yq"
    else
        echo -e "  ${YELLOW}?${NC} yq (可选，建议安装)"
        all_ok=false
    fi

    # 检查 Python 依赖
    echo ""
    echo "Python 依赖:"
    for dep in "${PIP_DEPS[@]}"; do
        if check_pip_package "$dep"; then
            echo -e "  ${GREEN}✓${NC} $dep"
        else
            echo -e "  ${RED}✗${NC} $dep (缺失)"
            missing_pip+=("$dep")
            all_ok=false
        fi
    done

    # 检查可选依赖
    echo ""
    echo "可选依赖:"
    for opt_dep in "${OPTIONAL_DEPS[@]}"; do
        IFS=':' read -r cmd pkg desc <<< "$opt_dep"
        if check_command "$cmd"; then
            echo -e "  ${GREEN}✓${NC} $desc"
        else
            echo -e "  ${YELLOW}-${NC} $desc (未安装)"
        fi
    done

    echo ""

    if [[ "$all_ok" == "true" ]]; then
        log_info "所有必需依赖已安装"
        return 0
    else
        log_warn "部分依赖缺失"
        return 1
    fi
}

# 安装 APT 依赖
install_apt_deps() {
    if [[ ${#APT_DEPS[@]} -eq 0 ]]; then
        return 0
    fi

    log_step "安装 APT 依赖: ${APT_DEPS[*]}"

    # 检查是否有缺失的
    local to_install=()
    for dep in "${APT_DEPS[@]}"; do
        if ! check_command "$dep"; then
            to_install+=("$dep")
        fi
    done

    if [[ ${#to_install[@]} -eq 0 ]]; then
        log_info "APT 依赖已全部安装"
        return 0
    fi

    # 更新并安装
    sudo apt-get update -qq
    sudo apt-get install -y "${to_install[@]}"

    log_info "APT 依赖安装完成"
}

# 安装 PIP 依赖
install_pip_deps() {
    if [[ ${#PIP_DEPS[@]} -eq 0 ]]; then
        return 0
    fi

    log_step "安装 PIP 依赖: ${PIP_DEPS[*]}"

    # 检查是否有缺失的
    local to_install=()
    for dep in "${PIP_DEPS[@]}"; do
        if ! check_pip_package "$dep"; then
            to_install+=("$dep")
        fi
    done

    if [[ ${#to_install[@]} -eq 0 ]]; then
        log_info "PIP 依赖已全部安装"
        return 0
    fi

    pip3 install "${to_install[@]}"

    log_info "PIP 依赖安装完成"
}

# 主函数
main() {
    echo ""
    echo "======================================"
    echo "  FlagTune 依赖安装器"
    echo "======================================"
    echo ""

    # 检查权限
    if [[ "$CHECK_ONLY" == "false" ]] && [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
        log_warn "需要 sudo 权限来安装 APT 依赖"
    fi

    if [[ "$CHECK_ONLY" == "true" ]]; then
        check_dependencies
        exit $?
    fi

    # 安装依赖
    install_apt_deps
    install_pip_deps
    install_yq || true

    # 最终检查
    echo ""
    check_dependencies

    log_info "依赖安装完成!"
}

main "$@"
