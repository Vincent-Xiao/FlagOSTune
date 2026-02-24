#!/usr/bin/env bash
#
# export-nsys.sh - 导出 Nsys 性能分析结果
#
# 将 .nsys-rep 文件转换为 .sqlite 格式以便后续分析
#
# 用法:
#   ./export-nsys.sh
#   ./export-nsys.sh --dir results/nsys-raw
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_CONFIG="${SCRIPT_DIR}/tools/tool_config.yaml"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# 默认目录
NSYS_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            NSYS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
用法: $0 [--dir DIR]

导出 Nsys 性能分析结果 (.nsys-rep -> .sqlite)

选项:
  --dir DIR   指定包含 .nsys-rep 文件的目录

默认从 tool_config.yaml 读取 nsys_output_dir 配置
EOF
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 获取目录
if [[ -z "$NSYS_DIR" ]]; then
    if [[ -f "$TOOL_CONFIG" ]] && command -v yq &>/dev/null; then
        NSYS_DIR=$(yq '.paths.nsys_output_dir' "$TOOL_CONFIG")
    fi
    [[ -z "$NSYS_DIR" || "$NSYS_DIR" == "null" ]] && NSYS_DIR="results/nsys-raw"
fi

# 处理所有子目录
process_directory() {
    local dir="$1"

    if [[ ! -d "$dir" ]]; then
        log_warn "目录不存在: $dir"
        return
    fi

    log_info "处理目录: $dir"

    # 查找所有 .nsys-rep 文件
    local count=0
    while IFS= read -r -d '' report_file; do
        filename=$(basename -- "$report_file")
        basename="${filename%.*}"

        sqlite_out="${dir}/${basename}.sqlite"

        log_info "  转换: $filename"
        log_info "    -> SQLite: $sqlite_out"

        nsys export --type sqlite --force-overwrite true -o "$sqlite_out" "$report_file" 2>/dev/null || {
            log_warn "转换失败: $filename"
        }

        ((count++)) || true
    done < <(find "$dir" -maxdepth 1 -name "*.nsys-rep" -print0 2>/dev/null)

    if [[ $count -eq 0 ]]; then
        log_warn "未找到 .nsys-rep 文件"
    else
        log_info "已转换 $count 个文件"
    fi
}

cd "$PROJECT_ROOT"

# 处理 nsys-cuda 和 nsys-gems 目录
for subdir in nsys-cuda nsys-gems; do
    if [[ -d "${PROJECT_ROOT}/${subdir}" ]]; then
        process_directory "${PROJECT_ROOT}/${subdir}"
    fi
done

# 处理配置的目录
if [[ -d "$NSYS_DIR" ]]; then
    process_directory "$NSYS_DIR"
fi

log_info "完成!"
