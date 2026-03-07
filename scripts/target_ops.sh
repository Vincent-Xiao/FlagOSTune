#!/usr/bin/env bash
#
# target_ops.sh - 目标算子列表
#
# 用法:
#   source scripts/target_ops.sh
#   # 然后可以使用 ${OPS[@]} 数组
#
# 或者在 config.yaml 中配置 target_ops
#

# 默认算子列表
# 可以通过修改此数组或使用 config.yaml 来配置
export OPS=(
    "reciprocal"
    "cat"
    "gather"
    "lt"
    "le"
    "softmax"
    "scatter"
    "cumsum_out"
    "layer_norm"
)

# 从 config.yaml 加载算子列表的函数
load_ops_from_config() {
    local config_file="${1:-config.yaml}"

    if [[ ! -f "$config_file" ]]; then
        echo "警告: 配置文件不存在: $config_file" >&2
        return 1
    fi

    # 使用 yq 解析 YAML
    mapfile -t OPS < <(yq '.target_ops[]' "$config_file" 2>/dev/null)
    export OPS
}

# 打印当前算子列表
print_ops() {
    echo "目标算子列表 (${#OPS[@]} 个):"
    for op in "${OPS[@]}"; do
        echo "  - $op"
    done
}

# 如果直接运行此脚本，打印算子列表
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 检查是否有配置文件参数
    if [[ -n "${1:-}" ]]; then
        load_ops_from_config "$1"
    fi
    print_ops
fi
