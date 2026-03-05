# FlagTune 用户文档

> vLLM + FlagGems 算子评估自动化框架

---

## 简介

FlagTune 是一套用于自动化评估 vLLM + FlagGems 算子性能的脚本框架。它支持三种工作流模式，帮助您快速完成性能基准测试和分析。此外，脚本搭配了Skills支持，允许配合Claude Code等Agent工具辅助调用脚本并获得调试等帮助。

### 核心功能

- **自动化工作流**: 一键完成服务启动、基准测试、结果收集
- **三种测试模式**: Optimized (快速)、Benchmark (完整)、Nsys Profile (深度分析)
- **灵活配置**: 通过 YAML 配置文件管理模型、服务参数和测试场景
- **结果管理**: 按模型名称自动组织结果和报告
- **算子级测试**: 支持逐个算子进行性能对比

---

## 快速开始

### 1. 安装依赖

```bash
./scripts/setup-deps.sh
```

这将安装以下依赖:
- `jq` - JSON 处理工具
- `tmux` - 会话管理
- `yq` - YAML 处理工具
- `pyyaml` - Python YAML 库

### 2. 配置模型

编辑 `config.yaml`，修改以下关键配置:

```yaml
model:
  path: /models/YOUR-MODEL-PATH    # 模型路径
  name: your-model-name             # 模型名称 (用于结果目录)

serve:
  tensor_parallel_size: 8           # GPU 数量
  gpu_memory_utilization: 0.9       # GPU 显存利用率
```

### 3. 运行测试

```bash
# 快速测试 (CUDA 基准)
./scripts/run-optimized.sh --mode cuda --device 0

# 快速测试 (Gems 基准)
./scripts/run-optimized.sh --mode gems --device 0
```

---

## 配置说明

### config.yaml 结构

```yaml
# 模型配置
model:
  path: /models/Qwen3.5-397B-A17B
  name: qwem3.5
  tensor_parallel_size: 8
  tokenizer_path: null              # null 表示使用 model.path

# 服务配置
serve:
  gpu_memory_utilization: 0.9
  max_num_batched_tokens: 16384
  max_num_seqs: 2048
  trust_remote_code: true
  reasoning_parser: qwen3
  extra_args: ""

# 基准测试配置
benchmark:
  host: 127.0.0.1
  port_base: 2345                   # 实际端口 = port_base + device
  num_runs: 2

  scenarios:
    optimized:                      # 快速测试场景
      - name: p4096d1024
        input_len: 4096
        output_len: 1024
        concurrency: 100

    full:                           # 完整测试场景
      - name: p128d128
        input_len: 128
        output_len: 128
        concurrency: 100
      # ... 更多场景

# 目标算子列表
target_ops:
  - layer_norm
  - softmax
  - gather
  # ...

# 路径配置
paths:
  results: results
  reports: reports
  use_model_name: true              # 结果按模型名分组
```

---

## 工作流详解

### 1. Optimized 工作流

**用途**: 快速验证，仅测试 p4096d1024 场景

```bash
# CUDA 模式
./scripts/run-optimized.sh --mode cuda --device 0

# Gems 模式 (所有算子)
./scripts/run-optimized.sh --mode gems --device 0

# Gems 模式 (逐算子测试)
./scripts/run-optimized.sh --mode gems --device 0 --batch

# Gems 模式 (单个算子)
./scripts/run-optimized.sh --mode gems --device 0 --gems-mode layer_norm
```

### 2. Benchmark 工作流

**用途**: 完整基准测试，包含多种场景

```bash
# CUDA 基准
./scripts/run-benchmark.sh --mode cuda --device 0

# Gems 基准
./scripts/run-benchmark.sh --mode gems --device 0

# 处理结果
./scripts/process-results.sh
```

### 3. Nsys Profile 工作流

**用途**: GPU kernel 级别性能分析

```bash
# CUDA 分析
./scripts/run-nsys-profile.sh --mode cuda --device 0

# Gems 分析
./scripts/run-nsys-profile.sh --mode gems --device 0

# 导出 Nsys 结果
./scripts/export-nsys.sh

# 性能分析
python scripts/tools/perf_analysis.py --nsys_path nsys-raw
```

---

## 命令参数

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode cuda\|gems` | 运行模式 | cuda |
| `--device N` | GPU 设备 ID | 0 |
| `--gems-mode MODE` | FlagGems 模式: all, NULL, 算子名 | all |
| `--batch` | 批量逐算子运行 | false |

### 工作流入口脚本

| 脚本 | 工作流 |
|------|--------|
| `run-optimized.sh` | Optimized |
| `run-benchmark.sh` | Benchmark |
| `run-nsys-profile.sh` | Nsys Profile |

---

## 目录结构

```
FlagTune/
├── config.yaml                    # 用户主配置
├── scripts/
│   ├── setup-deps.sh             # 依赖安装
│   ├── patch-vllm.sh             # vLLM 补丁
│   ├── restore-vllm.sh           # 还原补丁
│   ├── run-workflow.sh           # 核心调度器
│   ├── run-optimized.sh          # Optimized 入口
│   ├── run-nsys-profile.sh       # Nsys 入口
│   ├── run-benchmark.sh          # Benchmark 入口
│   ├── export-nsys.sh            # Nsys 导出
│   ├── process-results.sh        # 结果处理
│   └── tools/
│       ├── tool_config.yaml      # 运行时配置
│       └── benchmark_runner.py   # 基准测试执行器
├── processing/                    # 数据处理脚本
├── results/{model.name}/          # 测试结果
└── reports/{model.name}/          # 分析报告
```

---

## 结果说明

### 输出目录结构

```
results/qwem3.5/                           # 按模型名分组
├── server-logs/                           # 服务器日志
│   └── vllm_bench_gems_all_server.log
├── bench_optimized_log/                   # Optimized 工作流日志
│   └── vllm_bench_gems_all_logs/
│       ├── p4096d1024_run1.log
│       └── p4096d1024_run2.log
├── bench_log/                             # Benchmark 工作流日志
├── nsys-raw/                              # Nsys 原始数据
└── torch-raw/                             # Torch profiler 数据

reports/qwem3.5/                           # 分析报告
└── ...
```

### 查看结果

```bash
# 查看基准测试日志
cat results/qwem3.5/bench_optimized_log/vllm_bench_gems_all_logs/p4096d1024_run1.log

# 处理结果
./scripts/process-results.sh

# 查看报告
ls reports/qwem3.5/
```

---

## vLLM 补丁

### 为什么需要补丁?

在 Gems 模式下，需要修改 vLLM 的 `gpu_model_runner.py` 来启用 FlagGems 算子。

### 补丁操作

```bash
# 查看状态
./scripts/restore-vllm.sh --status

# 应用补丁
./scripts/patch-vllm.sh --method search

# 还原补丁
./scripts/restore-vllm.sh
```

### 注意事项

- CUDA 模式不需要补丁
- 补丁会自动备份原始文件 (`.bak`)
- 补丁位置: `site-packages/vllm/v1/worker/gpu_model_runner.py`

---

## 会话管理

### tmux 命令

```bash
# 查看服务器状态
tmux attach -t screen

# 退出但不停止服务
# 按 Ctrl+B 然后按 D

# 停止服务
tmux send-keys -t screen:bench C-c

# 终止会话
tmux kill-session -t screen
```

---

## 常见问题

### Q: 如何测试单个算子?

```bash
./scripts/run-optimized.sh --mode gems --device 0 --gems-mode layer_norm
```

脚本会自动获取FlagTune所在根目录，因此无需担心执行shell的目录位置。

### Q: OOM 怎么办?

修改 `config.yaml`:
```yaml
serve:
  gpu_memory_utilization: 0.8    # 降低
  max_num_batched_tokens: 8192   # 降低
```

### Q: 如何查看服务器日志?

```bash
# 方法1: tmux
tmux capture-pane -t screen:bench -p -S -100

# 方法2: 日志文件
cat results/qwem3.5/server-logs/*.log | tail -100
```

### Q: 依赖安装失败?

```bash
# 手动安装
sudo apt install jq tmux
pip install pyyaml

# 安装 yq
curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /tmp/yq
chmod +x /tmp/yq
sudo mv /tmp/yq /usr/local/bin/yq
```

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `USE_FLAGOS=1` | 启用 FlagGems |
| `USE_GEMS_MODE` | 算子模式: all, NULL, 或具体算子名 |
| `GEMS_ONCE` | 是否只启用一次 |

---

## 更多帮助

- 使用 `/flagtune:setup` 进行环境配置
- 使用 `/flagtune:debug` 进行问题调试
- 使用 `/flagtune:analyze` 进行结果分析
