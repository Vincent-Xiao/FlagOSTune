# Flagtune 使用说明

## FlagGems Pretune 安装
```bash
git clone https://github.com/Vincent-Xiao/FlagGems.git
cd FlagGems && git checkout v4.2.1.rc.0_flagtune
pip install scikit-build-core>=0.11 pybind11 ninja setuptools_scm py-test openpyxl
pip install --no-build-isolation -v -e .
```

## 以 qwen3.5 为例说明pretune执行流程


### 生成模型推理shape日志
模型推理时使用flaggems的API中开启shape日志功能：

```python
flag_gems.enable(record=True, once=False, path="./qwen3.5.txt")
```

将日志文件 `qwen3.5.txt` 放置到 `flagtune/shape-config/` 目录下。


### 一键标准pretune流程
执行
```bash
./flagtune/pretune.sh --model qwen3.5 --op mm --best true
```
pretune完毕后所有文件为：

```text
flagtune/
├── pretune.sh                          # 流程入口
├── processing/
│   ├── shape-gen.py                    # 生成 shape YAML
│   ├── config_replace.py               # 替换 mm 配置
│   └── summary.py                      # 汇总报告 + gain/lose 分流
├── tune-config/
│   └── mm_hopper_tma.yaml              # mm expand 配置
├── shape-config/
│   ├── qwen3.5.txt                     # 输入日志（模型推理生成shape日志）
│   ├── qwen3.5.yaml                    # 4维 shape
│   ├── qwen3.5_count.yaml              # 5维 shape + count
│   ├── qwen3.5_gain.yaml               # 收益 shape（gain>0）
│   └── qwen3.5_lose.yaml               # 非收益 shape（gain<=0）
└── reports/
    ├── qwen3.5_mm.md                   # Markdown 报告
    └── qwen3.5_mm.xlsx                 # Excel 报告
log/flagtune/qwen3.5/mm/pretune/pretune.log  # pretune 执行日志
/root/.flaggems/                             # pretune 生成的缓存数据库文件
```
### 模型推理调用pretune
pretune数据库文件默认保存在 `~/.flaggems/` 目录下，关闭模型推理的日志能力，执行正常的推理即可
```python
flag_gems.enable(record=True, once=True, path="./qwen3.5.txt")
```
---
## 目录结构与文件说明

```text
flagtune/
├── pretune.sh 一件式pretune流程入口脚本
├── processing/
│   ├── config_replace.py config替换脚本
│   ├── shape-gen.py 生成 shape 配置脚本
│   └── summary.py 汇总报告脚本
└── tune-config/
    └── mm_hopper_tma.yaml hopper架构mm算子扩展config配置
```

### `flagtune/pretune.sh`
- `flagtune` 的主入口脚本，负责执行完整pretune流程。
- 主要能力：
  - 生成 模型的shape yaml配置
  - 执行 `default configuration` / `expand configuration` 两轮 benchmark
  - 生成pretune 报告到 `flagtune/reports/`目录（markdown和excel）
  - 根据报告中的收益情况，生成 `gain/lose.yaml`，gain代表`expand configuration`有收益的shape，lose代表无收益的`shape`。
  - 可选执行 `best` 阶段（`--best`）：
    - 用 `lose.yaml` pretune `default configuration`
    - 用 `gain.yaml` pretune `best expand`
- 关键参数：
  - `--model`（默认 `qwen3.5`）
  - `--op`（默认 `mm`）
  - `--cache-dir`（默认 `/root/.flaggems`）,用于清理缓存，确保 pretune 结果的准确性。
  - `--best`（默认 `true`），是否执行最佳调优策略，分别`default configuration` / `expand configuration`对应的shape pretune，确保算子的所有shape执行pretune都是最优的

### `flagtune/processing/shape-gen.py`
- 输入：`flagtune/shape-config/{model}.txt`（模型推理debug日志）
- 输出：
  - `flagtune/shape-config/{model}.yaml`（4维：`B,M,N,K`）
  - `flagtune/shape-config/{model}_count.yaml`（5维：`B,M,N,K,Count`），`Count` 代表该 shape 在模型中被调用的频次。
- 支持 `--op` 过滤，产出 `{model}_{op}.yaml` 与 `{model}_{op}_count.yaml`。

### `flagtune/processing/config_replace.py`
- 根据 `tune-config` 中的参数，替换 `mm.py` 中 `matmul_get_configs` 实现。
- 典型用途：`expand` 阶段动态注入更激进的 `BLOCK_M/N/K`、`stages`、`warps` 组合。

### `flagtune/processing/summary.py`
- 从 `pretune.log` 抽取 `default/expand` 两阶段性能数据。
- 生成：
  - Markdown 报告：`flagtune/reports/{model}_{op}.md`
  - Excel 报告：`flagtune/reports/{model}_{op}.xlsx`
  - 按收益分流 shape：
    - `flagtune/shape-config/{model}_gain.yaml`（`Speedup Gain > 0`）
    - `flagtune/shape-config/{model}_lose.yaml`（`Speedup Gain <= 0`）
- 同时保持与 `flagtune/shape-config/{model}.yaml` 一致的算子顺序、shape 顺序。

### `flagtune/tune-config/mm_hopper_tma.yaml`
- `mm` 调优配置源，供 `config_replace.py` 读取并注入。
- 典型字段：`block_m`、`block_n`、`block_k`、`stages`、`warps`。

---


### 常见命令

```bash
# 查看入口脚本参数
bash flagtune/pretune.sh -h

# 关闭 best 阶段
./flagtune/pretune.sh --model qwen3.5 --op mm --best false

# 指定 cache 目录
./flagtune/pretune.sh --model qwen3.5 --op mm --cache-dir /tmp/.flaggems
```

---
