# FlagTune 使用说明

## 安装

```bash
git clone https://github.com/Vincent-Xiao/FlagGems.git
cd FlagGems
pip install scikit-build-core>=0.11 pybind11 ninja setuptools_scm py-test openpyxl
pip install --no-build-isolation -v -e .
git clone https://vincent-gitea.iepose.cn/FlagOS/FlagTune.git -b flaggems
```

## 目录说明

```text
FlagTune/
├── README.md
├── processing/
│   ├── shape-gen.py
│   └── summary.py
├── scripts/
│   ├── pretune.sh
│   └── pretune_batch.sh
└── shape-config/
    ├── <model>.txt
    ├── <model>.yaml
    ├── <model>_count.yaml
    ├── <model>_gain.yaml
    └── <model>_lose.yaml

log/
└── flagtune/<model>/<op>/pretune/pretune.log
```

- `FlagTune/shape-config/*.txt`：模型推理阶段采集到的原始 shape 日志。
- `FlagTune/shape-config/*.yaml`：由 `shape-gen.py` 生成的 shape 配置。
- `FlagTune/shape-config/*_count.yaml`：附带 shape 频次统计的配置。
- `FlagTune/shape-config/*_gain.yaml` / `*_lose.yaml`：根据报告拆分出的收益 shape 与非收益 shape。
- `FlagTune/reports/`：`summary.py` 生成的 Markdown / Excel 报告目录，首次运行时自动创建。
- `log/flagtune/.../pretune.log`：pretune 全流程日志。

## Shape 日志准备

模型推理时开启 shape 日志：

```python
flag_gems.enable(record=True, once=False, path="./Qwen3.5-35B-A3B-p32768d1024.txt")
```

将生成的日志文件放到 `FlagTune/shape-config/` 下，例如：

```text
FlagTune/shape-config/Qwen3.5-35B-A3B-p32768d1024.txt
```

## 单模型 pretune

入口脚本：

```bash
./FlagTune/scripts/pretune.sh
```

当前默认参数：

- `--model Qwen3.5-35B-A3B-p32768d1024`
- `--yaml Qwen3.5-35B-A3B-p32768d1024`
- `--clear-cache false`
- `--op mm`
- `--cache-dir /root/.flaggems`
- `--dtypes bfloat16`
- `--warmup 100`
- `--parallel 8`

参数行为：

- 显式传 `--model` 时，走 model 模式，先执行 `shape-gen.py`。
- 未传 `--model`、但显式传 `--yaml` 时，走 yaml 模式，直接加载 `FlagTune/shape-config/<yaml>.yaml`。
- `--model` 和 `--yaml` 都不传时，默认走 model 模式。
- 传入 `--master` 时，会额外使用 `master` 分支与当前分支做对比，两侧都使用 default configuration。
- 传入 `--clear-cache` 时，每次 benchmark 前都会删除 `cache-dir` 指向的 Flaggems cache。

执行流程：

1. model 模式下运行 `FlagTune/processing/shape-gen.py`，从 `FlagTune/shape-config/<model>.txt` 生成 `yaml` 和 `count.yaml`；yaml 模式下跳过这一步。
2. 对任意 `--op` 执行两轮 benchmark：
   默认模式下为 `default configuration` 和 `expand configuration`；
   `--master` 模式下为 `master` 和当前分支名，两侧都使用 default configuration。
3. benchmark 使用：

```bash
pytest benchmark/test_blas_perf_parallel.py \
  -m <op> \
  --shape_file FlagTune/shape-config/<shape_file>.yaml \
  --parallel <parallel> \
  --warmup <warmup>
```

4. 运行 `FlagTune/processing/summary.py` 生成报告，并输出 `*_gain.yaml` 与 `*_lose.yaml`。
   如果不存在 `*_count.yaml`，report 仍会生成，`Count` 列回退为 `1`，Excel 中不会生成 `Sorted by Count` sheet。

常见用法：

```bash
# 查看参数
./FlagTune/scripts/pretune.sh -h

# 指定模型与算子
./FlagTune/scripts/pretune.sh --model Qwen3.5-35B-A3B-p32768d1024 --op mm

# 直接使用已有 yaml，不再执行 shape-gen.py
./FlagTune/scripts/pretune.sh --yaml Qwen3.5-35B-A3B-p32768d1024_default_bad_case --op mm

# 对比 master 分支和当前分支
./FlagTune/scripts/pretune.sh --yaml Qwen3.5-35B-A3B-p32768d1024_default_bad_case --master --op w8a8_block_fp8_matmul

# 对比 master 分支和当前分支，并在每轮 benchmark 前清理 cache
./FlagTune/scripts/pretune.sh --model Qwen3.5-35B-A3B-p32768d1024 --op w8a8_block_fp8_matmul --master --clear-cache

# 调整 warmup 和并行卡数
./FlagTune/scripts/pretune.sh --warmup 200 --parallel 4

# 指定 dtype 和 cache 目录
./FlagTune/scripts/pretune.sh --dtypes float16 --cache-dir /tmp/.flaggems

# 跑 fp8 block matmul
./FlagTune/scripts/pretune.sh --op w8a8_block_fp8_matmul
```

输出结果：

- model 模式：
  - `FlagTune/shape-config/<model>.yaml`
  - `FlagTune/shape-config/<model>_count.yaml`
  - `FlagTune/shape-config/<model>_gain.yaml`
  - `FlagTune/shape-config/<model>_lose.yaml`
  - `FlagTune/reports/<model>_<op>.md`
  - `FlagTune/reports/<model>_<op>.xlsx`
  - `log/flagtune/<model>/<op>/pretune/pretune.log`
- yaml 模式：
  - 输入 yaml 会被直接使用，例如 `FlagTune/shape-config/<yaml>.yaml`
  - 输出报告与日志路径中的名字使用 `<yaml>`
  - 如果不存在 `FlagTune/shape-config/<yaml>_count.yaml`，则 report 中 `Count=1`
- `--master` 模式：
  - 使用临时 `git worktree` 在 `master` 分支执行 benchmark，避免切换当前工作区分支
  - report 中左侧列名为 `master`，右侧列名为当前分支名
  - 输出报告文件会增加 `_master` 后缀，例如 `FlagTune/reports/<name>_<op>_master.md`
- `--clear-cache` 模式：
  - 每次 `run_mm_benchmark` 执行前都会删除 `--cache-dir` 指向的缓存目录

## 批量 pretune

批量脚本会遍历 `FlagTune/shape-config/` 下所有 `*.txt`，逐个调用 `pretune.sh --model`。
批量模式只使用 `--model`，不会使用 `--yaml`。

```bash
./FlagTune/scripts/pretune_batch.sh
```

支持参数：

- `--op`
- `--cache-dir`
- `--dtypes`
- `--warmup`
- `--parallel`

示例：

```bash
./FlagTune/scripts/pretune_batch.sh --op mm --warmup 200 --parallel 4
./FlagTune/scripts/pretune_batch.sh --op w8a8_block_fp8_matmul --parallel 8
```

## 相关脚本

### `FlagTune/processing/shape-gen.py`

- 输入：`FlagTune/shape-config/<model>.txt`
- 输出：
  - `FlagTune/shape-config/<model>.yaml`
  - `FlagTune/shape-config/<model>_count.yaml`
- 支持 `--op` 过滤，按算子单独生成 `yaml`。

### `FlagTune/processing/summary.py`

- 输入：
  - `log/flagtune/<model>/<op>/pretune/pretune.log`
  - `FlagTune/shape-config/<model>.yaml`
  - `FlagTune/shape-config/<model>_count.yaml`，可选
- 输出：
  - `FlagTune/reports/<model>_<op>.md`
  - `FlagTune/reports/<model>_<op>.xlsx`
  - `FlagTune/shape-config/<model>_gain.yaml`
  - `FlagTune/shape-config/<model>_lose.yaml`
- 传入 `--output-suffix _master` 等参数时，报告文件名会附加对应后缀
- 当 `*_count.yaml` 缺失时：
  - Markdown / Excel report 仍会生成
  - `Count` 列回退为 `1`
  - Excel 不生成 `Sorted by Count` sheet
