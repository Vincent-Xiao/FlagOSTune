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
- `--op mm`
- `--cache-dir /root/.flaggems`
- `--dtypes bfloat16`
- `--warmup 100`
- `--parallel 8`

执行流程：

1. 运行 `FlagTune/processing/shape-gen.py`，从 `FlagTune/shape-config/<model>.txt` 生成 `yaml` 和 `count.yaml`。
2. 对 `mm` / `w8a8_block_fp8_matmul` / `w8a8_block_fp8_matmul_deepgemm` 执行两轮 benchmark：
   `default configuration` 和 `expand configuration`。
3. benchmark 使用：

```bash
pytest benchmark/test_blas_perf_parallel.py \
  -m <op> \
  --shape_file FlagTune/shape-config/<model>.yaml \
  --parallel <parallel> \
  --warmup <warmup>
```

4. 运行 `FlagTune/processing/summary.py` 生成报告，并输出 `*_gain.yaml` 与 `*_lose.yaml`。

常见用法：

```bash
# 查看参数
./FlagTune/scripts/pretune.sh -h

# 指定模型与算子
./FlagTune/scripts/pretune.sh --model Qwen3.5-35B-A3B-p32768d1024 --op mm

# 调整 warmup 和并行卡数
./FlagTune/scripts/pretune.sh --warmup 200 --parallel 4

# 指定 dtype 和 cache 目录
./FlagTune/scripts/pretune.sh --dtypes float16 --cache-dir /tmp/.flaggems

# 跑 fp8 block matmul
./FlagTune/scripts/pretune.sh --op w8a8_block_fp8_matmul
```

输出结果：

- `FlagTune/shape-config/<model>.yaml`
- `FlagTune/shape-config/<model>_count.yaml`
- `FlagTune/shape-config/<model>_gain.yaml`
- `FlagTune/shape-config/<model>_lose.yaml`
- `FlagTune/reports/<model>_<op>.md`
- `FlagTune/reports/<model>_<op>.xlsx`
- `log/flagtune/<model>/<op>/pretune/pretune.log`

## 批量 pretune

批量脚本会遍历 `FlagTune/shape-config/` 下所有 `*.txt`，逐个调用 `pretune.sh`。

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
  - `FlagTune/shape-config/<model>_count.yaml`
- 输出：
  - `FlagTune/reports/<model>_<op>.md`
  - `FlagTune/reports/<model>_<op>.xlsx`
  - `FlagTune/shape-config/<model>_gain.yaml`
  - `FlagTune/shape-config/<model>_lose.yaml`

