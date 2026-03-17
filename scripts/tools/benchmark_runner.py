#!/usr/bin/env python3
"""
benchmark_runner.py - FlagTune 基准测试执行器

从 tool_config.yaml 读取所有配置，执行 vLLM 基准测试。
支持多种场景和运行模式。

注意: 所有配置由 run-workflow.sh 从 config.yaml 复制到 tool_config.yaml
"""

import subprocess
import sys
import time
import ast
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 尝试导入 torch
try:
    import torch
    import contextlib
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    contextlib = None


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """加载配置 (仅从 tool_config.yaml)"""
    config_path = get_project_root() / "scripts" / "tools" / "tool_config.yaml"

    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_scenarios(config: dict) -> list:
    """获取测试场景"""
    current_run = config.get('current_run', {})
    scenario_type = current_run.get('scenario_type')

    # 兼容旧配置：如果没有 scenario_type，则回退到 optimized 布尔值
    if not scenario_type:
        optimized = current_run.get('optimized', False)
        scenario_type = 'optimized' if optimized else 'full'

    scenarios = config.get('benchmark', {}).get('scenarios', {}).get(scenario_type, [])

    # 默认场景
    if not scenarios:
        scenarios = [
            {'name': 'p4096d1024', 'input_len': 4096, 'output_len': 1024, 'concurrency': 64},
        ]

    return scenarios


def build_benchmark_command(scenario: dict, config: dict, is_last_run: bool = False) -> list:
    """构建基准测试命令"""
    current_run = config.get('current_run', {})
    nsys_profile = current_run.get('nsys_profile', False)
    torch_profile = current_run.get('torch_profile', False)

    host = config.get('benchmark', {}).get('host', '127.0.0.1')
    port = current_run.get('port', 2345)
    model_path = config.get('model', {}).get('path', '')
    model_name = config.get('model', {}).get('name', 'model')
    tokenizer_path = config.get('model', {}).get('tokenizer_path') or model_path
    tensor_parallel_size = config.get('model', {}).get('tensor_parallel_size', 8)

    input_len = scenario.get('input_len', 1024)
    output_len = scenario.get('output_len', 1024)
    concurrency = scenario.get('concurrency', 100)

    if torch_profile:
        # Torch profiling 使用 throughput 模式
        torch_report_dir = Path(
            config.get('paths', {}).get('torch_output_dir',
                                        'results/torch-raw'))
        mode = current_run.get('mode', 'cuda')
        gems_mode = current_run.get('gems', {}).get('mode', '')
        gems_suffix = f"-{gems_mode}" if mode == 'gems' else ""
        torch_report_dir = torch_report_dir / f"report-{mode}{gems_suffix}"
        torch_report_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            'vllm',
            'bench',
            'throughput',
            '--tensor-parallel-size',
            str(tensor_parallel_size),
            '--model',
            model_path,
            '--tokenizer',
            tokenizer_path,
            '--dataset-name',
            'random',
            '--trust-remote-code',
            '--input-len',
            str(input_len),
            '--output-len',
            str(output_len),
            '--num-prompts',
            str(concurrency),
        ]
        if is_last_run:
            cmd += [
                "--profile",
                "--profiler-config",
                f'{{"profiler": "torch","torch_profiler_dir":"{torch_report_dir}", "torch_profiler_with_stack": true, "torch_profiler_with_flops": true, "torch_profiler_use_gzip": false, "torch_profiler_dump_cuda_time_total": true, "torch_profiler_record_shapes": true, "torch_profiler_with_memory": false, "ignore_frontend": true, "delay_iterations": 2, "max_iterations": 0}}',
            ]
    else:
        # 普通模式使用 serve 模式
        cmd = [
            'vllm', 'bench', 'serve',
            '--host', host,
            '--port', str(port),
            '--backend', 'openai-chat',
            '--model', model_name,
            '--tokenizer', tokenizer_path,
            '--dataset-name', 'random',
            '--endpoint', '/v1/chat/completions',
            '--ignore-eos',
            '--trust-remote-code',
            '--random-input-len', str(input_len),
            '--random-output-len', str(output_len),
            '--num-prompts', str(concurrency),
            '--max-concurrency', str(concurrency),
        ]
        # if is_last_run:
        #     cmd.append('--profile')

    return cmd


def run_benchmark(scenario: dict, run_id: int, config: dict):
    """运行单个基准测试"""
    name = scenario.get('name', 'unknown')
    input_len = scenario.get('input_len', 1024)
    output_len = scenario.get('output_len', 1024)
    concurrency = scenario.get('concurrency', 100)

    current_run = config.get('current_run', {})
    nsys_profile = current_run.get('nsys_profile', False)
    torch_profile = current_run.get('torch_profile', False)
    num_runs = config.get('benchmark', {}).get('num_runs', 4)

    # 判断是否为最后一轮
    is_last_run = (run_id == num_runs)
    # 获取日志目录
    log_dir = Path(config.get('paths', {}).get('log_dir', 'results/bench-logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_run{run_id}.log"

    cmd = build_benchmark_command(scenario, config, is_last_run)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Starting scenario: {name} (Run {run_id})")
    print(f"    Input: {input_len}, Output: {output_len}, Concurrency: {concurrency}")
    print(f"    Logging to: {log_file}")
    print(f"    Command: {' '.join(cmd)}\n")

    if HAS_TORCH and torch_profile and is_last_run:
        print(f"Starting torch profiling in vllm subprocess")

    # 执行命令
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    # 停止 profiler
    if HAS_TORCH and torch_profile and is_last_run:
        print(f"vllm subprocess profiling complete. ")

    status = "Success" if result.returncode == 0 else "Failed"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {status}: {name} Run {run_id} (exit code: {result.returncode})\n")


def append_shape_scenario_marker(scenario: dict, config: dict) -> None:
    """在 shape 场景下，记录每个 benchmark 在 gems-all.txt 的开始行号。"""
    if not should_append_shape_marker(config):
        return

    model_name = str(config.get('model', {}).get('name', 'unknown-model'))
    scenario_name = str(scenario.get('name', 'unknown-scenario'))

    project_root = get_project_root()
    shape_dir = project_root / 'results' / model_name / 'gems-config-shape'
    shape_dir.mkdir(parents=True, exist_ok=True)

    gems_file = get_shape_gems_output_file(shape_dir, config)
    marker_file = shape_dir / 'marker.txt'

    # 静默窗口检测: gems-all.txt 在一段时间内 size/mtime 均不变化，认为当前写入已稳定
    if gems_file.exists():
        stable_rounds = 10
        stable_interval_sec = 0.2
        timeout_sec = 10.0
        stable_count = 0
        last_sig = None
        start_ts = time.time()

        while time.time() - start_ts < timeout_sec:
            st = gems_file.stat()
            cur_sig = (st.st_size, st.st_mtime_ns)

            if cur_sig == last_sig:
                stable_count += 1
                if stable_count >= stable_rounds:
                    break
            else:
                stable_count = 0
                last_sig = cur_sig

            time.sleep(stable_interval_sec)
        else:
            print(f"[WARN] gems 文件静默窗口检测超时，继续按当前内容记录行号: {gems_file}")

    # 记录 scenario 开始前 gems-all.txt 的当前行号
    start_line = 0
    if gems_file.exists():
        with gems_file.open('r', encoding='utf-8', errors='ignore') as f:
            start_line = sum(1 for _ in f)

    marker = f"{model_name}-{scenario_name}：{start_line}\n"
    with marker_file.open('a', encoding='utf-8') as f:
        f.write(marker)

    print(f"[INFO] shape 标记已记录: {marker.strip()} -> {marker_file}")


def get_shape_gems_output_file(shape_dir: Path, config: dict) -> Path:
    """根据 current_run.gems.mode 计算 shape 场景实际写入的 gems 文件路径。"""
    gems_mode = str(config.get('current_run', {}).get('gems', {}).get('mode', 'all'))

    if gems_mode in {"all", "mm", "NULL"}:
        filename = f"gems-{gems_mode}.txt"
        return shape_dir / filename

    try:
        parsed = ast.literal_eval(gems_mode)
        keep_ops = parsed if isinstance(parsed, list) else [parsed]
        filename = f"gems-{keep_ops}.txt"
    except (ValueError, SyntaxError):
        filename = f"gems-{gems_mode}.txt"

    return shape_dir / filename


def reset_shape_marker_file(config: dict) -> None:
    """在 shape 场景运行开始前清理 marker.txt。"""
    if not should_append_shape_marker(config):
        return

    model_name = str(config.get('model', {}).get('name', 'unknown-model'))
    marker_file = get_project_root() / 'results' / model_name / 'gems-config-shape' / 'marker.txt'

    if marker_file.exists():
        marker_file.unlink()
        print(f"[INFO] 已删除旧 marker 文件: {marker_file}")


def should_append_shape_marker(config: dict) -> bool:
    """仅在 shape 场景且 gems.once=false 时返回 True。"""
    current_run = config.get('current_run', {})
    scenario_type = current_run.get('scenario_type', '')
    if scenario_type != 'shape':
        return False

    once_val = current_run.get('gems', {}).get('once', True)
    if isinstance(once_val, str):
        once_bool = once_val.strip().lower() == 'true'
    else:
        once_bool = bool(once_val)

    return not once_bool


def main():
    print("=" * 60)
    print("FlagTune Benchmark Runner")
    print("=" * 60)

    # 加载配置 (仅从 tool_config.yaml)
    config = load_config()

    print(f"Mode: {config.get('current_run', {}).get('mode', 'cuda')}")
    print(f"Device: {config.get('current_run', {}).get('device', 0)}")
    print(f"Port: {config.get('current_run', {}).get('port', 2345)}")
    print(f"Model: {config.get('model', {}).get('name', 'unknown')}")

    # 每次 shape 运行开始前先清空 marker 文件
    reset_shape_marker_file(config)

    # 获取场景
    scenarios = get_scenarios(config)
    num_runs = config.get('benchmark', {}).get('num_runs', 4)

    print(f"\nStarting vLLM benchmark suite for {len(scenarios)} scenarios, each repeated {num_runs} times...\n")

    # 运行所有场景
    for scenario in scenarios:
        # 每个 shape scenario benchmark 开始前追加一次标记
        append_shape_scenario_marker(scenario, config)

        for run_id in range(1, num_runs + 1):
            run_benchmark(scenario, run_id, config)

    print("=" * 60)
    print("All scenarios and runs completed.")
    log_dir = config.get('paths', {}).get('log_dir', 'results/bench-logs')
    print(f"Logs saved in: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
