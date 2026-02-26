#!/usr/bin/env python3
"""
benchmark_runner.py - FlagTune 基准测试执行器

从 tool_config.yaml 读取所有配置，执行 vLLM 基准测试。
支持多种场景和运行模式。

注意: 所有配置由 run-workflow.sh 从 config.yaml 复制到 tool_config.yaml
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 尝试导入 yaml，如果没有则使用简单的解析
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

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

    if HAS_YAML:
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        # 简单的 YAML 解析
        return parse_simple_yaml(config_path)


def parse_simple_yaml(filepath: Path) -> dict:
    """简单的 YAML 解析器 (仅支持基本结构)"""
    result = {}
    current_section = None
    current_subsection = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 计算缩进
            indent = len(line) - len(line.lstrip())
            line = line.strip()

            if indent == 0 and ':' in line:
                # 顶级 section
                key = line.rstrip(':')
                result[key] = {}
                current_section = key
                current_subsection = None
            elif indent == 2 and ':' in line and current_section:
                # 子 section
                key = line.rstrip(':')
                if isinstance(result[current_section], dict):
                    result[current_section][key] = {}
                    current_subsection = key
            elif indent >= 4 and ':' in line and current_section:
                # 键值对
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()

                # 解析值
                parsed_value = parse_yaml_value(value)

                if current_subsection and isinstance(result[current_section].get(current_subsection), dict):
                    result[current_section][current_subsection][key] = parsed_value
                elif isinstance(result[current_section], dict):
                    result[current_section][key] = parsed_value

    return result


def parse_yaml_value(value: str) -> Any:
    """解析 YAML 值"""
    if not value:
        return None

    # 移除引号
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # 布尔值
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

    # null
    if value.lower() in ('null', '~', ''):
        return None

    # 数字
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def get_scenarios(config: dict) -> list:
    """获取测试场景"""
    optimized = config.get('current_run', {}).get('optimized', False)

    scenario_type = 'optimized' if optimized else 'full'
    scenarios = config.get('benchmark', {}).get('scenarios', {}).get(scenario_type, [])

    # 默认场景
    if not scenarios:
        scenarios = [
            {'name': 'p4096d1024', 'input_len': 4096, 'output_len': 1024, 'concurrency': 100}
        ]

    return scenarios


def build_benchmark_command(scenario: dict, config: dict) -> list:
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

    if nsys_profile or torch_profile:
        # 使用 throughput 模式
        cmd = [
            'vllm', 'bench', 'throughput',
            '--tensor-parallel-size', str(tensor_parallel_size),
            '--model', model_path,
            '--tokenizer', tokenizer_path,
            '--dataset-name', 'random',
            '--trust-remote-code',
            '--input-len', str(input_len),
            '--output-len', str(output_len),
            '--num-prompts', str(concurrency),
        ]
    else:
        # 使用 serve 模式
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

    # 获取日志目录
    log_dir = Path(config.get('paths', {}).get('log_dir', 'results/bench-logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_run{run_id}.log"

    cmd = build_benchmark_command(scenario, config)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Starting scenario: {name} (Run {run_id})")
    print(f"    Input: {input_len}, Output: {output_len}, Concurrency: {concurrency}")
    print(f"    Logging to: {log_file}")
    print(f"    Command: {' '.join(cmd)}\n")

    # Torch profiler 支持
    prof = None
    profiler_warmup = num_runs - 1

    if HAS_TORCH and run_id == profiler_warmup + 1:
        if nsys_profile:
            print("Start nsys profiling...")
            torch.cuda.profiler.start()
        elif torch_profile:
            torch_report_dir = Path(config.get('paths', {}).get('torch_output_dir', 'results/torch-raw'))
            torch_report_dir.mkdir(parents=True, exist_ok=True)

            mode = current_run.get('mode', 'cuda')
            gems_mode = current_run.get('gems', {}).get('mode', '')
            gems_suffix = f"-{gems_mode}" if mode == 'gems' else ""
            torch_report = torch_report_dir / f"report-{mode}{gems_suffix}.json"

            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False,
                with_stack=True,
                with_flops=True
            )
            print(f"Start torch profiling (Output: {torch_report})...")

    # 执行命令
    ctx = contextlib.nullcontext()
    if HAS_TORCH and nsys_profile and run_id == profiler_warmup + 1:
        ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
        torch.cuda.nvtx.range_push(f"Model_Chat_Step_{run_id}")

    try:
        with ctx:
            if HAS_TORCH and torch_profile and run_id == profiler_warmup + 1 and prof:
                with prof:
                    with open(log_file, 'w') as f:
                        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            else:
                with open(log_file, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    finally:
        if HAS_TORCH and (nsys_profile or torch_profile) and run_id == profiler_warmup + 1:
            torch.cuda.nvtx.range_pop()

    # 停止 profiler
    if HAS_TORCH and run_id == profiler_warmup + 1:
        if nsys_profile:
            torch.cuda.profiler.stop()
            print("Stop nsys profiling...")
        elif torch_profile and prof:
            mode = current_run.get('mode', 'cuda')
            gems_mode = current_run.get('gems', {}).get('mode', '')
            gems_suffix = f"-{gems_mode}" if mode == 'gems' else ""
            torch_report = Path(config.get('paths', {}).get('torch_output_dir', 'results/torch-raw')) / f"report-{mode}{gems_suffix}.json"
            prof.export_chrome_trace(str(torch_report))
            print(f"Stop torch profiling, trace exported to: {torch_report}")

    status = "Success" if result.returncode == 0 else "Failed"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {status}: {name} Run {run_id} (exit code: {result.returncode})\n")


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

    # 获取场景
    scenarios = get_scenarios(config)
    num_runs = config.get('benchmark', {}).get('num_runs', 4)

    print(f"\nStarting vLLM benchmark suite for {len(scenarios)} scenarios, each repeated {num_runs} times...\n")

    # 运行所有场景
    for scenario in scenarios:
        for run_id in range(1, num_runs + 1):
            run_benchmark(scenario, run_id, config)

    print("=" * 60)
    print("All scenarios and runs completed.")
    log_dir = config.get('paths', {}).get('log_dir', 'results/bench-logs')
    print(f"Logs saved in: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
