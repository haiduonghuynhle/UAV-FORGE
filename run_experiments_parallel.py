#!/usr/bin/env python3
"""
Run multiple experiments concurrently to maximize hardware utilization (e.g., MIG 20GB VRAM).

Usage:
  python run_experiments_parallel.py --datasets vpair avl --pipelines ours ours_cascade --max-workers 3 --device cuda

This script will run:
  benchmark_runner.py --dataset vpair --pipeline ours --device cuda
  benchmark_runner.py --dataset vpair --pipeline ours_cascade --device cuda
  benchmark_runner.py --dataset avl --pipeline ours --device cuda
  benchmark_runner.py --dataset avl --pipeline ours_cascade --device cuda
in parallel, limiting to `--max-workers` concurrent processes.
"""

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

# We can import from benchmark_runner if needed, but keeping it standalone is fine
# to avoid heavy ML imports before spawning processes.
try:
    from benchmark_runner import DATASET_CONFIGS, PIPELINE_CONFIGS
except ImportError:
    DATASET_CONFIGS = {'vpair': {}, 'avl': {}, 'uav_visloc': {}}
    PIPELINE_CONFIGS = {'ours': {}, 'ours_cascade': {}, 'roma_every_frame': {}}


def build_benchmark_command(dataset: str, pipeline: str, args: argparse.Namespace) -> list[str]:
    """Build the strict end-to-end benchmark command for one dataset/pipeline."""
    cmd = [
        sys.executable,
        "benchmark_runner.py",
        "--dataset", dataset,
        "--pipeline", pipeline,
        "--output", args.output,
        "--device", args.device,
        "--eval-mode", "end_to_end",
        "--uav-visloc-protocol", args.uav_visloc_protocol,
        "--strict-components",
        "--strict-end2end"
    ]

    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
        
    if args.max_samples_percent is not None:
        cmd.extend(["--max-samples-percent", str(args.max_samples_percent)])
        
    if args.test_interval is not None:
        cmd.extend(["--test-interval", str(args.test_interval)])
    if args.finetuned:
        cmd.append("--finetuned")

    return cmd


def run_experiment(dataset: str, pipeline: str, args: argparse.Namespace, log_dir: Path) -> dict:
    """Run a single experiment configuration as a subprocess."""
    cmd = build_benchmark_command(dataset, pipeline, args)
    cmd_str = " ".join(cmd)
    
    # Setup log files for this specific run so they don't clobber the console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_base = log_dir / f"{dataset}_{pipeline}_{timestamp}"
    out_file = log_file_base.with_suffix(".out")
    err_file = log_file_base.with_suffix(".err")

    print(f"[START] {dataset} | {pipeline}")
    print(f"        Command: {cmd_str}")
    print(f"        Logs: {out_file.name}, {err_file.name}")

    start_time = datetime.now()
    
    try:
        with open(out_file, "w") as f_out, open(err_file, "w") as f_err:
            process = subprocess.run(
                cmd,
                stdout=f_out,
                stderr=f_err,
                text=True,
                check=False
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if process.returncode == 0:
            print(f"[DONE]  {dataset} | {pipeline} (Took {duration:.1f}s)")
            return {"status": "success", "dataset": dataset, "pipeline": pipeline, "duration": duration, "returncode": 0}
        else:
            print(f"[FAIL]  {dataset} | {pipeline} (Took {duration:.1f}s, Return Code: {process.returncode})")
            print(f"        Check errors in: {err_file}")
            return {"status": "failed", "dataset": dataset, "pipeline": pipeline, "duration": duration, "returncode": process.returncode}
            
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[ERROR] {dataset} | {pipeline} failed to launch: {e}")
        return {"status": "error", "dataset": dataset, "pipeline": pipeline, "duration": duration, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run multiple localization pipelines concurrently")
    
    # Accept lists instead of single items
    parser.add_argument('--datasets', nargs='+', type=str, default=['vpair'],
                        help="List of datasets to run (e.g., vpair avl). Use 'all' for all configured datasets.")
    parser.add_argument('--pipelines', nargs='+', type=str, default=['ours', 'ours_cascade'],
                        help="List of pipelines to run (e.g., ours ours_cascade). Use 'all' for all configured pipelines.")
    
    # Concurrency control
    parser.add_argument('--max-workers', type=int, default=2,
                        help="Max concurrent tasks. Default is 2 (suitable for 20GB VRAM if each takes ~6GB).")
    
    # Pass-through arguments to benchmark_runner.py
    parser.add_argument('--output', type=str, default='results_parallel',
                        help='Output directory for the benchmark results')
    parser.add_argument('--log-dir', type=str, default='logs_parallel',
                        help='Directory to store stdout/stderr logs of the concurrent processes')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per dataset (for quick testing)')
    parser.add_argument('--max-samples-percent', type=int, default=None,
                        help='Percentage of dataset to run')
    parser.add_argument('--test-interval', type=int, default=None,
                        help='Sample every N-th frame')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Compute device')
    parser.add_argument('--finetuned', action='store_true',
                        help='Use finetuned model checkpoints')
    parser.add_argument('--uav-visloc-protocol', type=str, default='intra_flight',
                        choices=['intra_flight', 'cross_flight', 'all_maps'],
                        help='UAV-VisLoc retrieval protocol passed to benchmark_runner.py')
    
    args = parser.parse_args()

    # Resolve datasets
    if 'all' in args.datasets:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets
        
    # Resolve pipelines
    if 'all' in args.pipelines:
        pipelines = list(PIPELINE_CONFIGS.keys())
    else:
        pipelines = args.pipelines

    # Validate
    invalid_ds = [ds for ds in datasets if ds not in DATASET_CONFIGS]
    if invalid_ds:
        print(f"Warning: Unknown datasets {invalid_ds}")
    invalid_pipe = [p for p in pipelines if p not in PIPELINE_CONFIGS]
    if invalid_pipe:
        print(f"Warning: Unknown pipelines {invalid_pipe}")
        
    datasets = [ds for ds in datasets if ds in DATASET_CONFIGS]
    pipelines = [p for p in pipelines if p in PIPELINE_CONFIGS]

    if not datasets or not pipelines:
        print("Error: No valid datasets and pipelines specified. Exiting.")
        sys.exit(1)

    # Generate all combinations
    tasks = list(itertools.product(datasets, pipelines))
    
    print("=" * 60)
    print(" Parallel Experiment Runner")
    print("=" * 60)
    print(f" Datasets:   {datasets}")
    print(f" Pipelines:  {pipelines}")
    print(f" Total tasks:{len(tasks)}")
    print(f" Max workers:{args.max_workers}")
    print("=" * 60)

    # Setup directories
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    results = []
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_experiment, ds, pipe, args, log_dir): (ds, pipe)
            for ds, pipe in tasks
        }
        
        # Process as they complete
        for future in as_completed(future_to_task):
            task_info = future_to_task[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                print(f"[EXCEPTION] Task {task_info} generated an exception: {exc}")
                results.append({"status": "exception", "dataset": task_info[0], "pipeline": task_info[1], "error": str(exc)})

    # Print summary
    print("\n" + "=" * 60)
    print(" RUN SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"Total: {len(results)} | Success: {success_count} | Failed: {len(results) - success_count}")
    
    for r in results:
        status = r.get('status', 'unknown').upper()
        ds = r.get('dataset', '?')
        pipe = r.get('pipeline', '?')
        dur = r.get('duration', 0)
        print(f"[{status:7s}] {ds} | {pipe} ({dur:.1f}s)")
        
    if success_count < len(results):
        sys.exit(1)

if __name__ == "__main__":
    main()
