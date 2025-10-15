#!/usr/bin/env python3
"""
Automated Collatz Analysis Pipeline with Hardware Adaptation and Persistent Caching

This script integrates with your existing Collatz research repository,
adds hardware-aware parallel processing, plus persistent filtering of trivial
and previously computed cases using a JSON record cache.

Usage:
  python automated_pipeline.py --create-config
  python automated_pipeline.py --config pipeline_config.json
  python automated_pipeline.py --param-a 3 --param-b 4 --param-c 1 --range-end 1000
"""

import multiprocessing
import psutil
import json
import time
import argparse
import math
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

# Import your existing modules
from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit
from src.statistical_tests import analyze_conformity


def create_config_file(filename: str = "pipeline_config.json"):
    default_config = {
        "runtime": {
            "param_a": 2,
            "param_b": 7,
            "param_c": 1,
            "initial_range_start": 1,
            "initial_range_end": 10000,
            "max_iterations": 2000,
            "batch_size": 1000,
            "output_dir": "results",
            "log_level": "INFO",
            "save_intermediate": True,
            "timeout_per_batch": 300,
        },
        "hardware": {
            "max_cores": None,
            "max_memory_gb": None,
            "memory_safety_factor": 0.8,
            "core_safety_factor": 0.8,
        }
    }
    with open(filename, "w") as f:
        json.dump(default_config, f, indent=2)
    return filename


def load_config(filename: str):
    with open(filename, "r") as f:
        config_data = json.load(f)
    runtime_conf = RuntimeConfig(**config_data["runtime"])
    hardware_conf = HardwareConfig(**config_data["hardware"])
    return runtime_conf, hardware_conf


@dataclass
class HardwareConfig:
    max_cores: Optional[int] = None
    max_memory_gb: Optional[float] = None
    memory_safety_factor: float = 0.8
    core_safety_factor: float = 0.8

    def __post_init__(self):
        if self.max_cores is None:
            available_cores = multiprocessing.cpu_count()
            self.max_cores = max(1, int(available_cores * self.core_safety_factor))
        if self.max_memory_gb is None:
            memory_bytes = psutil.virtual_memory().available
            self.max_memory_gb = (memory_bytes / (1024 ** 3)) * self.memory_safety_factor


@dataclass
class RuntimeConfig:
    param_a: int = 2
    param_b: int = 7
    param_c: int = 1
    initial_range_start: int = 1
    initial_range_end: int = 10000
    max_iterations: int = 2000
    batch_size: int = 1000
    output_dir: str = "results"
    log_level: str = "INFO"
    save_intermediate: bool = True
    timeout_per_batch: int = 300

    @property
    def as_legacy_config(self) -> Dict:
        return {
            "param_a": self.param_a,
            "param_b": self.param_b,
            "param_c": self.param_c,
            "initial_range": (self.initial_range_start, self.initial_range_end),
            "max_iterations": self.max_iterations,
        }


@dataclass
class PipelineResults:
    parameter_set: str
    total_samples: int
    digit_frequencies: Dict[int, int]
    p_value: float
    mad: float
    conforms_benford: bool
    execution_time: float
    batches_processed: int
    hardware_used: Dict
    dmix_score: Optional[float] = None


def get_hardware_info() -> Dict:
    return {
        "cpu_count": multiprocessing.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_percent": psutil.virtual_memory().percent,
    }


def estimate_memory_usage(batch_size: int, max_iterations: int) -> float:
    avg_sequence_length = min(max_iterations, 200)
    estimated_numbers_per_batch = batch_size * avg_sequence_length
    memory_gb = (estimated_numbers_per_batch * 100) / (1024 ** 3)
    return memory_gb


def create_batches(start: int, end: int, batch_size: int) -> List[Tuple[int, int]]:
    batches = []
    current = start
    while current <= end:
        batch_end = min(current + batch_size - 1, end)
        batches.append((current, batch_end))
        current = batch_end + 1
    return batches


def load_processed_cases(file_path: Path) -> set:
    if file_path.exists():
        with open(file_path, "r") as f:
            processed = json.load(f)
            return set(processed)
    return set()


def save_processed_cases(file_path: Path, processed_cases: set):
    with open(file_path, "w") as f:
        json.dump(list(processed_cases), f)


def process_batch(
    batch_info: Tuple[int, int, RuntimeConfig],
    processed_cases_file: Path,
    processed_cases: set,
) -> Dict:
    start, end, config = batch_info

    # Reload latest processed cases in case updated from other processes
    processed_cases |= load_processed_cases(processed_cases_file)

    all_digits = []
    n_processed = 0

    for n in range(start, end + 1):
        if n == 1 or n in processed_cases:
            continue  # Skip trivial or already processed

        try:
            sequence = generalized_collatz(
                n, config.param_a, config.param_b, config.param_c, config.max_iterations
            )
            for term in sequence:
                if term > 1:
                    digit = get_leading_digit(term)
                    all_digits.append(digit)
            n_processed += 1
            processed_cases.add(n)
        except Exception:
            continue

    # Save updated processed cases after batch completes
    save_processed_cases(processed_cases_file, processed_cases)

    final_counts = Counter(all_digits)
    observed_counts = {d: final_counts.get(d, 0) for d in range(1, 10)}

    return {
        "batch_range": (start, end),
        "processed_count": n_processed,
        "digit_counts": observed_counts,
        "total_digits": len(all_digits),
    }


def run_automated_pipeline(runtime_config: RuntimeConfig, hardware_config: HardwareConfig) -> PipelineResults:
    logging.basicConfig(
        level=getattr(logging, runtime_config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    start_time = time.time()
    output_dir = Path(runtime_config.output_dir)
    output_dir.mkdir(exist_ok=True)

    processed_cases_file = output_dir / "processed_cases.json"
    processed_cases = load_processed_cases(processed_cases_file)

    # DEBUG: log loaded processed cases count
    logger.debug(f"Loaded {len(processed_cases)} previously processed cases.")

    estimated_memory = estimate_memory_usage(runtime_config.batch_size, runtime_config.max_iterations)

    if estimated_memory > hardware_config.max_memory_gb:
        new_batch_size = int(runtime_config.batch_size * (hardware_config.max_memory_gb / estimated_memory))
        new_batch_size = max(10, new_batch_size)
        logger.warning(f"Reducing batch size from {runtime_config.batch_size} to {new_batch_size} due to memory constraints")
        runtime_config.batch_size = new_batch_size

    batches = create_batches(runtime_config.initial_range_start, runtime_config.initial_range_end, runtime_config.batch_size)

    logger.info(f"Created {len(batches)} batches for processing range {runtime_config.initial_range_start}-{runtime_config.initial_range_end}")
    logger.info(f"Using {hardware_config.max_cores} cores with {hardware_config.max_memory_gb:.2f}GB memory limit")

    all_digit_counts = {d: 0 for d in range(1, 10)}
    total_processed = 0
    completed_batches = 0

    with ProcessPoolExecutor(max_workers=hardware_config.max_cores) as executor:
        futures = {
            executor.submit(process_batch, batch, processed_cases_file, processed_cases): batch
            for batch in [(start, end, runtime_config) for start, end in batches]
        }
        for future in as_completed(futures):
    try:
        result = future.result(timeout=runtime_config.timeout_per_batch)

        for digit, count in result["digit_counts"].items():
            all_digit_counts[digit] += count
        total_processed += result["processed_count"]
        completed_batches += 1

        if runtime_config.save_intermediate:
            batch_file = output_dir / f"batch_{result['batch_range'][0]}_{result['batch_range'][1]}.json"
            with open(batch_file, "w") as f:
                json.dump(result, f, indent=2)

        logger.info(
            f"Completed batch {completed_batches}/{len(batches)}: Range {result['batch_range']}, "
            f"Numbers processed: {result['processed_count']}, Digits collected: {result['total_digits']}"
        )
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        continue


                if runtime_config.save_intermediate:
                    batch_file = output_dir / f"batch_{result['batch_range'][0]}_{result['batch_range'][1]}.json"
                    with open(batch_file, "w") as f:
                        json.dump(result, f, indent=2)

                logger.info(
                    f"Completed batch {completed_batches
                    