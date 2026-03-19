#!/usr/bin/env python3
"""
Automated Collatz Analysis Pipeline with Hardware Adaptation and Persistent Caching
Fixed Version - Corrected Indentation and Logic Flow
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

# Import your existing modules (Ensure these exist in your /src folder)
try:
    from src.collatz_generator import generalized_collatz
    from src.benford_analyzer import get_leading_digit
    from src.statistical_tests import analyze_conformity
except ImportError:
    print("Error: Ensure 'src/collatz_generator.py' and others are in the same directory.")
    sys.exit(1)

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
    param_a: int = 21
    param_b: int = 14
    param_c: int = 12
    initial_range_start: int = 1
    initial_range_end: int = 10000
    max_iterations: int = 2000
    batch_size: int = 1000
    output_dir: str = "results"
    log_level: str = "INFO"
    save_intermediate: bool = True
    timeout_per_batch: int = 300

def load_processed_cases(file_path: Path) -> set:
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_processed_cases(file_path: Path, processed_cases: set):
    with open(file_path, "w") as f:
        json.dump(list(processed_cases), f)

def process_batch(batch_info: Tuple[int, int, RuntimeConfig]) -> Dict:
    start, end, config = batch_info
    all_digits = []
    n_processed = 0
    local_processed = []

    for n in range(start, end + 1):
        if n <= 1: continue
        try:
            sequence = generalized_collatz(
                n, config.param_a, config.param_b, config.param_c, config.max_iterations
            )
            for term in sequence:
                if term > 1:
                    all_digits.append(get_leading_digit(term))
            n_processed += 1
            local_processed.append(n)
        except Exception:
            continue

    counts = Counter(all_digits)
    return {
        "batch_range": (start, end),
        "processed_count": n_processed,
        "digit_counts": {d: counts.get(d, 0) for d in range(1, 10)},
        "total_digits": len(all_digits),
        "processed_numbers": local_processed
    }

def run_automated_pipeline(runtime_config: RuntimeConfig, hardware_config: HardwareConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    output_dir = Path(runtime_config.output_dir)
    output_dir.mkdir(exist_ok=True)
    processed_file = output_dir / "processed_cases.json"
    
    # Simple batching logic
    batches = []
    for i in range(runtime_config.initial_range_start, runtime_config.initial_range_end + 1, runtime_config.batch_size):
        batches.append((i, min(i + runtime_config.batch_size - 1, runtime_config.initial_range_end), runtime_config))

    all_digit_counts = {d: 0 for d in range(1, 10)}
    total_processed_n = 0
    global_processed_set = load_processed_cases(processed_file)

    logger.info(f"Starting pipeline for ({runtime_config.param_a}, {runtime_config.param_b}, {runtime_config.param_c})")

    with ProcessPoolExecutor(max_workers=hardware_config.max_cores) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]
        for future in as_completed(futures):
            try:
                result = future.result(timeout=runtime_config.timeout_per_batch)
                for d, count in result["digit_counts"].items():
                    all_digit_counts[d] += count
                
                total_processed_n += result["processed_count"]
                global_processed_set.update(result["processed_numbers"])
                
                # Save progress periodically
                save_processed_cases(processed_file, global_processed_set)
                logger.info(f"Batch {result['batch_range']} finished. Total digits: {sum(all_digit_counts.values())}")
            except Exception as e:
                logger.error(f"Batch failed: {e}")

    # Final Statistics (Calling your statistical_tests module)
    final_stats = analyze_conformity(all_digit_counts)
    logger.info(f"FINAL MAD: {final_stats.get('mad', 'N/A')}")
    return final_stats

if __name__ == "__main__":
    # Standard setup for your (21, 14, 12) research
    runtime = RuntimeConfig(initial_range_end=1000) # Small test run
    hardware = HardwareConfig()
    run_automated_pipeline(runtime, hardware)
