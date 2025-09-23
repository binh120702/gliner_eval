#!/usr/bin/env python3
"""
Example script showing how to use the updated eval.py for comparing span selection algorithms.

This script demonstrates different ways to run the evaluation:
1. Compare all three algorithms (greedy, NMS, MWIS)
2. Run a single algorithm
3. Customize parameters like threshold and IoU threshold
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

def main():
    # Base command arguments
    base_args = [
        "python", "eval.py",
        "--model", "logs/model_12000",  # Update this path as needed
        "--log_dir", "logs",
        "--data", "data/pilener_train.json",  # Update this path as needed
        "--batch_size", "8",  # Smaller batch size for demo
        "--max_samples", "100"  # Limit samples for demo
    ]
    
    print("GLiNER Span Selection Algorithm Comparison Examples")
    print("="*60)
    
    # Example 1: Compare all three algorithms
    print("\n1. Comparing all three algorithms (greedy, NMS, MWIS)")
    cmd1 = base_args + ["--compare_all"]
    run_command(cmd1, "Compare all algorithms")
    
    # Example 2: Run only greedy algorithm
    print("\n2. Running only greedy algorithm")
    cmd2 = base_args + ["--algorithms", "greedy"]
    run_command(cmd2, "Greedy algorithm only")
    
    # Example 3: Run only NMS algorithm with custom IoU threshold
    print("\n3. Running NMS algorithm with custom IoU threshold")
    cmd3 = base_args + ["--algorithms", "nms", "--iou_threshold", "0.3"]
    run_command(cmd3, "NMS with IoU threshold 0.3")
    
    # Example 4: Run only MWIS algorithm
    print("\n4. Running MWIS algorithm")
    cmd4 = base_args + ["--algorithms", "mwis"]
    run_command(cmd4, "MWIS algorithm only")
    
    # Example 5: Compare greedy and NMS with different threshold
    print("\n5. Comparing greedy and NMS with threshold 0.3")
    cmd5 = base_args + ["--compare_all", "--algorithms", "greedy", "nms", "--threshold", "0.3"]
    run_command(cmd5, "Greedy vs NMS with threshold 0.3")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the 'logs' directory for detailed results:")
    print("  - algorithm_comparison_results.txt (detailed results)")
    print("  - algorithm_comparison_summary.txt (summary table)")

if __name__ == "__main__":
    main()
