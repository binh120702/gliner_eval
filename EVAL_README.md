# GLiNER Span Selection Algorithm Comparison

This updated evaluation script allows you to compare three different span selection algorithms on the PILENER dataset:

1. **Greedy Search** - Original algorithm that selects spans greedily based on confidence scores
2. **Non-Maximum Suppression (NMS)** - Removes overlapping spans based on IoU threshold
3. **Maximum Weight Independent Set (MWIS)** - Finds optimal non-overlapping span selection using dynamic programming

## Features

- **Algorithm Comparison**: Compare all three algorithms side-by-side
- **Individual Evaluation**: Run a single algorithm for focused analysis
- **Configurable Parameters**: Adjust thresholds, batch sizes, and IoU thresholds
- **PILENER Dataset Support**: Optimized for PILENER dataset evaluation
- **Detailed Reporting**: Save results to files with summary tables
- **Robust Error Handling**: Skip problematic batches and continue evaluation
- **Memory Management**: Automatic CUDA cache clearing and memory optimization

## Usage

### Basic Usage

```bash
# Compare all three algorithms
python eval.py --compare_all --model logs/model_12000 --data data/pilener_train.json

# Run single algorithm
python eval.py --algorithms greedy --model logs/model_12000 --data data/pilener_train.json
```

### Command Line Arguments

- `--model`: Path to model folder (default: logs/model_12000)
- `--log_dir`: Path to log directory (default: logs)
- `--data`: Path to PILENER dataset (default: data/pilener_train.json)
- `--algorithms`: List of algorithms to compare (default: ['greedy', 'nms', 'mwis'])
- `--compare_all`: Run all algorithms and compare results
- `--threshold`: Prediction threshold (default: 0.5)
- `--batch_size`: Batch size for evaluation (default: 12)
- `--max_samples`: Maximum samples to evaluate (-1 for all, default: -1)
- `--iou_threshold`: IoU threshold for NMS algorithm (default: 0.5)
- `--max_skipped_batches`: Maximum batches to skip before giving up (default: 100)

### Examples

```bash
# Compare all algorithms with custom parameters
python eval.py --compare_all \
    --model logs/model_12000 \
    --data data/pilener_train.json \
    --threshold 0.3 \
    --iou_threshold 0.4 \
    --batch_size 8 \
    --max_samples 1000

# Run only NMS with custom IoU threshold
python eval.py --algorithms nms \
    --model logs/model_12000 \
    --data data/pilener_train.json \
    --iou_threshold 0.3

# Compare greedy and MWIS algorithms
python eval.py --compare_all \
    --algorithms greedy mwis \
    --model logs/model_12000 \
    --data data/pilener_train.json
```

## Output Files

When using `--compare_all`, the script generates two output files in the log directory:

1. **`algorithm_comparison_results.txt`**: Detailed results for each algorithm
2. **`algorithm_comparison_summary.txt`**: Summary table with F1 scores

## Algorithm Details

### Greedy Search
- **Method**: Selects spans greedily based on confidence scores
- **Overlap Handling**: Removes overlapping spans sequentially
- **Use Case**: Fast baseline method, good for most scenarios

### Non-Maximum Suppression (NMS)
- **Method**: Removes overlapping spans based on IoU threshold
- **Parameters**: `iou_threshold` controls overlap sensitivity
- **Use Case**: Good for reducing false positives from overlapping predictions

### Maximum Weight Independent Set (MWIS)
- **Method**: Finds optimal non-overlapping span selection using dynamic programming
- **Constraint**: Only works with flat NER (no nested entities)
- **Use Case**: Optimal solution for non-overlapping span selection

## Requirements

- PyTorch
- GLiNER model and dependencies
- PILENER dataset in JSON format
- CUDA support (recommended)

## Example Output

```
============================================================
FINAL COMPARISON SUMMARY
============================================================
Algorithm       F1 Score  
-------------------------
greedy          0.7234    
nms             0.7156    
mwis            0.7289    
-------------------------
BEST            0.7289    (mwis)
```

## Error Handling

The evaluation script includes robust error handling to deal with common issues:

- **CUDA Out of Memory**: Automatically skips problematic batches and clears CUDA cache
- **Batch Processing Errors**: Continues evaluation even if some batches fail
- **Memory Management**: Automatic cache clearing between algorithms
- **Configurable Limits**: Set maximum skipped batches before giving up

### Memory Issues

If you encounter CUDA out of memory errors:

1. **Reduce batch size**: Use `--batch_size 1` (already set as default)
2. **Limit samples**: Use `--max_samples 1000` for testing
3. **Adjust skip limit**: Use `--max_skipped_batches 50` to fail faster
4. **Clear memory**: The script automatically clears CUDA cache between batches

## Notes

- PILENER dataset should be processed using `data/process_pilener.py` first
- The script automatically handles flat NER mode for PILENER
- Results are saved with timestamps for easy comparison across runs
- Use `--max_samples` to limit evaluation for faster testing
- The script will continue evaluation even if some batches fail due to memory issues
