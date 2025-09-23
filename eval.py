import argparse
import json
import os
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

from gliner import GLiNER
from gliner.evaluation import Evaluator


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER with Multiple Decoding Algorithms")
    parser.add_argument("--model", type=str, default="logs/model_12000", help="Path to model folder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to log directory")
    parser.add_argument('--data', type=str, default='data/pilener_train.json', help='Path to PILENER dataset')
    parser.add_argument('--algorithms', nargs='+', default=['greedy', 'nms', 'mwis'], 
                       help='List of algorithms to compare: greedy, nms, mwis')
    parser.add_argument('--compare_all', action='store_true', 
                       help='Run all algorithms and compare results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=-1, help='Maximum samples to evaluate (-1 for all)')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for NMS algorithm')
    parser.add_argument('--max_skipped_batches', type=int, default=100, help='Maximum batches to skip before giving up')
    return parser


def load_pilener_data(filepath: str, max_samples: int = -1) -> List[Dict]:
    """Load PILENER dataset from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if max_samples > 0:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples from PILENER dataset")
    return data


def evaluate_with_algorithm(model: GLiNER, test_data: List[Dict], algorithm: str, 
                          threshold: float = 0.5, batch_size: int = 12, 
                          iou_threshold: float = 0.5, max_skipped_batches: int = 100) -> Tuple[str, float]:
    """
    Evaluate model with a specific span selection algorithm.
    
    Args:
        model: GLiNER model instance
        test_data: Test dataset
        algorithm: Algorithm name ('greedy', 'nms', 'mwis')
        threshold: Prediction threshold
        batch_size: Batch size
        iou_threshold: IoU threshold for NMS
        max_skipped_batches: Maximum batches to skip before giving up
    
    Returns:
        Tuple of (results_string, f1_score)
    """
    model.eval()
    
    # Clear CUDA cache before starting evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create the dataset and data loader
    from gliner.data_processing.collator import DataCollator
    collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        return_tokens=True,
        return_entities=True,
        return_id_to_classes=True,
        prepare_labels=False,
    )
    data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    all_preds = []
    all_trues = []

    # Iterate over data batches
    batch_count = 0
    skipped_batches = 0
    
    for batch in tqdm(data_loader, desc=f"Evaluating with {algorithm}"):
        batch_count += 1
        
        try:
            # Move the batch to the appropriate device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(model.device)

            # Perform predictions
            model_output = model.model(**batch)[0]

            if not isinstance(model_output, torch.Tensor):
                model_output = torch.from_numpy(model_output)

            # Decode with specific algorithm
            decoded_outputs = model.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_output,
                flat_ner=True,  # PILENER is flat NER
                threshold=threshold,
                multi_label=False,
                decoding_algo=algorithm,
                iou_threshold=iou_threshold
            )
            all_preds.extend(decoded_outputs)
            all_trues.extend(batch["entities"])
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                skipped_batches += 1
                print(f"\nWarning: Skipping batch {batch_count} due to CUDA OOM: {str(e)}")
                print(f"Skipped batches so far: {skipped_batches}")
                
                # Check if we've skipped too many batches
                if skipped_batches >= max_skipped_batches:
                    print(f"\nError: Skipped {skipped_batches} batches (max allowed: {max_skipped_batches})")
                    print("Stopping evaluation due to too many failed batches.")
                    break
                
                # Clear CUDA cache to free up memory
                torch.cuda.empty_cache()
                
                # Continue to next batch
                continue
            else:
                # Re-raise non-OOM errors
                raise e
        except Exception as e:
            print(f"\nWarning: Skipping batch {batch_count} due to error: {str(e)}")
            skipped_batches += 1
            
            # Check if we've skipped too many batches
            if skipped_batches >= max_skipped_batches:
                print(f"\nError: Skipped {skipped_batches} batches (max allowed: {max_skipped_batches})")
                print("Stopping evaluation due to too many failed batches.")
                break
            continue
    
    if skipped_batches > 0:
        print(f"\nTotal skipped batches: {skipped_batches}/{batch_count}")
        print(f"Successfully processed: {batch_count - skipped_batches}/{batch_count} batches")
    
    # Check if we have enough data to evaluate
    if len(all_preds) == 0:
        print(f"\nWarning: No predictions generated for {algorithm} algorithm!")
        return f"No predictions generated - all batches failed", 0.0
    
    if len(all_trues) == 0:
        print(f"\nWarning: No ground truth data available for {algorithm} algorithm!")
        return f"No ground truth data - all batches failed", 0.0
    
    print(f"\nEvaluating {len(all_preds)} predictions against {len(all_trues)} ground truth samples")
    
    # Evaluate the predictions
    evaluator = Evaluator(all_trues, all_preds)
    out, f1 = evaluator.evaluate()

    return out, f1


def compare_algorithms(model: GLiNER, test_data: List[Dict], algorithms: List[str],
                      threshold: float = 0.5, batch_size: int = 12, 
                      iou_threshold: float = 0.5, log_dir: str = "logs",
                      max_skipped_batches: int = 100) -> Dict[str, Tuple[str, float]]:
    """
    Compare multiple span selection algorithms.
    
    Args:
        model: GLiNER model instance
        test_data: Test dataset
        algorithms: List of algorithm names to compare
        threshold: Prediction threshold
        batch_size: Batch size
        iou_threshold: IoU threshold for NMS
        log_dir: Directory to save results
        max_skipped_batches: Maximum batches to skip before giving up
    
    Returns:
        Dictionary mapping algorithm names to (results_string, f1_score)
    """
    results = {}
    
    print(f"Comparing {len(algorithms)} algorithms: {', '.join(algorithms)}")
    
    for algorithm in algorithms:
        print(f"\n{'='*50}")
        print(f"Evaluating with {algorithm.upper()} algorithm")
        print(f"{'='*50}")
        
        try:
            out, f1 = evaluate_with_algorithm(
                model, test_data, algorithm, threshold, batch_size, iou_threshold, max_skipped_batches
            )
            results[algorithm] = (out, f1)
            print(f"\n{algorithm.upper()} Results:")
            print(out)
            print(f"F1 Score: {f1:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {algorithm}: {str(e)}")
            results[algorithm] = (f"Error: {str(e)}", 0.0)
    
    # Save comparison results
    save_comparison_results(results, log_dir, threshold, iou_threshold)
    
    return results


def save_comparison_results(results: Dict[str, Tuple[str, float]], log_dir: str, 
                           threshold: float, iou_threshold: float):
    """Save comparison results to files."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(log_dir, "algorithm_comparison_results.txt")
    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("SPAN SELECTION ALGORITHM COMPARISON RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"IoU Threshold (NMS): {iou_threshold}\n")
        f.write("="*60 + "\n\n")
        
        for algorithm, (out, f1) in results.items():
            f.write(f"{algorithm.upper()} ALGORITHM:\n")
            f.write("-" * 30 + "\n")
            f.write(out + "\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
    
    # Save summary table
    summary_file = os.path.join(log_dir, "algorithm_comparison_summary.txt")
    with open(summary_file, "w") as f:
        f.write("ALGORITHM COMPARISON SUMMARY\n")
        f.write("="*40 + "\n")
        f.write(f"{'Algorithm':<15} {'F1 Score':<10}\n")
        f.write("-" * 25 + "\n")
        
        for algorithm, (_, f1) in results.items():
            f.write(f"{algorithm:<15} {f1:<10.4f}\n")
        
        f.write("-" * 25 + "\n")
        best_algorithm = max(results.keys(), key=lambda k: results[k][1])
        best_f1 = results[best_algorithm][1]
        f.write(f"{'BEST':<15} {best_f1:<10.4f} ({best_algorithm})\n")
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary: {summary_file}")


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}")
    model = GLiNER.from_pretrained(args.model, load_tokenizer=True).to("cuda:0")
    
    # Load PILENER dataset
    print(f"Loading PILENER dataset from {args.data}")
    test_data = load_pilener_data(args.data, args.max_samples)
    
    if args.compare_all:
        # Compare all algorithms
        results = compare_algorithms(
            model, test_data, args.algorithms, 
            args.threshold, args.batch_size, args.iou_threshold, args.log_dir, args.max_skipped_batches
        )
        
        # Print final comparison
        print(f"\n{'='*60}")
        print("FINAL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Algorithm':<15} {'F1 Score':<10}")
        print("-" * 25)
        
        for algorithm, (_, f1) in results.items():
            print(f"{algorithm:<15} {f1:<10.4f}")
        
        print("-" * 25)
        best_algorithm = max(results.keys(), key=lambda k: results[k][1])
        best_f1 = results[best_algorithm][1]
        print(f"{'BEST':<15} {best_f1:<10.4f} ({best_algorithm})")
        
    else:
        # Run single algorithm (first one in the list)
        algorithm = args.algorithms[0]
        print(f"Evaluating with {algorithm} algorithm")
        
        out, f1 = evaluate_with_algorithm(
            model, test_data, algorithm, 
            args.threshold, args.batch_size, args.iou_threshold, args.max_skipped_batches
        )
        
        print(f"\n{algorithm.upper()} Results:")
        print(out)
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()