import argparse
import json
import os
import glob
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
    parser.add_argument('--ner_data_dir', type=str, default='data/NER/', help='Path to NER datasets directory')
    parser.add_argument('--eval_all_ner', action='store_true', help='Evaluate all NER datasets with algorithm comparison')
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


def load_ner_data(filepath: str, max_samples: int = -1) -> List[Dict]:
    """Load NER dataset from JSON file and convert to PILENER format."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if max_samples > 0:
        data = data[:max_samples]
    
    # Convert NER format to PILENER format
    converted_data = []
    for item in data:
        # Tokenize the sentence (simple whitespace tokenization)
        tokens = item["sentence"].split()
        
        # Convert entities to PILENER format
        ner_entities = []
        for entity in item["entities"]:
            # Find token positions for the entity
            entity_text = entity["name"]
            entity_type = entity["type"]
            char_start, char_end = entity["pos"]
            
            # Find which tokens this entity spans
            token_start = len(item["sentence"][:char_start].split())
            token_end = len(item["sentence"][:char_end].split()) - 1
            
            # Ensure token positions are valid
            if token_start < len(tokens) and token_end < len(tokens) and token_start <= token_end:
                ner_entities.append([token_start, token_end, entity_type])
        
        converted_data.append({
            "tokenized_text": tokens,
            "ner": ner_entities
        })
    
    print(f"Loaded {len(converted_data)} samples from NER dataset (converted from {len(data)} original samples)")
    return converted_data


def load_dataset(filepath: str, max_samples: int = -1) -> List[Dict]:
    """Load dataset from JSON file, auto-detecting format."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if not data:
        return []
    
    # Check if it's PILENER format (has 'tokenized_text' and 'ner')
    if isinstance(data[0], dict) and 'tokenized_text' in data[0] and 'ner' in data[0]:
        print("Detected PILENER format")
        return load_pilener_data(filepath, max_samples)
    
    # Check if it's NER format (has 'sentence' and 'entities')
    elif isinstance(data[0], dict) and 'sentence' in data[0] and 'entities' in data[0]:
        print("Detected NER format")
        return load_ner_data(filepath, max_samples)
    
    else:
        raise ValueError(f"Unknown data format in {filepath}. Expected PILENER or NER format.")


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


def evaluate_all_ner_datasets(model: GLiNER, ner_data_dir: str, algorithms: List[str],
                             threshold: float = 0.5, batch_size: int = 12, 
                             iou_threshold: float = 0.5, log_dir: str = "logs",
                             max_samples: int = -1, max_skipped_batches: int = 100) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """
    Evaluate all NER datasets with algorithm comparison.
    
    Args:
        model: GLiNER model instance
        ner_data_dir: Directory containing NER datasets
        algorithms: List of algorithm names to compare
        threshold: Prediction threshold
        batch_size: Batch size
        iou_threshold: IoU threshold for NMS
        log_dir: Directory to save results
        max_samples: Maximum samples per dataset (-1 for all)
        max_skipped_batches: Maximum batches to skip before giving up
    
    Returns:
        Dictionary mapping dataset names to algorithm results
    """
    # Get all dataset directories
    all_paths = glob.glob(f"{ner_data_dir}/*")
    all_paths = sorted([p for p in all_paths if os.path.isdir(p)])
    
    print(f"Found {len(all_paths)} NER datasets to evaluate")
    
    # Zero-shot benchmark datasets (as defined in original evaluation)
    zero_shot_benchmarks = ["mit-movie", "mit-restaurant", "CrossNER_AI", "CrossNER_literature", 
                           "CrossNER_music", "CrossNER_politics", "CrossNER_science"]
    
    all_results = {}  # Results for all datasets
    zero_shot_results = {}  # Results for zero-shot benchmarks
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(log_dir, "all_ner_datasets_algorithm_comparison.txt")
    with open(results_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE NER DATASET EVALUATION WITH ALGORITHM COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"IoU Threshold (NMS): {iou_threshold}\n")
        f.write(f"Algorithms: {', '.join(algorithms)}\n")
        f.write("="*80 + "\n\n")
    
    # Evaluate each dataset
    for dataset_path in tqdm(all_paths, desc="Evaluating datasets"):
        dataset_name = os.path.basename(dataset_path)
        
        # Skip sample datasets
        if "sample_" in dataset_name:
            continue
            
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Check if test.json exists
        test_file = os.path.join(dataset_path, "test.json")
        if not os.path.exists(test_file):
            print(f"Warning: test.json not found in {dataset_path}, skipping...")
            continue
        
        try:
            # Load dataset
            test_data = load_dataset(test_file, max_samples)
            
            if not test_data:
                print(f"Warning: No data loaded from {test_file}, skipping...")
                continue
            
            # Compare algorithms for this dataset
            dataset_results = compare_algorithms(
                model, test_data, algorithms, threshold, batch_size, 
                iou_threshold, log_dir, max_skipped_batches
            )
            
            # Store results
            all_results[dataset_name] = dataset_results
            
            # Check if it's a zero-shot benchmark
            if dataset_name in zero_shot_benchmarks:
                zero_shot_results[dataset_name] = dataset_results
            
            # Write results to file
            with open(results_file, "a") as f:
                f.write(f"DATASET: {dataset_name}\n")
                f.write("-" * 40 + "\n")
                
                for algorithm, (out, f1) in dataset_results.items():
                    f.write(f"{algorithm.upper()}:\n")
                    f.write(f"F1 Score: {f1:.4f}\n")
                    f.write(f"Details: {out}\n\n")
                
                f.write("="*80 + "\n\n")
            
            print(f"Completed evaluation for {dataset_name}")
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            # Store error result
            error_result = {alg: (f"Error: {str(e)}", 0.0) for alg in algorithms}
            all_results[dataset_name] = error_result
            
            if dataset_name in zero_shot_benchmarks:
                zero_shot_results[dataset_name] = error_result
    
    # Generate summary tables
    generate_summary_tables(all_results, zero_shot_results, algorithms, log_dir)
    
    return all_results


def generate_summary_tables(all_results: Dict[str, Dict[str, Tuple[str, float]]], 
                          zero_shot_results: Dict[str, Dict[str, Tuple[str, float]]],
                          algorithms: List[str], log_dir: str):
    """Generate summary tables for all datasets and zero-shot benchmarks."""
    
    # Summary file
    summary_file = os.path.join(log_dir, "all_ner_datasets_summary.txt")
    
    with open(summary_file, "w") as f:
        f.write("COMPREHENSIVE NER DATASET EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Table for all datasets
        f.write("ALL DATASETS RESULTS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Dataset':<25} " + " ".join([f"{alg.upper():<10}" for alg in algorithms]) + "\n")
        f.write("-" * 60 + "\n")
        
        all_algorithm_avgs = {alg: [] for alg in algorithms}
        
        for dataset_name, dataset_results in all_results.items():
            f.write(f"{dataset_name:<25} ")
            for algorithm in algorithms:
                if algorithm in dataset_results:
                    _, f1 = dataset_results[algorithm]
                    f.write(f"{f1:<10.4f} ")
                    all_algorithm_avgs[algorithm].append(f1)
                else:
                    f.write(f"{'N/A':<10} ")
            f.write("\n")
        
        # Calculate averages
        f.write("-" * 60 + "\n")
        f.write(f"{'AVERAGE':<25} ")
        for algorithm in algorithms:
            if all_algorithm_avgs[algorithm]:
                avg_f1 = sum(all_algorithm_avgs[algorithm]) / len(all_algorithm_avgs[algorithm])
                f.write(f"{avg_f1:<10.4f} ")
            else:
                f.write(f"{'N/A':<10} ")
        f.write("\n")
        
        # Zero-shot benchmark table
        if zero_shot_results:
            f.write("\n\nZERO-SHOT BENCHMARK RESULTS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Dataset':<25} " + " ".join([f"{alg.upper():<10}" for alg in algorithms]) + "\n")
            f.write("-" * 60 + "\n")
            
            zs_algorithm_avgs = {alg: [] for alg in algorithms}
            
            for dataset_name, dataset_results in zero_shot_results.items():
                f.write(f"{dataset_name:<25} ")
                for algorithm in algorithms:
                    if algorithm in dataset_results:
                        _, f1 = dataset_results[algorithm]
                        f.write(f"{f1:<10.4f} ")
                        zs_algorithm_avgs[algorithm].append(f1)
                    else:
                        f.write(f"{'N/A':<10} ")
                f.write("\n")
            
            # Calculate zero-shot averages
            f.write("-" * 60 + "\n")
            f.write(f"{'AVERAGE':<25} ")
            for algorithm in algorithms:
                if zs_algorithm_avgs[algorithm]:
                    avg_f1 = sum(zs_algorithm_avgs[algorithm]) / len(zs_algorithm_avgs[algorithm])
                    f.write(f"{avg_f1:<10.4f} ")
                else:
                    f.write(f"{'N/A':<10} ")
            f.write("\n")
        
        # Best algorithm analysis
        f.write("\n\nBEST ALGORITHM ANALYSIS:\n")
        f.write("-" * 60 + "\n")
        
        for algorithm in algorithms:
            if all_algorithm_avgs[algorithm]:
                avg_f1 = sum(all_algorithm_avgs[algorithm]) / len(all_algorithm_avgs[algorithm])
                f.write(f"{algorithm.upper()} Average F1: {avg_f1:.4f}\n")
        
        # Find best algorithm
        best_algorithm = None
        best_avg = 0.0
        for algorithm in algorithms:
            if all_algorithm_avgs[algorithm]:
                avg_f1 = sum(all_algorithm_avgs[algorithm]) / len(all_algorithm_avgs[algorithm])
                if avg_f1 > best_avg:
                    best_avg = avg_f1
                    best_algorithm = algorithm
        
        if best_algorithm:
            f.write(f"\nBEST OVERALL ALGORITHM: {best_algorithm.upper()} (F1: {best_avg:.4f})\n")
    
    print(f"\nSummary saved to: {summary_file}")


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}")
    model = GLiNER.from_pretrained(args.model, load_tokenizer=True).to("cuda:0")
    
    if args.eval_all_ner:
        # Evaluate all NER datasets with algorithm comparison
        print(f"Evaluating all NER datasets in {args.ner_data_dir}")
        all_results = evaluate_all_ner_datasets(
            model, args.ner_data_dir, args.algorithms,
            args.threshold, args.batch_size, args.iou_threshold, 
            args.log_dir, args.max_samples, args.max_skipped_batches
        )
        
        # Print final summary
        print(f"\n{'='*80}")
        print("FINAL COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        # Calculate overall averages
        all_algorithm_avgs = {alg: [] for alg in args.algorithms}
        for dataset_results in all_results.values():
            for algorithm in args.algorithms:
                if algorithm in dataset_results:
                    _, f1 = dataset_results[algorithm]
                    all_algorithm_avgs[algorithm].append(f1)
        
        print(f"{'Algorithm':<15} {'Average F1':<12} {'Datasets':<10}")
        print("-" * 40)
        
        for algorithm in args.algorithms:
            if all_algorithm_avgs[algorithm]:
                avg_f1 = sum(all_algorithm_avgs[algorithm]) / len(all_algorithm_avgs[algorithm])
                print(f"{algorithm:<15} {avg_f1:<12.4f} {len(all_algorithm_avgs[algorithm]):<10}")
        
        # Find best algorithm
        best_algorithm = None
        best_avg = 0.0
        for algorithm in args.algorithms:
            if all_algorithm_avgs[algorithm]:
                avg_f1 = sum(all_algorithm_avgs[algorithm]) / len(all_algorithm_avgs[algorithm])
                if avg_f1 > best_avg:
                    best_avg = avg_f1
                    best_algorithm = algorithm
        
        if best_algorithm:
            print("-" * 40)
            print(f"{'BEST':<15} {best_avg:<12.4f} ({best_algorithm})")
        
        print(f"\nDetailed results saved to: {args.log_dir}/all_ner_datasets_algorithm_comparison.txt")
        print(f"Summary saved to: {args.log_dir}/all_ner_datasets_summary.txt")
        
    else:
        # Single dataset evaluation (auto-detect format)
        print(f"Loading dataset from {args.data}")
        test_data = load_dataset(args.data, args.max_samples)
        
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