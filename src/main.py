"""CLI entry point for ICD10 code extraction evaluation."""

import argparse
import os
import sys
from pathlib import Path
import yaml
from typing import Optional
from tqdm import tqdm
from data import ICD10HierarchyLoader, ICD10TranscriptDataset
from models import LLMModel
from models.llm_model import LLMConfig
from evaluator import MetricsCalculator, EvaluationResults
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    if "OPENAI_API_KEY" in os.environ:
        config.setdefault("llm", {})["api_key"] = os.environ["OPENAI_API_KEY"]

    return config


def process_one(i, config, use_uncertainity, hierarchy_loader, sample):
    transcript = sample["transcript"]
    gt_codes = sample["ground_truth_codes"]
    llm_config = LLMConfig(
        provider=config["llm"]["provider"],
        model_name=config["llm"]["model_name"],
        api_key=config["llm"].get("api_key"),
        base_url=config["llm"].get("base_url"),
        max_output_tokens=config["llm"]["max_output_tokens"],
        temperature=config["llm"]["temperature"],
        strategy=config["llm"]["strategy"],
    )
    model = LLMModel(hierarchy_loader, llm_config)
    if use_uncertainity:
        result = model.predict_with_uncertainty(
            transcript,
            n_samples=config["evaluation"]["n_samples"],
            temperature_range=tuple(config["evaluation"]["temperature_range"]),
        )
        pred_codes = result["codes"]
        confidence = result["confidence"]
        trace = result["trace_history"]
    else:
        pred_codes = model.predict(transcript)
        confidence = -1
        trace = model.trace_history

    return i, pred_codes, gt_codes, confidence, trace


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="ICD10 Code Extraction Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/gnahum12345/projects/icd10_prediction/src/config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["hierarchical", "flattened", "freeform"],
        default="freeform",
        help="Prediction strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="Model name (e.g., gpt-4, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--uncertainty", action="store_true", help="Run uncertainty quantification"
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=5,
        help="Number of Monte Carlo samples for uncertainty",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.strategy:
        config["llm"]["strategy"] = args.strategy
    if args.model:
        config["llm"]["model_name"] = args.model
    if args.n_mc_samples:
        config["evaluation"]["n_samples"] = args.n_mc_samples

    # Load hierarchy
    print("Loading ICD10 hierarchy...")
    xml_path = config["data"]["icd10cm_tabular_path"]
    hierarchy_loader = ICD10HierarchyLoader(xml_path)
    hierarchy_loader.load()
    print(f"Loaded {len(hierarchy_loader.get_all_codes())} ICD10 codes")
    # Load dataset
    print("Loading dataset...")
    csv_path = config["data"]["icd10_transcript_dataset_path"]
    dataset = ICD10TranscriptDataset(csv_path, hierarchy_loader)
    print(f"Loaded {len(dataset)} samples")
    # Initialize evaluator
    evaluator = MetricsCalculator(hierarchy_loader)
    results = EvaluationResults()

    # Evaluate
    n_samples = min(args.samples or 1, len(dataset))
    print(f"\nEvaluating {n_samples} samples...")
    predictions = [None] * n_samples
    ground_truth = [None] * n_samples
    confidences = [None] * n_samples
    traces = [None] * n_samples

    max_workers = min(30, n_samples)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one, i, config, args.uncertainty, hierarchy_loader, dataset[i]
            ): i
            for i in range(n_samples)
        }

        for future in tqdm(as_completed(futures), total=n_samples):
            i, pred, gt, conf, trace = future.result()

            predictions[i] = pred
            ground_truth[i] = gt
            confidences[i] = conf
            traces[i] = trace

            results.add_result(pred, gt, conf, trace)

    print("\nComputing metrics...")

    # Compute metrics
    metrics = evaluator.compute_metrics(
        predictions, ground_truth, confidences if args.uncertainty else None
    )
    _ = evaluator.collect_wrong_cases(
        predictions, ground_truth, [t for th in traces for t in th]
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro Recall: {metrics['micro_recall']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    if "hierarchical_accuracy" in metrics:
        print(f"Hierarchical Accuracy: {metrics['hierarchical_accuracy']:.4f}")
        print(f"Hierarchical F1: {metrics['hierarchical_f1']:.4f}")
    # if args.uncertainty and "calibration_correlation" in metrics:
    #     print(f"Calibration Correlation: {metrics['calibration_correlation']:.4f}")
    #     print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
    #     print(f"Confidence Mean: {metrics['confidence_mean']:.4f}")
    #     print(f"Confidence Std: {metrics['confidence_std']:.4f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
