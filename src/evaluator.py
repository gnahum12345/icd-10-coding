"""Evaluation metrics for ICD10 code extraction."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import confusion_matrix
from data import ICD10HierarchyLoader
import json


@dataclass
class EvaluationResults:
    """Store evaluation results."""

    predictions: List[List[str]] = field(default_factory=list)
    ground_truth: List[List[str]] = field(default_factory=list)

    # NEW
    avg_confidences: List[float] = field(default_factory=list)
    std_confidences: List[float] = field(default_factory=list)

    traces: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_result(
        self,
        prediction: List[str],
        ground_truth: List[str],
        avg_confidence: Optional[float] = None,
        std_confidence: Optional[float] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single prediction result.

        Args:
            prediction: Predicted codes
            ground_truth: Ground truth codes
            confidence: Confidence score
            trace: Trace information
        """
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        if avg_confidence is not None:
            self.avg_confidences.append(avg_confidence)
        if std_confidence is not None:
            self.std_confidences.append(std_confidence)
        if trace is not None:
            self.traces.append(trace)


class MetricsCalculator:
    """Calculate evaluation metrics for ICD10 code extraction."""

    def __init__(self, hierarchy_loader: Optional[ICD10HierarchyLoader] = None):
        """Initialize metrics calculator.

        Args:
            hierarchy_loader: Optional hierarchy loader for hierarchical metrics
        """
        self.hierarchy_loader = hierarchy_loader

    def _get_hierarchy_level(self, code: str) -> Optional[int]:
        """Get hierarchy level/depth of a code.

        Args:
            code: ICD10 code

        Returns:
            Hierarchy level (0 for ROOT, 1 for first level, etc.) or None if code not found
        """
        if self.hierarchy_loader is None:
            return None

        path = self.hierarchy_loader.get_path_to_root(code)
        if not path:
            return None

        # Level is depth from ROOT (0-indexed, so ROOT is level 0)
        # Return None if code is ROOT itself
        if len(path) <= 1:
            return None

        # Return level (1-indexed: first level after ROOT is level 1)
        return len(path) - 1

    def compute_metrics(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        avg_confidences: Optional[List[float]] = None,
        std_confidences: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists
            avg_confidences: Optional average within the monte carlo simulation of confidence scores
            std_confidences: Optional standard deviation of the monte carlo simulation of confidence scores

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Exact match accuracy
        metrics["exact_match_accuracy"] = self._exact_match_accuracy(
            predictions, ground_truth
        )

        # Global precision/recall (micro-averaged, code-level)
        global_metrics = self._global_metrics(predictions, ground_truth)
        metrics.update(global_metrics)

        # Micro metrics (code-level) - same as global but with different naming
        micro_metrics = self._micro_metrics(predictions, ground_truth)
        metrics.update(micro_metrics)

        # Macro metrics (sample-level)
        macro_metrics = self._macro_metrics(predictions, ground_truth)
        metrics.update(macro_metrics)

        # Hierarchical accuracy
        if self.hierarchy_loader is not None:
            metrics["hierarchical_accuracy"] = self._hierarchical_accuracy(
                predictions, ground_truth
            )
            metrics["hierarchical_f1"] = self._hierarchical_f1(
                predictions, ground_truth
            )

            # Per-level precision/recall
            per_level_metrics = self._per_level_metrics(predictions, ground_truth)
            metrics.update(per_level_metrics)

        # Uncertainty metrics
        if avg_confidences:
            uncertainty_metrics = self._uncertainty_metrics(
                predictions, ground_truth, avg_confidences, std_confidences
            )
            metrics.update(uncertainty_metrics)

        return metrics

    def _exact_match_accuracy(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """Compute exact match accuracy.

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Exact match accuracy
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        matches = sum(
            1 for pred, gt in zip(predictions, ground_truth) if set(pred) == set(gt)
        )

        return matches / len(predictions) if len(predictions) > 0 else 0.0

    def _global_metrics(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """Compute global precision/recall metrics (code-level aggregation).

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Dictionary with global_precision and global_recall
        """
        # Flatten all predictions and ground truth
        all_pred_codes = set()
        all_gt_codes = set()
        for pred, gt in zip(predictions, ground_truth):
            all_pred_codes.update(pred)
            all_gt_codes.update(gt)

        # True positives: codes that appear in both
        true_positives = len(all_pred_codes & all_gt_codes)

        # Precision: TP / (TP + FP)
        precision = (
            true_positives / len(all_pred_codes) if len(all_pred_codes) > 0 else 0.0
        )

        # Recall: TP / (TP + FN)
        recall = true_positives / len(all_gt_codes) if len(all_gt_codes) > 0 else 0.0

        # F1 score
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "global_precision": precision,
            "global_recall": recall,
            "global_f1": f1,
        }

    def _micro_metrics(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """Compute micro-averaged metrics (code-level).

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Dictionary of micro metrics
        """
        # Flatten all predictions and ground truth
        all_pred_codes = set()
        all_gt_codes = set()
        for pred, gt in zip(predictions, ground_truth):
            all_pred_codes.update(pred)
            all_gt_codes.update(gt)

        # True positives: codes that appear in both
        true_positives = len(all_pred_codes & all_gt_codes)

        # Precision: TP / (TP + FP)
        precision = (
            true_positives / len(all_pred_codes) if len(all_pred_codes) > 0 else 0.0
        )

        # Recall: TP / (TP + FN)
        recall = true_positives / len(all_gt_codes) if len(all_gt_codes) > 0 else 0.0

        # F1 score
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
        }

    def _macro_metrics(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """Compute macro-averaged metrics (sample-level).

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Dictionary of macro metrics
        """
        precisions = []
        recalls = []
        f1_scores = []

        for pred, gt in zip(predictions, ground_truth):
            pred_set = set(pred)
            gt_set = set(gt)

            # True positives
            tp = len(pred_set & gt_set)

            # Precision
            precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            precisions.append(precision)

            # Recall
            recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
            recalls.append(recall)

            # F1
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores.append(f1)

        return {
            "macro_precision": np.mean(precisions) if precisions else 0.0,
            "macro_recall": np.mean(recalls) if recalls else 0.0,
            "macro_f1": np.mean(f1_scores) if f1_scores else 0.0,
        }

    def _hierarchical_accuracy(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """Compute hierarchical accuracy (partial credit for parent matches).

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Hierarchical accuracy
        """
        if self.hierarchy_loader is None:
            return 0.0

        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_set = set(pred)
            gt_set = set(gt)

            # Exact match
            if pred_set == gt_set:
                correct += 1
                continue

            # Check if any predicted code matches or is ancestor/descendant of ground truth
            matched = False
            for pred_code in pred_set:
                # Direct match
                if pred_code in gt_set:
                    matched = True
                    break

                # Check if pred_code is ancestor of any gt_code
                pred_node = self.hierarchy_loader.get_node(pred_code)
                if pred_node is not None:
                    # Get all descendants of pred_code
                    descendants = self._get_descendants(pred_node)
                    if descendants & gt_set:
                        matched = True
                        break

                # Check if pred_code is descendant of any gt_code
                pred_path = self.hierarchy_loader.get_path_to_root(pred_code)
                pred_ancestors = {node.code for node in pred_path}
                if pred_ancestors & gt_set:
                    matched = True
                    break

            if matched:
                correct += 1

        return correct / len(predictions) if len(predictions) > 0 else 0.0

    def _hierarchical_f1(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """Compute hierarchical F1 score.

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Hierarchical F1 score
        """
        if self.hierarchy_loader is None:
            return 0.0

        # Compute hierarchical precision and recall
        tp = 0
        fp = 0
        fn = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_set = set(pred)
            gt_set = set(gt)

            # Count true positives (matches or hierarchical matches)
            for pred_code in pred_set:
                matched = False
                # Direct match
                if pred_code in gt_set:
                    matched = True
                else:
                    # Check hierarchical relationships
                    pred_node = self.hierarchy_loader.get_node(pred_code)
                    if pred_node is not None:
                        descendants = self._get_descendants(pred_node)
                        if descendants & gt_set:
                            matched = True
                        else:
                            pred_path = self.hierarchy_loader.get_path_to_root(
                                pred_code
                            )
                            pred_ancestors = {node.code for node in pred_path}
                            if pred_ancestors & gt_set:
                                matched = True

                if matched:
                    tp += 1
                else:
                    fp += 1

            # Count false negatives (ground truth codes not matched)
            for gt_code in gt_set:
                matched = False
                # Direct match
                if gt_code in pred_set:
                    matched = True
                else:
                    # Check hierarchical relationships
                    gt_node = self.hierarchy_loader.get_node(gt_code)
                    if gt_node is not None:
                        descendants = self._get_descendants(gt_node)
                        if descendants & pred_set:
                            matched = True
                        else:
                            gt_path = self.hierarchy_loader.get_path_to_root(gt_code)
                            gt_ancestors = {node.code for node in gt_path}
                            if gt_ancestors & pred_set:
                                matched = True

                if not matched:
                    fn += 1

        # Compute F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return f1

    def _get_descendants(self, node: Any) -> set:
        """Get all descendant codes of a node.

        Args:
            node: ICD10Node

        Returns:
            Set of descendant codes
        """
        descendants = set()
        for child in node.children:
            descendants.add(child.code)
            descendants.update(self._get_descendants(child))
        return descendants

    def _per_level_metrics(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """Compute precision/recall per hierarchy level.

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists

        Returns:
            Dictionary with precision and recall for each level
        """
        if self.hierarchy_loader is None:
            return {}

        # Group codes by hierarchy level
        pred_by_level: Dict[int, set] = {}
        gt_by_level: Dict[int, set] = {}

        # Process predictions
        for pred_list in predictions:
            for code in pred_list:
                level = self._get_hierarchy_level(code)
                if level is not None:
                    if level not in pred_by_level:
                        pred_by_level[level] = set()
                    pred_by_level[level].add(code)

        # Process ground truth
        for gt_list in ground_truth:
            for code in gt_list:
                level = self._get_hierarchy_level(code)
                if level is not None:
                    if level not in gt_by_level:
                        gt_by_level[level] = set()
                    gt_by_level[level].add(code)

        # Compute metrics for each level
        metrics = {}
        all_levels = set(pred_by_level.keys()) | set(gt_by_level.keys())

        for level in sorted(all_levels):
            pred_codes = pred_by_level.get(level, set())
            gt_codes = gt_by_level.get(level, set())

            # True positives: codes that appear in both
            true_positives = len(pred_codes & gt_codes)

            # Precision: TP / (TP + FP)
            precision = true_positives / len(pred_codes) if len(pred_codes) > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = true_positives / len(gt_codes) if len(gt_codes) > 0 else 0.0

            # F1 score
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[f"level_{level}_precision"] = precision
            metrics[f"level_{level}_recall"] = recall
            metrics[f"level_{level}_f1"] = f1
            metrics[f"level_{level}_predicted_count"] = len(pred_codes)
            metrics[f"level_{level}_ground_truth_count"] = len(gt_codes)
            metrics[f"level_{level}_true_positives"] = true_positives

        return metrics

    def _uncertainty_metrics(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        avg_confidences: List[float],
        std_confidences: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        # Calibration uses *avg_confidences* only
        accuracies = [
            1.0 if set(pred) == set(gt) else 0.0
            for pred, gt in zip(predictions, ground_truth)
        ]

        calibration_correlation = (
            np.corrcoef(avg_confidences, accuracies)[0, 1]
            if len(avg_confidences) > 1
            else 0.0
        )

        # ECE with avg_confidences
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        # --- Flatten confidences & accuracies ---
        flat_conf = []
        flat_acc = []

        for pred, gt, conf_dict in zip(predictions, ground_truth, avg_confidences):
            for code in pred:
                if code not in conf_dict:
                    continue

                # Rescale 1–10 → 0–1
                conf = conf_dict[code] / 10.0
                acc = 1.0 if code in gt else 0.0

                flat_conf.append(conf)
                flat_acc.append(acc)

        # --- ECE computation ---
        ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = [
                (conf, acc) for conf, acc in zip(flat_conf, flat_acc) if lo <= conf < hi
            ]
            if len(in_bin) > 0:
                bin_conf = np.mean([x[0] for x in in_bin])
                bin_acc = np.mean([x[1] for x in in_bin])
                weight = len(in_bin) / len(flat_conf)
                ece += weight * abs(bin_conf - bin_acc)

        # --- Confidence statistics ---
        all_avg_conf_vals = [
            c for conf_dict in avg_confidences for c in conf_dict.values()
        ]
        all_std_conf_vals = [
            s for std_dict in std_confidences for s in std_dict.values()
        ]
        avg_mean = float(np.mean(all_avg_conf_vals)) if all_avg_conf_vals else 0.0
        avg_std = float(np.std(all_avg_conf_vals)) if all_avg_conf_vals else 0.0

        std_mean = float(np.mean(all_std_conf_vals)) if all_std_conf_vals else 0.0
        std_std = float(np.std(all_std_conf_vals)) if all_std_conf_vals else 0.0

        return {
            "calibration_correlation": calibration_correlation,
            "average_confidence_mean": avg_mean,
            "average_confidence_std": avg_std,
            "std_confidence_mean": std_mean,
            "std_confidence_std": std_std,
            "expected_calibration_error": ece,
        }

    def get_confusion_matrix(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Get confusion matrix.

        Args:
            predictions: List of predicted code lists
            ground_truth: List of ground truth code lists
            labels: Optional list of labels (codes) to include

        Returns:
            Confusion matrix
        """
        # Get all unique codes
        all_codes = set()
        for pred, gt in zip(predictions, ground_truth):
            all_codes.update(pred)
            all_codes.update(gt)

        if labels is None:
            labels = sorted(all_codes)

        # Create binary vectors for each sample
        y_pred = []
        y_true = []
        for pred, gt in zip(predictions, ground_truth):
            pred_vec = [1 if code in pred else 0 for code in labels]
            gt_vec = [1 if code in gt else 0 for code in labels]
            y_pred.append(pred_vec)
            y_true.append(gt_vec)

        # Flatten for confusion matrix (multi-label)
        y_pred_flat = np.array(y_pred).flatten()
        y_true_flat = np.array(y_true).flatten()

        # Compute confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        return cm

    def collect_wrong_cases(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        metadata: Optional[List[Dict[str, str]]] = None,
        output_path: str = "wrong_cases.json",
    ) -> List[dict]:
        """
        Collect all wrong cases and write them into a JSON file.

        Args:
            predictions: list of predicted code lists
            ground_truth: list of ground truth code lists
            metadata: optional list of dict with transcripts , traces, etc for each sample
            output_path: where to save the JSON file

        Returns:
            List of wrong-case dictionaries
        """
        wrong_cases = []

        for idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            if set(pred) != set(gt):
                breakpoint()
                wrong_cases.append(
                    {
                        "index": idx,
                        "predicted": pred,
                        "ground_truth": gt,
                        **(metadata[idx] if metadata else {}),
                    }
                )

        # Save JSON file
        with open(output_path, "w") as f:
            json.dump(wrong_cases, f, indent=2)

        return wrong_cases
