"""Data loading and processing module for ICD10 code extraction."""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


@dataclass
class ICD10Node:
    """Represents a node in the ICD10 hierarchy."""

    code: str
    description: str
    laymen_definition: Optional[str] = None
    parent: Optional["ICD10Node"] = None
    children: List["ICD10Node"] = field(default_factory=list)
    inclusion_terms: List[str] = field(default_factory=list)
    is_leaf: bool = False

    def __repr__(self) -> str:
        return f"ICD10Node(code={self.code}, description={self.description[:50]}...)"


class ICD10HierarchyLoader:
    """Loads and parses ICD10 hierarchy from XML file."""

    def __init__(self, xml_path: str):
        """Initialize loader with path to XML file.

        Args:
            xml_path: Path to icd10cm_tabular_2024.xml
        """
        self.xml_path = Path(xml_path)
        self.root_node: Optional[ICD10Node] = None
        self.code_map: Dict[str, ICD10Node] = {}

    def load(self) -> ICD10Node:
        """Load and parse XML file to build hierarchy.

        Returns:
            Root node of the hierarchy
        """
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Create root node
        self.root_node = ICD10Node(
            code="ROOT",
            description="ICD10 Root",
            laymen_definition="Root of ICD10 hierarchy",
        )
        self.code_map["ROOT"] = self.root_node

        # Parse chapters
        for chapter in root.findall(".//chapter"):
            self._parse_chapter(chapter, self.root_node)

        # Mark leaf nodes
        self._mark_leaf_nodes(self.root_node)

        return self.root_node

    def _parse_chapter(self, chapter_elem: ET.Element, parent: ICD10Node) -> None:
        """Parse a chapter element."""
        # Parse sections within chapter - only direct children
        for section in chapter_elem.findall("section"):
            self._parse_section(section, parent)

    def _parse_section(self, section_elem: ET.Element, parent: ICD10Node) -> None:
        """Parse a section element."""
        # Parse diag elements (diagnoses) - only direct children
        for diag in section_elem.findall("diag"):
            self._parse_diag(diag, parent)

    def _parse_diag(
        self, diag_elem: ET.Element, parent: ICD10Node
    ) -> Optional[ICD10Node]:
        """Parse a diag element recursively.

        Args:
            diag_elem: XML element for diagnosis
            parent: Parent node in hierarchy

        Returns:
            Created node or None
        """
        # Extract code
        name_elem = diag_elem.find("name")
        if name_elem is None or name_elem.text is None:
            return None

        code = name_elem.text.strip()

        # Extract description
        desc_elem = diag_elem.find("desc")
        description = (
            desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
        )

        # Extract inclusion terms
        inclusion_terms = []
        inclusion_term_elem = diag_elem.find("inclusionTerm")
        if inclusion_term_elem is not None:
            for note in inclusion_term_elem.findall("note"):
                if note.text:
                    inclusion_terms.append(note.text.strip())

        # Create laymen definition from description and inclusion terms
        laymen_parts = [description]
        if inclusion_terms:
            laymen_parts.extend(inclusion_terms)
        laymen_definition = ". ".join(laymen_parts)

        # Create node
        node = ICD10Node(
            code=code,
            description=description,
            laymen_definition=laymen_definition,
            parent=parent,
            inclusion_terms=inclusion_terms,
        )

        # Add to parent's children
        parent.children.append(node)

        # Store in code map
        self.code_map[code] = node

        # Parse nested diag elements (children)
        for child_diag in diag_elem.findall("diag"):
            self._parse_diag(child_diag, node)

        return node

    def _mark_leaf_nodes(self, node: ICD10Node) -> None:
        """Mark leaf nodes (nodes with no children)."""
        if not node.children:
            node.is_leaf = True
        else:
            for child in node.children:
                self._mark_leaf_nodes(child)

    def get_node(self, code: str) -> Optional[ICD10Node]:
        """Get node by code.

        Args:
            code: ICD10 code

        Returns:
            Node if found, None otherwise
        """
        return self.code_map.get(code)

    def get_path_to_root(self, code: str) -> List[ICD10Node]:
        """Get path from node to root.

        Args:
            code: ICD10 code

        Returns:
            List of nodes from root to the specified node
        """
        node = self.get_node(code)
        if node is None:
            return []

        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def get_all_codes(self) -> List[str]:
        """Get all ICD10 codes in hierarchy.

        Returns:
            List of all codes
        """
        return list(self.code_map.keys())

    def get_leaf_codes(self) -> List[str]:
        """Get all leaf codes (most specific codes).

        Returns:
            List of leaf codes
        """
        return [code for code, node in self.code_map.items() if node.is_leaf]

    def get_children(self, code: str) -> List[ICD10Node]:
        """Get children of a code.

        Args:
            code: ICD10 code

        Returns:
            List of child nodes
        """
        node = self.get_node(code)
        if node is None:
            return []
        return node.children

    def get_flattened_list(self) -> List[Dict[str, Any]]:
        """Get flattened list of all codes with descriptions.

        Returns:
            List of dictionaries with code, description, and laymen_definition
        """
        result = []
        for code, node in self.code_map.items():
            if code == "ROOT":
                continue
            result.append(
                {
                    "code": node.code,
                    "description": node.description,
                    "laymen_definition": node.laymen_definition or node.description,
                    "is_leaf": node.is_leaf,
                }
            )
        return result


class ICD10TranscriptDataset:
    """Dataset for loading transcripts with ground truth ICD10 codes."""

    def __init__(
        self, csv_path: str, hierarchy_loader: Optional[ICD10HierarchyLoader] = None
    ):
        """Initialize dataset.

        Args:
            csv_path: Path to CSV file with transcripts
            hierarchy_loader: Optional hierarchy loader for validation
        """
        self.csv_path = Path(csv_path)
        self.hierarchy_loader = hierarchy_loader
        self.df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load data from CSV file."""
        self.df = pd.read_csv(self.csv_path)

        # Parse reference answers to extract ICD10 codes
        self.df["ground_truth_codes"] = self.df["reference_answer"].apply(
            self._parse_reference_answer
        )

    def _parse_reference_answer(self, reference_answer: str) -> List[str]:
        """Parse reference answer to extract ICD10 codes.

        Args:
            reference_answer: String containing codes and descriptions

        Returns:
            List of ICD10 codes
        """
        if pd.isna(reference_answer):
            return []

        # Pattern to match ICD10 codes (e.g., I10, M79.605, E78.5)
        # ICD10 codes can be alphanumeric with dots
        code_pattern = r"\b([A-Z][0-9]{2}(?:\.[0-9A-Z]+)?)\b"
        codes = re.findall(code_pattern, reference_answer)

        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)

        return unique_codes

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df) if self.df is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with transcript and ground truth codes
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded")

        row = self.df.iloc[idx]
        return {
            "encounter_id": row.get("encounter_id", ""),
            "transcript": row.get("transcript", ""),
            "ground_truth_codes": row.get("ground_truth_codes", []),
            "reference_answer": row.get("reference_answer", ""),
            "age": row.get("age", None),
            "age_unit": row.get("age_unit", None),
            "sex": row.get("sex", None),
            "visit_reason": row.get("visit_reason", None),
        }

    def get_metrics(self, predictions: List[List[str]]) -> Dict[str, float]:
        """Compute aggregate metrics.

        Args:
            predictions: List of predicted code lists for each sample

        Returns:
            Dictionary of metrics
        """
        if len(predictions) != len(self):
            raise ValueError(
                f"Predictions length ({len(predictions)}) doesn't match dataset length ({len(self)})"
            )

        # Get ground truth
        ground_truth = [self[i]["ground_truth_codes"] for i in range(len(self))]

        # Compute metrics
        exact_matches = sum(
            1 for pred, gt in zip(predictions, ground_truth) if set(pred) == set(gt)
        )
        exact_match_accuracy = exact_matches / len(self) if len(self) > 0 else 0.0

        # Micro accuracy (code-level)
        all_pred_codes = set()
        all_gt_codes = set()
        for pred, gt in zip(predictions, ground_truth):
            all_pred_codes.update(pred)
            all_gt_codes.update(gt)

        true_positives = len(all_pred_codes & all_gt_codes)
        micro_precision = (
            true_positives / len(all_pred_codes) if len(all_pred_codes) > 0 else 0.0
        )
        micro_recall = (
            true_positives / len(all_gt_codes) if len(all_gt_codes) > 0 else 0.0
        )
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Macro accuracy (sample-level average)
        sample_precisions = []
        sample_recalls = []
        for pred, gt in zip(predictions, ground_truth):
            pred_set = set(pred)
            gt_set = set(gt)
            if len(pred_set) > 0:
                precision = len(pred_set & gt_set) / len(pred_set)
                sample_precisions.append(precision)
            if len(gt_set) > 0:
                recall = len(pred_set & gt_set) / len(gt_set)
                sample_recalls.append(recall)

        macro_precision = (
            sum(sample_precisions) / len(sample_precisions)
            if sample_precisions
            else 0.0
        )
        macro_recall = (
            sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
        )
        macro_f1 = (
            2 * macro_precision * macro_recall / (macro_precision + macro_recall)
            if (macro_precision + macro_recall) > 0
            else 0.0
        )

        # Hierarchical accuracy (partial credit for parent matches)
        hierarchical_accuracy = 0.0
        if self.hierarchy_loader is not None:
            hierarchical_matches = 0
            for pred, gt in zip(predictions, ground_truth):
                pred_set = set(pred)
                gt_set = set(gt)

                # Check exact matches
                if pred_set == gt_set:
                    hierarchical_matches += 1
                    continue

                # Check if any predicted code is in the path of a ground truth code
                matched = False
                for pred_code in pred_set:
                    if pred_code in gt_set:
                        matched = True
                        break
                    # Check if pred_code is ancestor of any gt_code
                    pred_path = self.hierarchy_loader.get_path_to_root(pred_code)
                    pred_codes_in_path = {node.code for node in pred_path}
                    if pred_codes_in_path & gt_set:
                        matched = True
                        break

                if matched:
                    hierarchical_matches += 1

            hierarchical_accuracy = (
                hierarchical_matches / len(self) if len(self) > 0 else 0.0
            )

        return {
            "exact_match_accuracy": exact_match_accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "hierarchical_accuracy": hierarchical_accuracy,
        }
