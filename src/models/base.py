"""Base model class for ICD10 code extraction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional


class BaseModel(ABC):
    """Abstract base class for ICD10 code extraction models."""
    
    @abstractmethod
    def predict(self, transcript: str) -> List[str]:
        """Predict ICD10 codes from transcript.
        
        Args:
            transcript: Medical transcript text
            
        Returns:
            List of predicted ICD10 codes
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(
        self,
        transcript: str,
        n_samples: int = 10,
        temperature_range: Tuple[float, float] = (0.3, 1.5)
    ) -> Dict[str, Any]:
        """Predict ICD10 codes with uncertainty quantification.
        
        Args:
            transcript: Medical transcript text
            n_samples: Number of Monte Carlo samples
            temperature_range: Temperature range for sampling (min, max)
            
        Returns:
            Dictionary containing:
                - predictions: List of predicted code lists
                - confidence: Confidence scores
                - reasoning: Reasoning explanations
                - uncertainty_metrics: Uncertainty statistics
        """
        pass
    
    def validate_codes(self, codes: List[str], hierarchy_loader: Any) -> List[str]:
        """Validate codes against hierarchy.
        
        Args:
            codes: List of codes to validate
            hierarchy_loader: ICD10HierarchyLoader instance
            
        Returns:
            List of valid codes
        """
        valid_codes = []
        for code in codes:
            node = hierarchy_loader.get_node(code)
            if node is not None:
                valid_codes.append(code)
        return valid_codes

