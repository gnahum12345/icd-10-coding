"""LLM model for ICD10 code extraction."""

from typing import List, Dict, Tuple, Any, Optional, Literal
from dataclasses import dataclass
from openai import OpenAI
import tiktoken
from models.base import BaseModel
from data import ICD10HierarchyLoader, ICD10Node
import logging
from json_repair import repair_json
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM model."""

    provider: str = "openai"  # "openai", "qwen", etc.
    model_name: str = "gpt-5-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_output_tokens: int = 16384
    temperature: float = 0.3
    strategy: Literal["hierarchical", "flattened", "freeform"] = "freeform"


class LLMModel(BaseModel):
    """LLM model for ICD10 code extraction."""

    def __init__(
        self, hierarchy_loader: ICD10HierarchyLoader, config: Optional[LLMConfig] = None
    ):
        """Initialize LLM model.

        Args:
            hierarchy_loader: ICD10HierarchyLoader instance
            config: LLM configuration
        """
        self.hierarchy_loader = hierarchy_loader
        self.config = config or LLMConfig()
        self.client = self._initialize_client()
        self.trace_history: List[Dict[str, Any]] = []

    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client."""
        kwargs = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        return OpenAI(**kwargs)

    def _get_encoding(self) -> tiktoken.Encoding:
        """Get tiktoken encoding for the model.

        Returns:
            tiktoken encoding
        """
        # Map model names to encodings
        # Most modern OpenAI models use cl100k_base

        # Default to cl100k_base for most models
        encoding_name = "cl100k_base"
        return tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Total number of tokens
        """
        encoding = self._get_encoding()
        tokens_per_message = (
            3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = 1  # If there's a name, the role is omitted

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def predict(self, transcript: str, include_outputs: bool = False) -> List[str]:
        """Predict ICD10 codes from transcript.

        Args:
            transcript: Medical transcript text

        Returns:
            List of predicted ICD10 codes
        """
        output = None
        if self.config.strategy == "hierarchical":
            output = self._predict_hierarchical(transcript)
        elif self.config.strategy == "flattened":
            output = self._predict_flattened(transcript)
        elif self.config.strategy == "freeform":
            output = self._predict_freeform(transcript)
        else:
            raise ValueError(f"Invalid strategy: {self.config.strategy}")
        if include_outputs:
            return output
        return output[0]

    def _predict_hierarchical(
        self, transcript: str
    ) -> Tuple[List[str], Dict[str, str]]:
        """Predict ICD10 codes from transcript using hierarchical strategy.

        Args:
            transcript: Medical transcript text

        Returns:
            Tuple of (predicted codes, reasoning dictionary)
            - predicted codes: List of ICD10 codes
            - reasoning: Dict mapping each code to its selection reasoning
        """
        root = self.hierarchy_loader.root_node
        predictions = set()
        reasoning_map = {}  # Track reasoning for each prediction
        max_depth = 10

        def ask_llm(node: ICD10Node, children: List[ICD10Node]) -> List[ICD10Node]:
            """LLM decides which children are relevant (can be multiple)."""
            if not children:
                return []

            child_block = "\n".join(
                f"- {c.code}: {c.laymen_definition or c.description}" for c in children
            )

            node_desc = (
                f"{node.code} — {node.description}"
                if node.code != "ROOT"
                else "ROOT (starting point)"
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an ICD10 hierarchical classifier. "
                        "Select ALL child categories that are relevant to the transcript. "
                        'Respond ONLY with JSON: {"selected": ["CODE1", "CODE2", ...], '
                        '"reasoning": {"CODE1": "brief reason", "CODE2": "brief reason"}}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CURRENT NODE: {node_desc}\n\n"
                        f"CHILD OPTIONS:\n{child_block}\n\n"
                        f"TRANSCRIPT:\n{transcript}\n\n"
                        "Return JSON only with selected codes and reasoning for each. No explanation."
                    ),
                },
            ]

            try:
                self.trace_history.append({"messages": messages})
                resp = self._call_llm(messages)
                print(resp)
                data = self._parse_llm_response(resp)

                # Handle both list and dict responses
                if isinstance(data, list) and len(data) > 0:
                    parsed = data[0] if isinstance(data[0], dict) else {}
                elif isinstance(data, dict):
                    parsed = data
                else:
                    parsed = {}

                selected_codes = parsed.get("selected", [])
                code_reasoning = parsed.get("reasoning", {})

                # Store reasoning for selected codes
                for code in selected_codes:
                    if code not in reasoning_map:
                        reasoning_map[code] = code_reasoning.get(
                            code, "Selected during hierarchical traversal"
                        )

                code_lookup = {c.code: c for c in children}
                return [code_lookup[c] for c in selected_codes if c in code_lookup]
            except Exception as e:
                logger.error(f"Error in ask_llm for node {node.code}: {e}")
                return []

        def explore(node: ICD10Node, depth: int, path: List[str]):
            """
            Recursive, multi-branch traversal.
            Each node may spawn multiple next-level nodes.

            Args:
                node: Current node in traversal
                depth: Current depth level
                path: List of codes representing the path from root to current node
            """
            if depth >= max_depth:
                # Depth exhausted → use node as final prediction
                if node.code != "ROOT":
                    predictions.add(node.code)
                    if node.code not in reasoning_map:
                        reasoning_map[node.code] = (
                            f"Maximum depth reached at {' → '.join(path + [node.code])}"
                        )
                return

            # Leaf node → final prediction
            if node.is_leaf:
                predictions.add(node.code)
                if node.code not in reasoning_map:
                    reasoning_map[node.code] = (
                        f"Leaf node reached via path: {' → '.join(path + [node.code])}"
                    )
                return

            children = node.children if hasattr(node, "children") else []
            if not children:
                if node.code != "ROOT":
                    predictions.add(node.code)
                    if node.code not in reasoning_map:
                        reasoning_map[node.code] = (
                            f"No further children available at {' → '.join(path + [node.code])}"
                        )
                return

            # Ask LLM to choose ANY number of children (possibly many)
            selected_children = ask_llm(node, children)

            # No children selected → treat current node as terminal choice
            if not selected_children:
                if node.code != "ROOT":
                    predictions.add(node.code)
                    if node.code not in reasoning_map:
                        reasoning_map[node.code] = (
                            f"No relevant child categories found; selected this level ({' → '.join(path + [node.code])})"
                        )
                return

            # Recurse into all selected children
            for child in selected_children:
                explore(child, depth + 1, path + [node.code])

        # Start traversal from root
        explore(root, depth=0, path=[])

        return sorted(predictions), reasoning_map

    def _predict_flattened(self, transcript: str) -> List[str]:
        """Predict ICD10 codes from transcript using flattened strategy.

        Args:
            transcript: Medical transcript text

        Returns:
            List of predicted ICD10 codes
        """
        messages = [
            {
                "role": "system",
                "content": "You are a medical coding expert. Given a medical transcript, select the most appropriate ICD10 codes from the list below. Do not hallucinate and you must cite your source",
            },
            {
                "role": "user",
                "content": f"""## ICD10-Codes: \n{self.hierarchy_loader.get_leaf_codes()}""",
            },
            {
                "role": "user",
                "content": """## Task: 
            You are to listen on to a conversation from doctor/patient and predict the ICD10-codes along with the confidence and reason for choosing those codes. Ensure the high confidence cases are first and make sure to get all cases! 
            ## Output Format: 
            You must output a JSON only: 
            ```
            [ 
                {{ 
                    'reason': '<reasoning with citation>', 
                    'confidence': '<number 1-10> where 1 is not confident and 10 is perfectly confident', 
                    'code': '<ICD10-code>',
                }}, 
                ...
            ]
            ```
            """,
            },
            {"role": "user", "content": f"## Transcript:\n{transcript}"},
        ]
        # Store token count in trace history
        return self._predict_with_messages(messages)

    def _predict_freeform(self, transcript: str) -> List[str]:
        """Predict ICD10 codes from transcript using freeform strategy."""
        messages = [
            {
                "role": "system",
                "content": "You are a medical coding expert. Given a medical transcript, select the most appropriate ICD10 codes. Do not hallucinate and you must cite your source",
            },
            {
                "role": "user",
                "content": """## Task: 
            You are to listen on to a conversation from doctor/patient and predict the ICD10-codes along with the confidence and reason for choosing those codes. 
            Please cite your exact source. Ensure the high confidence cases are first and make sure to get all cases! 
            ## Output Format: 
            You must output a JSON only: 
            ```
            [ 
                {{ 
                    'reason': '<reasoning with citation>', 
                    'confidence': '<number 1-10> where 1 is not confident and 10 is perfectly confident', 
                    'code': '<ICD10-code>',
                }}, 
                .... 
            ] 
            """,
            },
            {"role": "user", "content": f"## Transcript:\n{transcript}"},
        ]
        # Store token count in trace history

        return self._predict_with_messages(messages)

    def _predict_with_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[str], Any]:
        self.trace_history.append({"messages": messages})

        # Call LLM
        try:
            response = self._call_llm(messages)
        except Exception as e:
            logger.error("Error occured in LLM Response")
            logger.exception(e)
            return []

        self.trace_history[-1].update({"raw_output": response})
        # Parse response to extract ICD10 codes
        parsed_output = self._parse_llm_response(response)
        codes = [output.get("code") for output in parsed_output]
        codes = [c for c in codes if c]  # filter nulls
        self.trace_history[-1].update({"output": parsed_output, "codes": codes})
        return codes, parsed_output

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM with messages.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature

        Returns:
            LLM response
        """
        # Count input tokens
        input_tokens = self._count_tokens(messages)

        # Check if tokens exceed reasonable limit (most models have context windows)
        # Using a conservative limit of 100k tokens
        if input_tokens > 16_000:
            logger.warning(
                f"Prompt is {input_tokens} length which exceeds the 16k window!"
            )

        try:
            response = self.client.responses.create(
                model=self.config.model_name,
                input=messages,
                temperature=None,
                max_output_tokens=self.config.max_output_tokens,
                metadata={"enable_prompt_caching": "True"},
                #  Explicitly avoiding guided decoding to not over-influence the model.
            )
            content = response.output_text
            # Update trace history with token usage
            if hasattr(response, "usage"):
                usage = response.usage
                if len(self.trace_history) > 0:
                    self.trace_history[-1].update(
                        {
                            "input_tokens": usage.input_tokens
                            if hasattr(usage, "input_tokens")
                            else input_tokens,
                            "output_tokens": usage.output_tokens
                            if hasattr(usage, "output_tokens")
                            else 0,
                            "total_tokens": usage.total_tokens
                            if hasattr(usage, "total_tokens")
                            else input_tokens,
                        }
                    )
            return content if content else ""
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            # Include token count in error message
            logger.error(f"Input messages contained {input_tokens} tokens.")
            return ""

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract ICD10 codes.

        Args:
            response: LLM response string

        Returns:
            List of ICD10 codes
        """
        if not response:
            logger.warning("Response is empty!")
            return []
        try:
            decoded_object = repair_json(response, return_objects=True)
            if isinstance(decoded_object, list):
                return decoded_object
            # add assertion/logging.
            return [decoded_object]
        except Exception as e:
            logger.exception("Json repair failed", e)
        return []

    def predict_with_uncertainty(
        self,
        transcript: str,
        n_samples: int = 10,
        temperature_range: Tuple[float, float] = (0.3, 1.5),
    ) -> Dict[str, Any]:
        """Predict ICD10 codes with uncertainty quantification.

        Args:
            transcript: Medical transcript text
            n_samples: Number of Monte Carlo samples
            temperature_range: Temperature range for sampling (min, max)

        Returns:
            Dictionary containing predictions, confidence, reasoning, and uncertainty metrics
        """
        # TODO: Implement uncertainty quantification
        results = []
        confidence_map = defaultdict(list)
        reasoning_map = defaultdict(list)

        for _ in range(n_samples):
            res, full_output = self.predict(transcript, True)
            results.append(full_output)
            for item in full_output:
                code = item.get("code")
                if not code:
                    continue
                conf = float(item.get("confidence", 0.0))
                reason = item.get("reason", "")
                confidence_map[code].append(conf)
                if reason:
                    reasoning_map[code].append(reason)

        # ---- Build merged final output ----
        final_codes = sorted(confidence_map.keys())

        avg_conf = {c: float(np.mean(confidence_map[c])) for c in final_codes}
        std_conf = {c: float(np.std(confidence_map[c])) for c in final_codes}

        # Sort by *average* confidence descending
        final_sorted = sorted(final_codes, key=lambda c: avg_conf[c], reverse=True)

        return {
            "codes": final_sorted,
            "avg_confidence": avg_conf,
            "std_confidence": std_conf,
            "reasons": reasoning_map,
            "trace_history": self.trace_history,
        }
