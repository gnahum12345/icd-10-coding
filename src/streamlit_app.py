"""Streamlit application for ICD10 code extraction evaluation."""

import streamlit as st
import pandas as pd
import yaml
import os
from pathlib import Path
from typing import Optional

from data import ICD10HierarchyLoader, ICD10TranscriptDataset
from models import LLMModel
from models.llm_model import LLMConfig
from evaluator import MetricsCalculator

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None


@st.cache_data
def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        print(Path(__file__).parent)
        config_path = Path(__file__).parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        st.error(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    if "OPENAI_API_KEY" in os.environ:
        config.setdefault("llm", {})["api_key"] = os.environ["OPENAI_API_KEY"]

    return config


@st.cache_resource
def load_hierarchy(xml_path: str) -> ICD10HierarchyLoader:
    """Load ICD10 hierarchy.

    Args:
        xml_path: Path to XML file

    Returns:
        ICD10HierarchyLoader instance
    """
    loader = ICD10HierarchyLoader(xml_path)
    loader.load()
    return loader


def load_dataset(
    csv_path: str, hierarchy_loader: ICD10HierarchyLoader
) -> ICD10TranscriptDataset:
    """Load dataset.

    Args:
        csv_path: Path to CSV file
        hierarchy_loader: ICD10HierarchyLoader instance

    Returns:
        ICD10Dataset instance
    """
    return ICD10TranscriptDataset(csv_path, hierarchy_loader)


def initialize_model(
    hierarchy_loader: ICD10HierarchyLoader,
    config: dict,
    strategy: str,
    model_name: str,
    api_key: Optional[str] = None,
) -> LLMModel:
    """Initialize LLM model.

    Args:
        hierarchy_loader: ICD10HierarchyLoader instance
        config: Configuration dictionary
        strategy: Prediction strategy
        model_name: Model name
        api_key: API key

    Returns:
        LLMModel instance
    """
    llm_config = LLMConfig(
        provider=config.get("llm", {}).get("provider", "openai"),
        model_name=model_name,
        api_key=api_key or config.get("llm", {}).get("api_key"),
        base_url=config.get("llm", {}).get("base_url"),
        max_output_tokens=config.get("llm", {}).get("max_output_tokens", 2000),
        temperature=config.get("llm", {}).get("temperature", 0.7),
        strategy=strategy,
    )
    return LLMModel(hierarchy_loader, llm_config)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="ICD10 Code Extraction", page_icon="🏥", layout="wide"
    )

    st.title("ICD10 Code Extraction System")
    st.markdown("Extract ICD10 codes from medical transcripts using LLMs")

    # Load configuration
    config = load_config()
    if not config:
        st.stop()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # API key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=config.get("llm", {}).get("api_key", ""),
        type="password",
    )
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Model", options=["gpt-5", "gpt-5-mini", "gpt-5-nano"], index=0
    )

    # Strategy selection
    strategy_options = ["freeform", "hierarchical", "flattened"]

    # Read strategy from config safely
    configured_strategy = (
        (config.get("llm", {}).get("strategy", "freeform") or "freeform")
        .strip()
        .lower()
    )

    # Pick index if valid, otherwise default to "freeform"
    default_index = (
        strategy_options.index(configured_strategy)
        if configured_strategy in strategy_options
        else 0  # freeform
    )
    strategy = st.sidebar.selectbox(
        "Strategy", options=strategy_options, index=default_index
    )

    # Load hierarchy and dataset
    xml_path = config.get("data", {}).get(
        "xml_path", "data/FY24-CMS-1785-F-ICD-10-Table-Index/icd10cm_tabular_2024.xml"
    )
    csv_path = config.get("data", {}).get(
        "csv_path", "data/Test_Project_ICD10_Dataset.csv"
    )

    try:
        hierarchy_loader = load_hierarchy(xml_path)
        dataset = load_dataset(csv_path, hierarchy_loader)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Dashboard",
            "Trace Viewer",
            "Individual Transcript",
            "Results Table",
            "Run Full Evaluation",
        ]
    )

    # Dashboard tab
    with tab1:
        st.header("Dashboard")

        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(dataset))
        with col2:
            st.metric("Total Codes", len(hierarchy_loader.get_all_codes()))
        with col3:
            st.metric("Leaf Codes", len(hierarchy_loader.get_leaf_codes()))

        # Code distribution
        st.subheader("Code Distribution")
        code_counts = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            for code in sample["ground_truth_codes"]:
                code_counts[code] = code_counts.get(code, 0) + 1

        if code_counts:
            df_codes = (
                pd.DataFrame(list(code_counts.items()), columns=["Code", "Count"])
                .sort_values("Count", ascending=False)
                .head(20)
            )

            if PLOTLY_AVAILABLE and px is not None:
                fig = px.bar(df_codes, x="Code", y="Count", title="Top 20 ICD10 Codes")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df_codes.set_index("Code"))

        # Evaluation metrics (if available)
        if "evaluation_results" in st.session_state:
            st.subheader("Evaluation Metrics")
            metrics = st.session_state["evaluation_results"]["metrics"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Exact Match", f"{metrics.get('exact_match_accuracy', 0):.4f}"
                )
            with col2:
                st.metric("Micro F1", f"{metrics.get('micro_f1', 0):.4f}")
            with col3:
                st.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
            with col4:
                st.metric(
                    "Hierarchical Accuracy",
                    f"{metrics.get('hierarchical_accuracy', 0):.4f}",
                )

            # Metrics table
            metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": "Exact Match Accuracy",
                        "Value": metrics.get("exact_match_accuracy", 0),
                    },
                    {
                        "Metric": "Micro Precision",
                        "Value": metrics.get("micro_precision", 0),
                    },
                    {"Metric": "Micro Recall", "Value": metrics.get("micro_recall", 0)},
                    {"Metric": "Micro F1", "Value": metrics.get("micro_f1", 0)},
                    {
                        "Metric": "Macro Precision",
                        "Value": metrics.get("macro_precision", 0),
                    },
                    {"Metric": "Macro Recall", "Value": metrics.get("macro_recall", 0)},
                    {"Metric": "Macro F1", "Value": metrics.get("macro_f1", 0)},
                    {
                        "Metric": "Hierarchical Accuracy",
                        "Value": metrics.get("hierarchical_accuracy", 0),
                    },
                    {
                        "Metric": "Hierarchical F1",
                        "Value": metrics.get("hierarchical_f1", 0),
                    },
                ]
            )
            st.dataframe(metrics_df, use_container_width=True)

    # Trace Viewer tab
    with tab2:
        st.header("Trace Viewer")

        if "evaluation_results" not in st.session_state:
            st.info("Run evaluation first to view traces")
        else:
            results = st.session_state["evaluation_results"]
            traces = results.get("traces", [])

            if not traces:
                st.info("No traces available")
            else:
                # Select sample
                sample_idx = st.selectbox(
                    "Select Sample",
                    options=range(len(traces)),
                    format_func=lambda x: f"Sample {x + 1}",
                )

                trace = traces[sample_idx]
                sample = dataset[sample_idx]

                # Display transcript
                st.subheader("Transcript")
                st.text_area("", sample["transcript"], height=200, disabled=True)

                # Display ground truth
                st.subheader("Ground Truth")
                st.write(", ".join(sample["ground_truth_codes"]))

                # Display prediction
                st.subheader("Prediction")
                pred_codes = results["predictions"][sample_idx]
                st.write(", ".join(pred_codes))

                # Display trace
                st.subheader("Trace")
                if isinstance(trace, list):
                    for i, trace_entry in enumerate(trace):
                        st.write(f"**Step {i + 1}:**")
                        st.json(trace_entry)
                else:
                    st.json(trace)

    # Individual Transcript tab
    with tab3:
        st.header("Individual Transcript Explorer")

        # Input transcript
        transcript = st.text_area(
            "Enter Transcript",
            height=300,
            placeholder="Paste medical transcript here...",
        )

        col1, col2 = st.columns(2)
        with col1:
            run_uncertainty = st.checkbox("Run Uncertainty Quantification", value=False)
        with col2:
            n_mc_samples = st.number_input(
                "Monte Carlo Samples",
                min_value=1,
                max_value=50,
                value=10,
                disabled=not run_uncertainty,
            )

        if st.button("Predict", type="primary"):
            if not transcript:
                st.warning("Please enter a transcript")
            elif not api_key:
                st.error("Please enter an OpenAI API key")
            else:
                with st.spinner("Predicting..."):
                    try:
                        model = initialize_model(
                            hierarchy_loader, config, strategy, model_name, api_key
                        )

                        if run_uncertainty:
                            result = model.predict_with_uncertainty(
                                transcript,
                                n_samples=n_mc_samples,
                                temperature_range=tuple(
                                    config.get("evaluation", {}).get(
                                        "temperature_range", [0.3, 1.5]
                                    )
                                ),
                            )

                            # Display results
                            st.subheader("Predicted Codes")
                            st.write(", ".join(result["aggregated_codes"]))

                            st.subheader("Confidence")
                            st.metric("Mean Confidence", f"{result['confidence']:.4f}")
                            st.metric(
                                "Std Confidence", f"{result['confidence_std']:.4f}"
                            )

                            st.subheader("Uncertainty Metrics")
                            uncertainty = result["uncertainty_metrics"]
                            st.json(uncertainty)

                            st.subheader("Reasoning")
                            if result["reasoning"]:
                                st.write(
                                    result["reasoning"][0]
                                    if isinstance(result["reasoning"], list)
                                    else result["reasoning"]
                                )

                            st.subheader("Trace")
                            st.json(result["trace_history"])

                            # Display prediction distribution
                            st.subheader("Prediction Distribution")
                            all_pred_codes = []
                            for pred in result["predictions"]:
                                all_pred_codes.extend(pred)

                            if all_pred_codes:
                                pred_counts = pd.Series(all_pred_codes).value_counts()
                                if PLOTLY_AVAILABLE and px is not None:
                                    fig = px.bar(
                                        x=pred_counts.index,
                                        y=pred_counts.values,
                                        title="Code Prediction Frequency",
                                        labels={"x": "Code", "y": "Count"},
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    df_pred = pd.DataFrame(
                                        {
                                            "Code": pred_counts.index,
                                            "Count": pred_counts.values,
                                        }
                                    )
                                    st.bar_chart(df_pred.set_index("Code"))
                        else:
                            pred_codes = model.predict(transcript)

                            st.subheader("Predicted Codes")
                            st.write(", ".join(pred_codes))

                            st.subheader("Trace")
                            st.json(model.trace_history)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

    # Results Table tab
    with tab4:
        st.header("Results Table")

        if "evaluation_results" not in st.session_state:
            st.info("Run evaluation first to view results")
        else:
            results = st.session_state["evaluation_results"]

            # Create results DataFrame
            data = []
            for i in range(len(results["predictions"])):
                sample = dataset[i]
                pred = results["predictions"][i]
                gt = sample["ground_truth_codes"]

                # Compute accuracy
                exact_match = 1 if set(pred) == set(gt) else 0
                precision = len(set(pred) & set(gt)) / len(set(pred)) if pred else 0
                recall = len(set(pred) & set(gt)) / len(set(gt)) if gt else 0

                data.append(
                    {
                        "Sample": i + 1,
                        "Encounter ID": sample.get("encounter_id", ""),
                        "Predicted Codes": ", ".join(pred),
                        "Ground Truth": ", ".join(gt),
                        "Exact Match": exact_match,
                        "Precision": precision,
                        "Recall": recall,
                        "Confidence": results.get(
                            "confidences", [0.5] * len(results["predictions"])
                        )[i]
                        if "confidences" in results
                        else 0.5,
                    }
                )

            df_results = pd.DataFrame(data)

            # Filtering
            col1, col2 = st.columns(2)
            with col1:
                filter_exact_match = st.checkbox("Show only exact matches", value=False)
            with col2:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)

            if filter_exact_match:
                df_results = df_results[df_results["Exact Match"] == 1]
            if min_confidence > 0:
                df_results = df_results[df_results["Confidence"] >= min_confidence]

            # Display table
            st.dataframe(df_results, use_container_width=True)

            # Run evaluation button
            if st.button("Run Evaluation", type="primary"):
                if not api_key:
                    st.error("Please enter an OpenAI API key")
                else:
                    with st.spinner("Running evaluation..."):
                        try:
                            model = initialize_model(
                                hierarchy_loader, config, strategy, model_name, api_key
                            )

                            evaluator = MetricsCalculator(hierarchy_loader)

                            predictions = []
                            ground_truth = []
                            confidences = []
                            traces = []

                            n_samples = st.slider(
                                "Number of Samples",
                                1,
                                len(dataset),
                                min(10, len(dataset)),
                            )

                            progress_bar = st.progress(0)
                            for i in range(n_samples):
                                sample = dataset[i]
                                transcript = sample["transcript"]
                                gt_codes = sample["ground_truth_codes"]

                                pred_codes = model.predict(transcript)

                                predictions.append(pred_codes)
                                ground_truth.append(gt_codes)
                                confidences.append(0.5)  # Default confidence
                                traces.append(model.trace_history)

                                progress_bar.progress((i + 1) / n_samples)

                            # Compute metrics
                            metrics = evaluator.compute_metrics(
                                predictions, ground_truth, confidences
                            )

                            # Store results
                            st.session_state["evaluation_results"] = {
                                "predictions": predictions,
                                "ground_truth": ground_truth,
                                "confidences": confidences,
                                "traces": traces,
                                "metrics": metrics,
                            }

                            st.success("Evaluation completed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during evaluation: {e}")

    # Run Full Evaluation tab
    with tab5:
        st.header("Run Model on Entire Dataset")

        st.write("""
        This tab runs the selected model on the entire dataset (or a subset), 
        computes metrics, collects traces, and stores everything in `session_state`.
        """)

        # Number of samples
        total_samples = len(dataset)
        n_samples = st.number_input(
            "Number of Samples to Evaluate",
            min_value=1,
            max_value=total_samples,
            value=total_samples,
            step=1,
        )

        # Uncertainty toggle
        run_uncertainty = st.checkbox("Enable Uncertainty Quantification", value=False)
        if run_uncertainty:
            n_mc_samples = st.slider(
                "Monte Carlo Samples", min_value=2, max_value=50, value=10
            )

        # Evaluate button
        if st.button("Run Full Evaluation", type="primary"):
            if not api_key:
                st.error("Please enter an OpenAI API key")
            else:
                try:
                    st.markdown("### Running Evaluation…")
                    progress = st.progress(0)

                    # Initialize model + evaluator
                    model = initialize_model(
                        hierarchy_loader, config, strategy, model_name, api_key
                    )
                    evaluator = MetricsCalculator(hierarchy_loader)

                    predictions = []
                    ground_truth = []
                    confidences = []
                    traces = []

                    # Loop through dataset
                    for i in range(n_samples):
                        sample = dataset[i]
                        transcript = sample["transcript"]
                        gt_codes = sample["ground_truth_codes"]

                        # Predict
                        if run_uncertainty:
                            result = model.predict_with_uncertainty(
                                transcript,
                                n_samples=n_mc_samples,
                                temperature_range=tuple(
                                    config.get("evaluation", {}).get(
                                        "temperature_range", [0.3, 1.5]
                                    )
                                ),
                            )
                            pred_codes = result["aggregated_codes"]
                            confidence = float(result.get("confidence", 0.5))
                            trace = result.get("trace_history", {})
                        else:
                            pred_codes = model.predict(transcript)
                            confidence = 0.5
                            trace = model.trace_history

                        predictions.append(pred_codes)
                        ground_truth.append(gt_codes)
                        confidences.append(confidence)
                        traces.append(trace)

                        progress.progress((i + 1) / n_samples)

                    # Compute metrics
                    metrics = evaluator.compute_metrics(
                        predictions, ground_truth, confidences
                    )

                    # Store results
                    st.session_state["evaluation_results"] = {
                        "predictions": predictions,
                        "ground_truth": ground_truth,
                        "confidences": confidences,
                        "traces": traces,
                        "metrics": metrics,
                    }

                    st.success(
                        "Evaluation complete! Switch to Dashboard, Trace Viewer, or Results Table to inspect results."
                    )
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during full evaluation: {e}")


if __name__ == "__main__":
    main()
