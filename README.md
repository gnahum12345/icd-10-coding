# ICD10 Code Extraction System

Extract ICD10 codes from medical transcripts using LLMs with hierarchical prediction, uncertainty quantification, and interactive evaluation.

## Full Slide Deck: 

- https://shorturl.at/sWW9e 

## Features

- **Hierarchical Prediction**: Layer-by-layer prediction through ICD10 hierarchy
- **Flattened Prediction**: Single-shot prediction from flattened code list
- **Uncertainty Quantification**: Monte Carlo simulation with varying temperatures
- **Evaluation Metrics**: Micro/macro F1, hierarchical accuracy, uncertainty metrics
- **Interactive Dashboard**: Streamlit app for exploration and evaluation
- **CLI Tool**: Command-line interface for batch evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd icd10_prediction
```

2. Install dependencies:
```bash
pip install -e .
```

Or install manually:
```bash
pip install openai streamlit pandas numpy lxml pyyaml scikit-learn plotly
```

3. Set up configuration:
- Copy `config/config.yaml` and update with your API keys
- Set `OPENAI_API_KEY` environment variable (recommended)

## Usage

### CLI Evaluation

Run batch evaluation from command line:

```bash
python -m src.main --strategy hierarchical --model gpt-4 --samples 10
```

Options:
- `--strategy`: Prediction strategy (`hierarchical` or `flattened`)
- `--model`: Model name (e.g., `gpt-4`, `gpt-3.5-turbo`)
- `--samples`: Number of samples to evaluate
- `--uncertainty`: Run uncertainty quantification
- `--n-mc-samples`: Number of Monte Carlo samples for uncertainty

### Streamlit App

Launch the interactive dashboard:

```bash
streamlit run src/streamlit_app.py
```

The app provides:
- **Dashboard**: Overall metrics and dataset overview
- **Trace Viewer**: View LLM traces for individual samples
- **Individual Transcript**: Test custom transcripts
- **Results Table**: Browse and filter evaluation results

### Python API

Use the system programmatically:

```python
from src.data import ICD10HierarchyLoader, ICD10Dataset
from src.models import LLMModel
from src.models.llm_model import LLMConfig
from src.evaluator import MetricsCalculator

# Load hierarchy and dataset
hierarchy_loader = ICD10HierarchyLoader("data/FY24-CMS-1785-F-ICD-10-Table-Index/icd10cm_tabular_2024.xml")
hierarchy_loader.load()

dataset = ICD10Dataset("data/Test_Project_ICD10_Dataset.csv", hierarchy_loader)

# Initialize model
config = LLMConfig(
    provider="openai",
    model_name="gpt-4",
    strategy="hierarchical"
)
model = LLMModel(hierarchy_loader, config)

# Predict
transcript = "Patient presents with hypertension and diabetes..."
codes = model.predict(transcript)

# Evaluate
evaluator = MetricsCalculator(hierarchy_loader)
predictions = [model.predict(dataset[i]["transcript"]) for i in range(10)]
ground_truth = [dataset[i]["ground_truth_codes"] for i in range(10)]
metrics = evaluator.compute_metrics(predictions, ground_truth)
```

## Configuration

Edit `config/config.yaml` to customize:

- **LLM Settings**: Model, API key, temperature, strategy
- **Data Paths**: XML and CSV file paths
- **Evaluation**: Number of samples, temperature range
- **Codify**: Optional Codify integration settings

## Project Structure

```
icd10_prediction/
├── src/
│   ├── data.py              # Data loading and processing
│   ├── models/
│   │   ├── base.py          # Base model class
│   │   └── llm_model.py     # LLM model implementation
│   ├── evaluator.py         # Evaluation metrics
│   ├── streamlit_app.py     # Streamlit dashboard
│   └── main.py              # CLI entry point
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── FY24-CMS-1785-F-ICD-10-Table-Index/
│   │   └── icd10cm_tabular_2024.xml
│   └── Test_Project_ICD10_Dataset.csv
└── README.md
```

## Evaluation Metrics

The system computes:

- **Exact Match Accuracy**: Percentage of samples with exact code matches
- **Micro Metrics**: Code-level precision, recall, F1
- **Macro Metrics**: Sample-level averaged precision, recall, F1
- **Hierarchical Accuracy**: Partial credit for parent/child matches
- **Uncertainty Metrics**: Calibration, confidence distribution, ECE

## Notes

- The system requires an OpenAI API key for LLM predictions
- Large datasets may take significant time to process
- Monte Carlo simulation for uncertainty quantification increases computation time
- Codify integration is optional and not yet implemented

## License

[Add your license here]

