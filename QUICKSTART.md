# ACADP Quick Start Guide

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd ACDP-BDA
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Full Pipeline

Run the complete ACADP pipeline end-to-end:

```bash
python src/pipeline.py --input data/raw_dataset.csv --epsilon 1.0
```

This will:
1. Load and preprocess your dataset
2. Detect correlations and build privacy blocks
3. Apply differential privacy with adaptive budget allocation
4. Generate baseline comparison and evaluation metrics

### Option 2: Run Individual Workflows

#### Workflow 1: Preprocessing
```bash
bash scripts/run_preprocessing.sh data/raw_dataset.csv data/preprocessed.parquet
```

#### Workflow 2: Correlation Detection
```bash
bash scripts/run_correlation.sh data/preprocessed.parquet config/privacy_blocks.json
```

#### Workflow 3: Differential Privacy
```bash
bash scripts/run_dp.sh \
    data/preprocessed.parquet \
    config/privacy_blocks.json \
    config/bounds.json \
    1.0 \
    data/privatized.parquet
```

#### Workflow 4: Evaluation
```bash
bash scripts/run_evaluation.sh \
    data/preprocessed.parquet \
    data/acadp_privatized.parquet \
    data/baseline_privatized.parquet \
    results/comparison.json
```

## Windows Users

If you're on Windows and the `.sh` scripts don't work, you can run the Python modules directly:

```powershell
# Workflow 1
python -m src.ingestion.load_data --input data/raw_dataset.csv --output data/preprocessed.parquet

# Workflow 2
python -m src.correlation.block_builder --input data/preprocessed.parquet --output config/privacy_blocks.json

# Workflow 3
python -m src.dp.privatize --input data/preprocessed.parquet --blocks config/privacy_blocks.json --bounds config/bounds.json --epsilon 1.0 --output data/privatized.parquet

# Workflow 4
python -m src.evaluation.comparisons --original data/preprocessed.parquet --acadp data/acadp_privatized.parquet --baseline data/baseline_privatized.parquet --output results/comparison.json
```

## Example Dataset

To test the pipeline, you can use any structured dataset (CSV or Parquet format) with:
- Numeric features (required for correlation analysis)
- Known or estimable feature bounds
- At least a few hundred rows for meaningful correlation detection

## Configuration

Key parameters you can adjust:

- **Privacy Budget (ε)**: `--epsilon` (default: 1.0)
  - Lower values = more privacy, less utility
  - Higher values = less privacy, more utility

- **Correlation Threshold**: `--corr-threshold` (default: 0.3)
  - Minimum correlation to consider features as dependent
  - Range: 0.0 to 1.0

- **MI Threshold**: `--mi-threshold` (default: 0.1)
  - Minimum mutual information to consider
  - Range: 0.0 to 1.0

- **DP Mechanism**: `--mechanism` (choices: 'laplace', 'gaussian')
  - Laplace: Pure ε-DP
  - Gaussian: (ε, δ)-DP (requires --delta parameter)

- **Budget Allocation**: `--allocation-method` (default: 'proportional')
  - 'proportional': Allocate based on sensitivity
  - 'equal': Equal allocation per block
  - 'inverse_sensitivity': Favor low-sensitivity blocks

## Output Files

After running the pipeline, check the `output/` directory:

- `preprocessed.parquet`: Cleaned and bounded dataset
- `bounds.json`: Feature bounds for DP
- `privacy_blocks.json`: Correlation-aware feature groups
- `acadp_privatized.parquet`: ACADP-privatized dataset
- `baseline_privatized.parquet`: Baseline (feature-independent) DP dataset
- `comparison.json`: Evaluation metrics comparing ACADP vs baseline

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've installed all dependencies (`pip install -r requirements.txt`)

2. **File not found**: Ensure your input dataset path is correct and the file exists

3. **Memory errors**: For very large datasets, consider:
   - Using chunked reading (modify `load_data.py`)
   - Sampling data for correlation detection
   - Using distributed computing (Spark/Dask)

4. **No correlations found**: Lower the correlation threshold or check if your dataset has numeric features

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore the `notebooks/` directory for visualization examples
3. Check `tests/` for unit tests (when available)
4. Review the code in `src/` to understand the implementation

## Support

For questions or issues, refer to the project documentation or contact the project team.
