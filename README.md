# Adaptive Correlation-Aware Differential Privacy (ACADP)

> **Team note (important):**  
> - **Do not commit directly to `master`/`main`.** Always create a feature branch for your workflow or task.  
> - Keep **Workflow 1–4 modules decoupled** and respect the directory structure (`src/ingestion/`, `src/correlation/`, `src/dp/`, `src/evaluation/`).  
> - Large datasets (≥10 GB) **must not be committed** to the repo; only small samples or paths/configs should be tracked.  
> - When adding new notebooks, put them under `notebooks/` and treat them as exploratory/prototype code; production logic must live in `src/`.

A scalable, batch-oriented framework for applying Differential Privacy to structured Big Data by automatically detecting feature correlations, grouping correlated attributes into privacy blocks, and adaptively allocating privacy budgets to improve data utility while preserving formal DP guarantees.

## Git Workflow for Teammates (Branching & PRs)

This section explains how each teammate should work **from their own branch** and open a **pull request (PR)** to merge changes into `master` without breaking the main codebase.

### 1. One-time setup (clone + install)

```bash
git clone https://github.com/<your-username>/ACDP-BDA.git
cd ACDP-BDA
pip install -r requirements.txt
```

### 2. Create a new branch for your work

Pick a descriptive branch name for your task, for example:
- `feature/workflow1-ingestion`
- `feature/workflow2-correlation`
- `feature/workflow3-dp-budget`
- `feature/workflow4-evaluation`

From inside the repo:

```bash
# Make sure you are on master and up to date
git checkout master
git pull origin master

# Create and switch to your feature branch
git checkout -b feature/workflow3-dp-budget
```

Now do **all your coding and commits on this feature branch**, not on `master`.

### 3. Commit your changes locally

After editing files:

```bash
git status          # See what changed
git add <files>     # Or: git add .
git commit -m "Short, clear message about the change"
```

You can commit multiple times while you work.

### 4. Push your branch to GitHub

```bash
git push -u origin feature/workflow3-dp-budget
```

If this is the first push for that branch, Git will create a remote branch with the same name.

### 5. Open a Pull Request (PR)

1. Go to the GitHub page for the repository.  
2. GitHub will usually show a banner like **"Compare & pull request"** for your new branch – click it.  
3. Ensure:
   - **Base branch**: `master`
   - **Compare branch**: your feature branch (e.g., `feature/workflow3-dp-budget`)
4. Fill in:
   - **Title**: short summary (e.g., "Add DP budget allocator module")  
   - **Description**: what you changed, which workflow it affects, any TODOs or known limitations.
5. Click **Create pull request**.

Ask at least one teammate to **review and approve** the PR before merging.

### 6. Merge the PR (after review)

Once the PR is approved and checks (if any) are passing:

1. Use the GitHub UI to click **Merge pull request** → **Confirm merge**.  
2. After merging, update your local `master`:

```bash
git checkout master
git pull origin master
```

3. Optionally delete the feature branch (both on GitHub and locally) once it’s no longer needed:

```bash
git branch -d feature/workflow3-dp-budget          # delete local branch
git push origin --delete feature/workflow3-dp-budget  # delete remote branch
```

### 7. Keeping your branch up to date (optional but recommended)

If `master` has moved ahead while you are still working:

```bash
git checkout master
git pull origin master         # get latest master

git checkout feature/your-branch
git merge master               # or: git rebase master (for advanced users)
```

Resolve any merge conflicts locally, test, then push your updated branch and continue your PR.


## Project Overview

ACADP addresses a critical limitation in traditional Differential Privacy (DP) approaches: the assumption that features are independent. In real-world structured datasets, features are often correlated, leading to over-allocation of privacy budget and reduced data utility. ACADP automatically detects these correlations, groups correlated features into **privacy blocks**, and adaptively allocates the global privacy budget (ε) across these blocks to maximize utility while maintaining formal DP guarantees.

### Key Innovation

**Correlation-Aware Privacy**: Instead of treating all features independently, ACADP:
1. **Detects correlations** using statistical dependency measures (Pearson correlation, Mutual Information)
2. **Groups correlated features** into privacy blocks that share joint sensitivity
3. **Adaptively allocates** the global ε budget across blocks based on their estimated sensitivities
4. **Preserves formal DP** guarantees through rigorous sensitivity analysis and mechanism composition

This approach significantly improves data utility compared to feature-independent DP baselines, especially for large-scale structured datasets (≥10 GB).

## Scope

### In Scope
- **Batch-mode processing** of structured, tabular Big Data
- **Real-world datasets** (≥10 GB scale)
- **Distributed computation** using Spark/Dask-compatible logic
- **Non-interactive data publishing** (one-time release)
- **Statistical correlation detection** (Pearson, Mutual Information)
- **Formal Differential Privacy** with provable guarantees

### Out of Scope
- Streaming data processing
- Unstructured data (text, images, etc.)
- ML-based correlation learning
- Interactive query systems
- Real-time privacy mechanisms

## Dataset Description and Scale Assumptions

ACADP is designed for **structured, tabular Big Data** with the following characteristics:

- **Format**: CSV, Parquet, or similar tabular formats
- **Scale**: ≥10 GB datasets with millions of rows and hundreds to thousands of features
- **Data Types**: Mixed (numeric, categorical, ordinal)
- **Correlation Structure**: Real-world datasets with inherent feature dependencies
- **Bounded Features**: All features must have known or estimated bounds (required for DP)

The framework assumes datasets are stored in distributed file systems (HDFS, S3) or local storage, and can be processed in batch mode using distributed computing frameworks.

## End-to-End Pipeline

The ACADP pipeline consists of four sequential workflows:

```
Raw Dataset
  ↓
[Workflow 1] Ingestion & Preprocessing
  ↓
[Workflow 2] Correlation Detection & Feature Grouping
  ↓
[Workflow 3] Sensitivity Estimation & Adaptive Budget Allocation
  ↓
[Workflow 4] Differential Privacy Mechanisms
  ↓
DP-Compliant Dataset
  ↓
[Workflow 4] Evaluation & Downstream Analytics
```

### Workflow 1: Data Ingestion & Preprocessing
- Batch loading of large datasets (CSV/Parquet)
- Schema validation and type enforcement
- Encoding and normalization
- **Critical**: Enforce feature bounds (required for DP sensitivity)
- Scalable sampling for correlation estimation
- Basic dataset statistics

### Workflow 2: Correlation Detection & Feature Grouping
- Dimensionality reduction and feature pruning
- Approximate Pearson correlation computation
- Discretized Mutual Information estimation
- Threshold-based correlation pruning
- Dependency graph construction
- **Privacy block generation**: Groups of correlated features

### Workflow 3: Differential Privacy & Budget Allocation
- **Block-level joint sensitivity** estimation
- **Adaptive ε allocation** per privacy block
- Laplace/Gaussian mechanism implementation
- Privacy-preserving noise injection
- Global privacy accounting (composition)

### Workflow 4: Evaluation & Validation
- Baseline comparison (feature-independent DP)
- Utility metrics (error, variance, correlation preservation)
- Scalability profiling (runtime, memory)
- Visualization of results

## Repository Structure

### Root Directory
```
acdap/
├── README.md                     # This file
├── config/                       # Centralized configuration
├── data/                         # Dataset references and samples
├── src/                          # Core implementation
├── notebooks/                    # Lightweight debugging & visualization
├── scripts/                      # Execution entrypoints
├── tests/                        # Unit and pipeline tests
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

### Workflow-Wise File Structure

#### Workflow 1: Data Ingestion & Preprocessing
**Location**: `src/ingestion/`

```
src/ingestion/
├── load_data.py        # Batch loading (CSV/Parquet)
├── schema.py           # Schema validation & typing
├── preprocess.py       # Encoding, normalization
├── bounds.py           # Enforce feature bounds (DP requirement)
├── sampling.py         # Scalable sampling for estimation
└── stats.py            # Basic dataset statistics
```

**Responsibilities**:
- Handle large files efficiently (streaming/chunked reading)
- Ensure all features are bounded (required for DP sensitivity)
- Output clean, validated datasets ready for correlation analysis

**Usage**:
```bash
python -m src.ingestion.load_data --input data/raw_dataset.csv --output data/preprocessed.parquet
```

#### Workflow 2: Correlation & Feature Grouping
**Location**: `src/correlation/`

```
src/correlation/
├── feature_filter.py   # Dimensionality reduction & pruning
├── approx_corr.py      # Approximate Pearson correlation
├── mi_estimation.py    # Discretized Mutual Information
├── prune_pairs.py      # Threshold-based pruning
├── graph_builder.py    # Dependency graph construction
└── block_builder.py    # Privacy block generation
```

**Responsibilities**:
- Avoid full pairwise computation (use sampling/approximation)
- Support large-scale execution (distributed-friendly)
- Output block definitions only (no DP mechanisms here)

**Usage**:
```bash
python -m src.correlation.block_builder --input data/preprocessed.parquet --output config/privacy_blocks.json
```

#### Workflow 3: Differential Privacy & Budget Allocation
**Location**: `src/dp/`

```
src/dp/
├── sensitivity.py      # Block-level joint sensitivity
├── budget_allocator.py # Adaptive ε allocation per block
├── mechanisms.py       # Laplace / Gaussian mechanisms
├── privatize.py        # Apply DP noise
└── accountant.py       # Global privacy accounting
```

**Responsibilities**:
- Preserve DP correctness (rigorous sensitivity analysis)
- Respect global ε constraint (composition)
- No assumptions about data semantics

**Usage**:
```bash
python -m src.dp.privatize --input data/preprocessed.parquet --blocks config/privacy_blocks.json --epsilon 1.0 --output data/privatized.parquet
```

#### Workflow 4: Evaluation & Validation
**Location**: `src/evaluation/`

```
src/evaluation/
├── baselines.py        # Feature-independent DP baseline
├── utility_metrics.py  # Error, variance, correlation loss
├── scalability.py      # Runtime & memory profiling
├── comparisons.py      # ACADP vs baseline
└── plots.py            # Result visualization
```

**Responsibilities**:
- Provide defensible comparisons with baselines
- Demonstrate utility gains from correlation awareness
- Document limitations honestly

**Usage**:
```bash
python -m src.evaluation.comparisons --acadp data/privatized.parquet --baseline data/baseline_privatized.parquet --output results/comparison.json
```

### Pipeline Orchestration
**Location**: `src/pipeline.py`

End-to-end execution controller that orchestrates all four workflows sequentially.

**Usage**:
```bash
python src/pipeline.py --input data/raw_dataset.csv --epsilon 1.0 --output data/final_privatized.parquet
```

### Execution Scripts
**Location**: `scripts/`

```
scripts/
├── run_preprocessing.sh    # Execute Workflow 1 only
├── run_correlation.sh     # Execute Workflow 2 only
├── run_dp.sh              # Execute Workflow 3 only
├── run_evaluation.sh      # Execute Workflow 4 only
└── run_full_pipeline.sh   # Execute all workflows end-to-end
```

Each script is runnable independently for modular testing and review.

## Running the Pipeline

### Prerequisites
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place raw dataset in `data/` directory
   - Ensure dataset is in CSV or Parquet format
   - All features must have known bounds (or will be estimated)

### Running Individual Workflows

#### Workflow 1: Preprocessing
```bash
bash scripts/run_preprocessing.sh
# Or directly:
python -m src.ingestion.load_data --input data/raw_dataset.csv --output data/preprocessed.parquet
```

#### Workflow 2: Correlation Detection
```bash
bash scripts/run_correlation.sh
# Or directly:
python -m src.correlation.block_builder --input data/preprocessed.parquet --output config/privacy_blocks.json
```

#### Workflow 3: Differential Privacy
```bash
bash scripts/run_dp.sh
# Or directly:
python -m src.dp.privatize --input data/preprocessed.parquet --blocks config/privacy_blocks.json --epsilon 1.0 --output data/privatized.parquet
```

#### Workflow 4: Evaluation
```bash
bash scripts/run_evaluation.sh
# Or directly:
python -m src.evaluation.comparisons --acadp data/privatized.parquet --baseline data/baseline_privatized.parquet
```

### Running Full Pipeline
```bash
bash scripts/run_full_pipeline.sh
# Or directly:
python src/pipeline.py --input data/raw_dataset.csv --epsilon 1.0 --output data/final_privatized.parquet
```

## Evaluation Methodology

### Baselines
1. **Feature-Independent DP**: Apply DP to each feature independently with uniform ε allocation
2. **Naive Blocking**: Random feature grouping (no correlation awareness)

### Utility Metrics
- **Mean Squared Error (MSE)**: Per-feature and aggregate
- **Variance Preservation**: Compare original vs. privatized variance
- **Correlation Preservation**: Measure correlation structure retention
- **Downstream ML Performance**: Train models on privatized data and compare accuracy

### Privacy Metrics
- **Formal ε-DP Guarantee**: Verify composition and accounting
- **Privacy Budget Utilization**: Track ε allocation across blocks

### Scalability Metrics
- **Runtime**: End-to-end processing time
- **Memory Usage**: Peak memory consumption
- **Scalability**: Performance on datasets of varying sizes (10 GB, 50 GB, 100 GB+)

### Expected Results
ACADP should demonstrate:
- **Better utility** than feature-independent DP (lower MSE, preserved correlations)
- **Scalability** beyond prototype datasets (handles ≥10 GB efficiently)
- **Clear alignment** between design, implementation, and results
- **Explicit documentation** of assumptions and failure cases

## Development Model

- **Local Development**: Full-scale runs on local high-compute machine
- **Git-based Collaboration**: Version control for integration
- **Cloud Notebooks**: Only for debugging and visualization (not for production runs)
- **Final Experiments**: Executed on single high-compute machine

## Design Principles

1. **Batch-mode processing** of structured, tabular Big Data
2. **Correlation-aware privacy** via statistical dependency detection
3. **Automated pipeline** driven by data statistics, not heuristics
4. **Scalable execution** using Spark/Dask-compatible logic
5. **Strict module separation** for correctness and review clarity
6. **Differential Privacy correctness** is non-negotiable

## Limitations and Assumptions

### Assumptions
- Features are bounded (required for DP sensitivity)
- Correlations are stable across the dataset
- Dataset fits in distributed memory (or can be processed in chunks)
- Global privacy budget (ε) is provided as input

### Known Limitations
- Correlation detection may miss non-linear dependencies (only uses linear and MI)
- Privacy block construction is greedy (may not be globally optimal)
- Budget allocation is heuristic-based (not theoretically optimal)
- Large numbers of highly correlated features may still require significant privacy budget

## Contributing

This is a Final Year Project. For questions or issues, contact the project team.

## License

[Specify license if applicable]

## References

[Add relevant papers and references]
