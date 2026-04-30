<p align="center">
  <h1 align="center">ACADP: Adaptive Correlation-Aware Differential Privacy</h1>
  <p align="center">A high-utility framework for preserving statistical properties and machine learning performance in privacy-preserving data publishing.</p>
</p>

## Project Overview

ACADP is an advanced Differential Privacy (DP) framework designed to address the "utility gap" in traditional privacy methods. Conventional DP often injects excessive noise into correlated features, destroying the statistical relationships necessary for accurate data analysis. 

This project implements an intelligent "Correlation-Aware" strategy that groups related features into blocks and applies an optimized budget allocation. By understanding how data features relate to one another, ACADP minimizes error and preserves the underlying structure of the dataset while maintaining rigorous mathematical privacy guarantees.

## Key Performance Improvements

The following results were captured using the NYC Taxi dataset (200,000+ records) comparing ACADP against standard Uniform DP baselines at $\epsilon=1.0$:

| Metric Category | Improvement vs. Uniform | Description |
|:--- |:---:|:--- |
| **Statistical Accuracy (MAE)** | **+1.5%** | Reduction in Mean Absolute Error across all features. |
| **Distributional Similarity** | **+3.3%** | Lower KL-Divergence, meaning privatized distributions stay closer to original data. |
| **ML Prediction (RMSE)** | **+13.0%** | ACADP produces data that is significantly more useful for training regression models. |
| **High-Privacy Scaling** | **+7.1%** | At stricter privacy levels ($\epsilon=5.0$), ACADP's smart allocation yields even higher utility gains. |

## Core Components

- **Adaptive Block Builder:** Uses community detection and mutual information to automatically group correlated features.
- **Optimal Budget Allocator:** A mathematical engine that distributes the "privacy budget" to minimize total expected error across the dataset.
- **Validation Engine:** A comprehensive suite of 8+ statistical metrics and downstream ML benchmarks to prove data utility.
- **Visual Analytics:** Generates publication-quality charts to compare data distributions and correlation preservation.

## Development & Environment

### Technical Stack
- **Language:** Python 3.10+
- **Core Libraries:** Pandas, NumPy, Scikit-Learn, NetworkX, SciPy
- **Data Format:** Apache Parquet for high-performance storage
- **Visualization:** Matplotlib, Seaborn

### Project Structure
- `src/dp/`: Core privacy logic, sensitivity calculation, and noise injection.
- `src/correlation/`: Feature grouping and dependency analysis logic.
- `src/evaluation/`: Benchmarking modules, baselines, and plotting utilities.
- `scripts/`: Utility scripts for data ingestion and pipeline execution.
- `output/`: Generated privatized datasets, JSON results, and analysis plots.

## Getting Started

1. **Setup Environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Data:**
   ```powershell
   powershell -File scripts/download_data.ps1
   ```

3. **Run Pipeline:**
   ```powershell
   python run_pipeline.py
   ```
   *Note: Results and plots will be generated in the `output/` directory.*
