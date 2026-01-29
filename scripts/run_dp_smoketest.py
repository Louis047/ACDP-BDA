"""
Small smoke test for Workflow 3 (DP & budget allocation).

Run with:

    # from project root, with your virtualenv activated
    python scripts/run_dp_smoketest.py

This does NOT touch any real data; it only uses synthetic data to check that:
- sensitivities and allocations are computed without errors
- privatized data has the same shape and stays within bounds
"""

import pandas as pd
import numpy as np

from src.dp.sensitivity import compute_all_block_sensitivities
from src.dp.budget_allocator import AdaptiveBudgetAllocator
from src.dp.privatize import DatasetPrivatizer


def make_synthetic_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "fare_amount": rng.uniform(0, 50, size=n),
            "trip_distance": rng.uniform(0, 20, size=n),
            "passenger_count": rng.integers(1, 5, size=n),
        }
    )


def main() -> None:
    print("=== DP Workflow 3 Smoke Test ===")
    df = make_synthetic_df()
    print(f"Input shape: {df.shape}")

    blocks = [
        {"fare_amount", "trip_distance"},
        {"passenger_count"},
    ]

    bounds = {
        "fare_amount": (0.0, 50.0),
        "trip_distance": (0.0, 20.0),
        "passenger_count": (1.0, 5.0),
    }

    # 1) Sensitivities
    sens = compute_all_block_sensitivities(blocks, df, bounds)
    print("Block sensitivities:", sens)

    # 2) Budget allocation
    total_epsilon = 0.5
    allocator = AdaptiveBudgetAllocator(method="proportional")
    alloc = allocator.allocate(sens, total_epsilon)
    print("Allocated epsilons:", alloc)
    print("Sum of epsilons:", sum(alloc.values()))

    # 3) Privatization
    privatizer = DatasetPrivatizer(
        blocks=blocks,
        bounds=bounds,
        total_epsilon=total_epsilon,
        mechanism="laplace",
        allocation_method="proportional",
        random_state=123,
    )
    privatized = privatizer.privatize(df)
    print(f"Privatized shape: {privatized.shape}")

    # Basic checks
    assert list(privatized.columns) == list(df.columns)
    assert privatized.shape == df.shape

    for col, (lo, hi) in bounds.items():
        col_min = privatized[col].min()
        col_max = privatized[col].max()
        print(f"{col}: [{col_min:.3f}, {col_max:.3f}] within [{lo}, {hi}]")
        assert col_min >= lo - 1e-9
        assert col_max <= hi + 1e-9

    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()

