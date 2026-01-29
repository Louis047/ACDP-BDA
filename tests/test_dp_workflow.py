"""
Basic sanity tests for Workflow 3 (DP & budget allocation).

These tests use small synthetic data to validate that:
- Block-level sensitivities are computed as expected
- Adaptive budget allocation respects the global epsilon
- DatasetPrivatizer returns bounded, noisy data without shape changes
"""

import pandas as pd
import numpy as np

from src.dp.sensitivity import compute_all_block_sensitivities
from src.dp.budget_allocator import AdaptiveBudgetAllocator
from src.dp.privatize import DatasetPrivatizer


def _make_synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset for tests."""
    rng = np.random.default_rng(42)
    n = 100

    return pd.DataFrame(
        {
            "fare_amount": rng.uniform(0, 50, size=n),
            "trip_distance": rng.uniform(0, 20, size=n),
            "passenger_count": rng.integers(1, 5, size=n),
        }
    )


def test_block_sensitivity_and_allocation_basic():
    df = _make_synthetic_df()

    # Simple two-block structure
    blocks = [
        {"fare_amount", "trip_distance"},
        {"passenger_count"},
    ]

    # Bounds chosen to roughly match synthetic ranges
    bounds = {
        "fare_amount": (0.0, 50.0),
        "trip_distance": (0.0, 20.0),
        "passenger_count": (1.0, 5.0),
    }

    # 1) Sensitivities
    sens = compute_all_block_sensitivities(blocks, df, bounds)
    assert set(sens.keys()) == {0, 1}
    # Sensitivities must be positive
    assert all(s > 0 for s in sens.values())

    # 2) Budget allocation
    total_epsilon = 1.0
    allocator = AdaptiveBudgetAllocator(method="proportional")
    alloc = allocator.allocate(sens, total_epsilon)

    # All blocks get some epsilon
    assert set(alloc.keys()) == {0, 1}
    assert all(eps > 0 for eps in alloc.values())
    # Composition should not exceed total epsilon (within float noise)
    assert abs(sum(alloc.values()) - total_epsilon) < 1e-6


def test_dataset_privatizer_end_to_end():
    df = _make_synthetic_df()

    blocks = [
        {"fare_amount", "trip_distance"},
        {"passenger_count"},
    ]

    bounds = {
        "fare_amount": (0.0, 50.0),
        "trip_distance": (0.0, 20.0),
        "passenger_count": (1.0, 5.0),
    }

    total_epsilon = 0.5

    privatizer = DatasetPrivatizer(
        blocks=blocks,
        bounds=bounds,
        total_epsilon=total_epsilon,
        mechanism="laplace",
        allocation_method="proportional",
        random_state=123,
    )

    privatized = privatizer.privatize(df)

    # Shape and columns must match
    assert list(privatized.columns) == list(df.columns)
    assert privatized.shape == df.shape

    # Values should be different due to noise (probabilistic, so we only
    # check that at least one value changed per feature).
    changed_counts = (privatized != df).sum()
    assert changed_counts["fare_amount"] > 0
    assert changed_counts["trip_distance"] > 0
    assert changed_counts["passenger_count"] > 0

    # All values must stay within bounds
    for col, (lo, hi) in bounds.items():
        assert privatized[col].min() >= lo - 1e-9
        assert privatized[col].max() <= hi + 1e-9

