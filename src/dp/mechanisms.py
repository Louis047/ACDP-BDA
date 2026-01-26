"""
Laplace and Gaussian mechanisms for Differential Privacy.
Implements standard DP noise addition mechanisms.
"""

import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    random_state: Optional[int] = None
) -> float:
    """
    Apply Laplace mechanism to a single value.
    
    Adds Laplace noise: Lap(Δ/ε) where Δ is sensitivity and ε is privacy budget.
    
    Args:
        value: True value to privatize
        sensitivity: Sensitivity (Δ)
        epsilon: Privacy budget (ε)
        random_state: Random seed
    
    Returns:
        Privatized value
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")
    if sensitivity < 0:
        raise ValueError(f"Sensitivity must be non-negative, got {sensitivity}")
    
    # Scale parameter for Laplace distribution
    scale = sensitivity / epsilon
    
    rng = np.random.default_rng(random_state)
    noise = rng.laplace(loc=0.0, scale=scale)
    
    privatized_value = value + noise
    
    return privatized_value


def gaussian_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
    random_state: Optional[int] = None
) -> float:
    """
    Apply Gaussian mechanism to a single value.
    
    Adds Gaussian noise for (ε, δ)-DP.
    Uses standard deviation: σ = (Δ * sqrt(2*ln(1.25/δ))) / ε
    
    Args:
        value: True value to privatize
        sensitivity: Sensitivity (Δ)
        epsilon: Privacy budget (ε)
        delta: Privacy parameter (δ), typically 1e-5 or smaller
        random_state: Random seed
    
    Returns:
        Privatized value
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"Delta must be in (0, 1), got {delta}")
    if sensitivity < 0:
        raise ValueError(f"Sensitivity must be non-negative, got {sensitivity}")
    
    # Standard deviation for Gaussian noise
    # Using the standard formula for (ε, δ)-DP
    c = np.sqrt(2 * np.log(1.25 / delta))
    sigma = (sensitivity * c) / epsilon
    
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=sigma)
    
    privatized_value = value + noise
    
    return privatized_value


def add_laplace_noise(
    values: np.ndarray,
    sensitivity: float,
    epsilon: float,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add Laplace noise to an array of values.
    
    Args:
        values: Array of true values
        sensitivity: Sensitivity (Δ)
        epsilon: Privacy budget (ε)
        random_state: Random seed
    
    Returns:
        Array of privatized values
    """
    scale = sensitivity / epsilon
    rng = np.random.default_rng(random_state)
    noise = rng.laplace(loc=0.0, scale=scale, size=values.shape)
    return values + noise


def add_gaussian_noise(
    values: np.ndarray,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to an array of values.
    
    Args:
        values: Array of true values
        sensitivity: Sensitivity (Δ)
        epsilon: Privacy budget (ε)
        delta: Privacy parameter (δ)
        random_state: Random seed
    
    Returns:
        Array of privatized values
    """
    c = np.sqrt(2 * np.log(1.25 / delta))
    sigma = (sensitivity * c) / epsilon
    
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=sigma, size=values.shape)
    return values + noise
