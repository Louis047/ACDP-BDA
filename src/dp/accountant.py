"""
Global privacy accounting.
Tracks privacy budget consumption and composition.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PrivacyAccountant:
    """Tracks privacy budget consumption across multiple queries/blocks."""
    
    def __init__(self, total_epsilon: float, delta: float = 0.0):
        """
        Initialize accountant.
        
        Args:
            total_epsilon: Total privacy budget (ε)
            delta: Privacy parameter (δ) for (ε, δ)-DP
        """
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.allocations: List[Dict] = []
    
    def record_allocation(
        self,
        block_index: int,
        epsilon: float,
        delta: float = 0.0,
        description: Optional[str] = None
    ) -> None:
        """
        Record a privacy budget allocation.
        
        Args:
            block_index: Index of the block/query
            epsilon: Allocated ε
            delta: Allocated δ (for Gaussian mechanism)
            description: Optional description
        """
        self.used_epsilon += epsilon
        self.used_delta += delta
        
        allocation = {
            'block_index': block_index,
            'epsilon': epsilon,
            'delta': delta,
            'description': description
        }
        self.allocations.append(allocation)
        
        logger.debug(f"Recorded allocation: block={block_index}, ε={epsilon:.6f}, δ={delta:.6f}")
    
    def check_budget(self) -> bool:
        """
        Check if budget is within limits.
        
        Returns:
            True if budget is available, False otherwise
        """
        epsilon_ok = self.used_epsilon <= self.total_epsilon
        delta_ok = self.used_delta <= self.delta if self.delta > 0 else True
        
        if not epsilon_ok:
            logger.warning(
                f"Privacy budget exceeded: used ε={self.used_epsilon:.6f} > total ε={self.total_epsilon:.6f}"
            )
        
        return epsilon_ok and delta_ok
    
    def get_remaining_budget(self) -> Dict[str, float]:
        """
        Get remaining privacy budget.
        
        Returns:
            Dict with 'epsilon' and 'delta' remaining
        """
        return {
            'epsilon': max(0.0, self.total_epsilon - self.used_epsilon),
            'delta': max(0.0, self.delta - self.used_delta)
        }
    
    def get_summary(self) -> Dict:
        """
        Get summary of privacy accounting.
        
        Returns:
            Dict with accounting summary
        """
        remaining = self.get_remaining_budget()
        
        return {
            'total_epsilon': self.total_epsilon,
            'total_delta': self.delta,
            'used_epsilon': self.used_epsilon,
            'used_delta': self.used_delta,
            'remaining_epsilon': remaining['epsilon'],
            'remaining_delta': remaining['delta'],
            'n_allocations': len(self.allocations),
            'within_budget': self.check_budget()
        }
    
    def print_summary(self) -> None:
        """Print accounting summary to console."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("PRIVACY BUDGET ACCOUNTING")
        print("=" * 60)
        print(f"Total ε: {summary['total_epsilon']:.6f}")
        print(f"Used ε:  {summary['used_epsilon']:.6f}")
        print(f"Remaining ε: {summary['remaining_epsilon']:.6f}")
        
        if summary['total_delta'] > 0:
            print(f"\nTotal δ: {summary['total_delta']:.6f}")
            print(f"Used δ:  {summary['used_delta']:.6f}")
            print(f"Remaining δ: {summary['remaining_delta']:.6f}")
        
        print(f"\nNumber of allocations: {summary['n_allocations']}")
        print(f"Within budget: {summary['within_budget']}")
        print("=" * 60)


def compose_epsilon(epsilon_values: List[float]) -> float:
    """
    Compose multiple ε-DP mechanisms.
    
    For sequential composition: total ε = sum(ε_i)
    
    Args:
        epsilon_values: List of ε values
    
    Returns:
        Composed ε
    """
    return sum(epsilon_values)


def compose_gaussian(epsilon_values: List[float], delta_values: List[float]) -> tuple:
    """
    Compose multiple (ε, δ)-DP mechanisms.
    
    Uses advanced composition (simplified version).
    For sequential composition: total ε = sum(ε_i), total δ = sum(δ_i)
    
    Args:
        epsilon_values: List of ε values
        delta_values: List of δ values
    
    Returns:
        Tuple of (composed_ε, composed_δ)
    """
    if len(epsilon_values) != len(delta_values):
        raise ValueError("epsilon_values and delta_values must have same length")
    
    composed_epsilon = sum(epsilon_values)
    composed_delta = sum(delta_values)
    
    return composed_epsilon, composed_delta
