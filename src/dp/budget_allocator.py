"""
Adaptive ε allocation per privacy block.
Allocates global privacy budget across blocks based on their sensitivities.
"""

import numpy as np
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class AdaptiveBudgetAllocator:
    """Adaptively allocates privacy budget across privacy blocks."""
    
    def __init__(self, method: str = 'proportional'):
        """
        Initialize allocator.
        
        Args:
            method: Allocation method ('proportional', 'equal', 'inverse_sensitivity')
        """
        self.method = method
        self.allocations: Dict[int, float] = {}
    
    def allocate(
        self,
        sensitivities: Dict[int, float],
        total_epsilon: float
    ) -> Dict[int, float]:
        """
        Allocate privacy budget across blocks.
        
        Args:
            sensitivities: Dict mapping block index to sensitivity
            total_epsilon: Total privacy budget (ε)
        
        Returns:
            Dict mapping block index to allocated ε
        """
        if not sensitivities:
            logger.warning("No sensitivities provided")
            return {}
        
        if total_epsilon <= 0:
            raise ValueError(f"Total epsilon must be positive, got {total_epsilon}")
        
        allocations = {}
        
        if self.method == 'equal':
            # Equal allocation
            n_blocks = len(sensitivities)
            epsilon_per_block = total_epsilon / n_blocks
            allocations = {i: epsilon_per_block for i in sensitivities.keys()}
        
        elif self.method == 'proportional':
            # Proportional to sensitivity (higher sensitivity gets more budget)
            total_sensitivity = sum(sensitivities.values())
            if total_sensitivity == 0:
                # Fall back to equal allocation
                n_blocks = len(sensitivities)
                epsilon_per_block = total_epsilon / n_blocks
                allocations = {i: epsilon_per_block for i in sensitivities.keys()}
            else:
                allocations = {
                    i: total_epsilon * (sensitivity / total_sensitivity)
                    for i, sensitivity in sensitivities.items()
                }
        
        elif self.method == 'inverse_sensitivity':
            # Inverse proportional (lower sensitivity gets more budget)
            # This favors blocks with lower sensitivity (better utility)
            inv_sensitivities = {i: 1.0 / (s + 1e-10) for i, s in sensitivities.items()}
            total_inv_sensitivity = sum(inv_sensitivities.values())
            
            allocations = {
                i: total_epsilon * (inv_s / total_inv_sensitivity)
                for i, inv_s in inv_sensitivities.items()
            }
        
        else:
            raise ValueError(f"Unknown allocation method: {self.method}")
        
        # Verify allocation sums to total_epsilon (within floating point error)
        allocated_total = sum(allocations.values())
        if abs(allocated_total - total_epsilon) > 1e-6:
            logger.warning(
                f"Allocation sum ({allocated_total:.6f}) != total_epsilon ({total_epsilon:.6f}). "
                "Normalizing..."
            )
            # Normalize
            scale = total_epsilon / allocated_total
            allocations = {i: eps * scale for i, eps in allocations.items()}
        
        self.allocations = allocations
        
        logger.info(f"Allocated ε={total_epsilon:.4f} across {len(allocations)} blocks using {self.method} method")
        logger.debug(f"Allocations: {allocations}")
        
        return allocations
    
    def get_allocation(self, block_index: int) -> float:
        """Get allocated epsilon for a specific block."""
        if block_index not in self.allocations:
            raise ValueError(f"No allocation for block {block_index}")
        return self.allocations[block_index]
    
    def get_all_allocations(self) -> Dict[int, float]:
        """Get all allocations."""
        return self.allocations.copy()


def allocate_budget(
    sensitivities: Dict[int, float],
    total_epsilon: float,
    method: str = 'proportional'
) -> Dict[int, float]:
    """
    Convenience function for budget allocation.
    
    Args:
        sensitivities: Dict mapping block index to sensitivity
        total_epsilon: Total privacy budget
        method: Allocation method
    
    Returns:
        Dict mapping block index to allocated ε
    """
    allocator = AdaptiveBudgetAllocator(method=method)
    return allocator.allocate(sensitivities, total_epsilon)
