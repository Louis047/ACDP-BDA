"""
Adaptive ε allocation per privacy block.
Allocates global privacy budget across blocks based on their properties.

Key insight: For Laplace mechanism with sequential composition,
ε_per_feature = ε_block / n_features. So blocks with more features
need proportionally more ε to achieve the same per-feature budget.

The optimal allocation minimizes total expected error:
    E[error] = sum_i (n_i * Δ_i / (ε_i / n_i))
where n_i = block size, Δ_i = avg feature sensitivity, ε_i = block budget.
"""

import numpy as np
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveBudgetAllocator:
    """Adaptively allocates privacy budget across privacy blocks."""

    def __init__(
        self,
        method: str = 'optimal',
        blocks: Optional[List[Set[str]]] = None,
        bounds: Optional[Dict[str, tuple]] = None,
    ):
        """
        Initialize allocator.

        Args:
            method: Allocation method:
                - 'equal': Equal allocation per block
                - 'proportional': More ε to higher-sensitivity blocks
                - 'inverse_sensitivity': More ε to lower-sensitivity blocks
                - 'optimal': Minimize total expected error (RECOMMENDED)
            blocks: Privacy blocks (needed for 'optimal' method)
            bounds: Feature bounds (needed for 'optimal' method)
        """
        self.method = method
        self.blocks = blocks
        self.bounds = bounds
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
            n_blocks = len(sensitivities)
            epsilon_per_block = total_epsilon / n_blocks
            allocations = {i: epsilon_per_block for i in sensitivities.keys()}

        elif self.method == 'proportional':
            total_sensitivity = sum(sensitivities.values())
            if total_sensitivity == 0:
                n_blocks = len(sensitivities)
                allocations = {i: total_epsilon / n_blocks for i in sensitivities.keys()}
            else:
                allocations = {
                    i: total_epsilon * (s / total_sensitivity)
                    for i, s in sensitivities.items()
                }

        elif self.method == 'inverse_sensitivity':
            inv_s = {i: 1.0 / (s + 1e-10) for i, s in sensitivities.items()}
            total_inv = sum(inv_s.values())
            allocations = {i: total_epsilon * (v / total_inv) for i, v in inv_s.items()}

        elif self.method == 'optimal':
            # Optimal allocation that minimizes total expected squared error.
            #
            # For Laplace with sequential composition:
            #   Block i has n_i features, each with sensitivity Δ_ij.
            #   Block gets ε_i, so each feature gets ε_i / n_i.
            #   Noise scale for feature j = Δ_ij / (ε_i / n_i) = Δ_ij * n_i / ε_i.
            #   Expected error for block i ∝ sum_j Δ_ij * n_i / ε_i.
            #
            # To minimize total error subject to sum(ε_i) = ε_total,
            # optimal allocation is:  ε_i ∝ sqrt(n_i * S_i)
            # where S_i = sum of per-feature sensitivities in block i.
            #
            # This gives MORE budget to blocks that have:
            # - More features (n_i) — because ε gets split more ways
            # - Higher total sensitivity (S_i) — because more noise is needed
            allocations = self._optimal_allocation(sensitivities, total_epsilon)

        else:
            raise ValueError(f"Unknown allocation method: {self.method}")

        # Verify allocation sums correctly
        allocated_total = sum(allocations.values())
        if abs(allocated_total - total_epsilon) > 1e-6:
            logger.warning(
                f"Allocation sum ({allocated_total:.6f}) != total_epsilon ({total_epsilon:.6f}). "
                "Normalizing..."
            )
            scale = total_epsilon / allocated_total
            allocations = {i: eps * scale for i, eps in allocations.items()}

        self.allocations = allocations

        logger.info(
            f"Allocated ε={total_epsilon:.4f} across {len(allocations)} blocks "
            f"using {self.method} method"
        )
        for i, eps in sorted(allocations.items()):
            block_size = len(self.blocks[i]) if self.blocks and i < len(self.blocks) else '?'
            logger.debug(f"  Block {i} ({block_size} features): ε={eps:.6f} ({eps/total_epsilon*100:.1f}%)")

        return allocations

    def _optimal_allocation(
        self,
        sensitivities: Dict[int, float],
        total_epsilon: float
    ) -> Dict[int, float]:
        """
        Compute optimal allocation that minimizes total expected error.

        ε_i ∝ sqrt(n_i * S_i) where n_i = block size, S_i = block sensitivity.
        """
        weights = {}
        for i, s in sensitivities.items():
            if self.blocks and i < len(self.blocks):
                n_i = len(self.blocks[i])
            else:
                n_i = 1
            # Weight = sqrt(n_i * S_i) — proportional to noise contribution
            weights[i] = np.sqrt(max(n_i, 1) * max(s, 1e-10))

        total_weight = sum(weights.values())
        if total_weight == 0:
            n = len(sensitivities)
            return {i: total_epsilon / n for i in sensitivities.keys()}

        return {
            i: total_epsilon * (w / total_weight)
            for i, w in weights.items()
        }

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
    method: str = 'optimal',
    blocks: Optional[List[Set[str]]] = None,
    bounds: Optional[Dict[str, tuple]] = None,
) -> Dict[int, float]:
    """
    Convenience function for budget allocation.

    Args:
        sensitivities: Dict mapping block index to sensitivity
        total_epsilon: Total privacy budget
        method: Allocation method
        blocks: Privacy blocks (for optimal method)
        bounds: Feature bounds (for optimal method)

    Returns:
        Dict mapping block index to allocated ε
    """
    allocator = AdaptiveBudgetAllocator(method=method, blocks=blocks, bounds=bounds)
    return allocator.allocate(sensitivities, total_epsilon)
