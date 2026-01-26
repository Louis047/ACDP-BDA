"""
Runtime and memory profiling for scalability evaluation.
"""

import time
import tracemalloc
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def profile_function(
    func: Callable,
    *args,
    **kwargs
) -> Dict:
    """
    Profile a function's runtime and memory usage.
    
    Args:
        func: Function to profile
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Dict with 'runtime' (seconds) and 'memory_peak' (MB)
    """
    # Start memory tracking
    tracemalloc.start()
    
    # Time execution
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
    finally:
        end_time = time.time()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        runtime = end_time - start_time
        memory_peak_mb = peak / 1024 / 1024
    
    return {
        'runtime': runtime,
        'memory_peak_mb': memory_peak_mb,
        'result': result
    }


class ScalabilityProfiler:
    """Profiles scalability across different dataset sizes."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict] = {}
    
    def profile(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict:
        """
        Profile a function and store results.
        
        Args:
            name: Name for this profile
            func: Function to profile
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Profile dict
        """
        profile = profile_function(func, *args, **kwargs)
        self.profiles[name] = profile
        return profile
    
    def get_summary(self) -> Dict:
        """Get summary of all profiles."""
        return {
            'n_profiles': len(self.profiles),
            'profiles': self.profiles,
            'total_runtime': sum(p['runtime'] for p in self.profiles.values()),
            'max_memory': max(p['memory_peak_mb'] for p in self.profiles.values()) if self.profiles else 0.0
        }
    
    def print_summary(self) -> None:
        """Print profiling summary."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("SCALABILITY PROFILING SUMMARY")
        print("=" * 60)
        print(f"Number of profiles: {summary['n_profiles']}")
        print(f"Total runtime: {summary['total_runtime']:.2f} seconds")
        print(f"Peak memory: {summary['max_memory']:.2f} MB")
        print("\nIndividual profiles:")
        for name, profile in summary['profiles'].items():
            print(f"  {name}:")
            print(f"    Runtime: {profile['runtime']:.2f} seconds")
            print(f"    Peak memory: {profile['memory_peak_mb']:.2f} MB")
        print("=" * 60)
