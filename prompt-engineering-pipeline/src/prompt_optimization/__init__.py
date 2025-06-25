"""
Prompt Optimization Module
Implements automated prompt optimization using meta-prompting and feedback loops
"""

from .optimizer import PromptOptimizer, PromptVariant, OptimizationResult
from .prompt_manager import PromptManager
from .optimization_runner import OptimizationRunner

__all__ = [
    'PromptOptimizer',
    'PromptVariant',
    'OptimizationResult',
    'PromptManager',
    'OptimizationRunner'
]

__version__ = "1.0.0"
