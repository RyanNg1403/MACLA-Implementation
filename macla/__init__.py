"""
MACLA - Memory-Augmented Contrastive Learning Agent
Domain-agnostic agent package for ALFWorld, WebShop, TravelPlanner, and IC-SQL.
"""

from .data_structures import (
    AtomicMemoryEntry,
    Procedure,
    MetaProcedure,
    ContrastiveContext,
    ProceduralMemoryEntry,
    PerformanceMetrics,
)
from .memory import EnhancedHierarchicalMemorySystem
from .bayesian_selector import BayesianProcedureSelector
from .contrastive import ContrastiveRefinementEngine
from .meta_learner import MetaProceduralLearner
from .llm_reasoner import FrozenLLMReasoner
from .agent import EnhancedMACLAAgent, LLMMACLAAgent
from .loaders import ALFWorldLikeLoader, WebShopLoader, TravelPlannerLoader, SQLLoader
from .evaluator import MACLAEvaluator
from .utils import (
    train_test_split_trajectories,
    build_agent_and_learn,
    run_evaluation,
    pretty_print_metrics,
)

__all__ = [
    "AtomicMemoryEntry",
    "Procedure",
    "MetaProcedure",
    "ContrastiveContext",
    "ProceduralMemoryEntry",
    "PerformanceMetrics",
    "EnhancedHierarchicalMemorySystem",
    "BayesianProcedureSelector",
    "ContrastiveRefinementEngine",
    "MetaProceduralLearner",
    "FrozenLLMReasoner",
    "EnhancedMACLAAgent",
    "LLMMACLAAgent",
    "ALFWorldLikeLoader",
    "WebShopLoader",
    "TravelPlannerLoader",
    "SQLLoader",
    "MACLAEvaluator",
    "train_test_split_trajectories",
    "build_agent_and_learn",
    "run_evaluation",
    "pretty_print_metrics",
]
