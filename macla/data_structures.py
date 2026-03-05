import time
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class AtomicMemoryEntry:
    action: str
    observation: str
    reward: float
    context: str
    trajectory_id: str = ""
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Procedure:
    goal: str
    preconditions: List[str]
    steps: List[str]
    postconditions: List[str]
    concepts: Set[str] = field(default_factory=set)
    alpha: int = 1
    beta: int = 1
    execution_count: int = 0
    generalizability_score: float = 0.5
    confidence: float = 0.5
    source_trajectory: str = ""

    @property
    def success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def success_variance(self) -> float:
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n**2 * (n + 1))


@dataclass
class MetaProcedure:
    goal_meta: str
    preconditions_meta: List[str]
    sub_procedures: List[str]
    composition_policy: Dict[str, any]
    alpha: int = 1
    beta: int = 1
    execution_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class ContrastiveContext:
    observation_init: str
    action_sequence: List[str]
    observation_term: str
    cumulative_reward: float
    trajectory_id: str
    success: bool


@dataclass
class ProceduralMemoryEntry:
    procedure: Procedure
    success_contexts: List[ContrastiveContext] = field(default_factory=list)
    failure_contexts: List[ContrastiveContext] = field(default_factory=list)
    discriminative_patterns: Dict[str, List[str]] = field(default_factory=dict)
    contexts: Set[str] = field(default_factory=set)
    goals: Set[str] = field(default_factory=set)
    performance_score: float = 0.5
    last_refined: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    avg_reward: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    success_rate: float = 0.0
    num_successful: int = 0
    num_total: int = 0
    mean_posterior_entropy: float = 0.0
    calibration_score: float = 0.0
    uncertainty_reduction: float = 0.0
    meta_procedure_usage_rate: float = 0.0
    composition_success_rate: float = 0.0
    procedures_refined: int = 0
    refinement_improvement: float = 0.0
    avg_inference_time: float = 0.0
    memory_utilization: float = 0.0
