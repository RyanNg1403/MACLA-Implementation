import random
import logging
from typing import Dict, List, Tuple

from .data_structures import PerformanceMetrics
from .agent import LLMMACLAAgent
from .evaluator import MACLAEvaluator

logger = logging.getLogger(__name__)


def train_test_split_trajectories(trajectories: List[Dict], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    indices = list(range(len(trajectories)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    train = [trajectories[i] for i in train_idx]
    test = [trajectories[i] for i in test_idx]
    return train, test


def build_agent_and_learn(train_trajectories: List[Dict], llm_model: str = "llama2", use_llm: bool = True) -> LLMMACLAAgent:
    agent = LLMMACLAAgent(llm_model=llm_model, use_llm=use_llm)
    agent.bayesian_selector.build_ontology(train_trajectories)
    agent.learn_from_trajectories(train_trajectories)
    return agent


def run_evaluation(agent: LLMMACLAAgent, test_trajectories: List[Dict]) -> PerformanceMetrics:
    evaluator = MACLAEvaluator()
    test_tasks = [{"task": t.get("task", "")} for t in test_trajectories]
    metrics = evaluator.evaluate_comprehensive(agent, test_trajectories, test_tasks)
    return metrics


def pretty_print_metrics(m: PerformanceMetrics):
    logger.info("=== Evaluation Metrics ===")
    logger.info(f"Avg Reward:           {m.avg_reward:.3f}")
    logger.info(f"Success Rate:         {m.success_rate:.3f}  ({m.num_successful}/{m.num_total})")
    logger.info(f"Accuracy / P / R / F1: {m.accuracy:.3f} / {m.precision:.3f} / {m.recall:.3f} / {m.f1_score:.3f}")
    logger.info(f"Mean Posterior Entropy: {m.mean_posterior_entropy:.3f}")
    logger.info(f"Calibration (proxy):    {m.calibration_score:.3f}")
    logger.info(f"Uncertainty Reduction:  {m.uncertainty_reduction:.3f}")
    logger.info(f"Meta-Usage / Composition: {m.meta_procedure_usage_rate:.3f} / {m.composition_success_rate:.3f}")
    logger.info(f"Procedures Refined / Improvement: {m.procedures_refined} / {m.refinement_improvement:.3f}")
    logger.info(f"Avg Inference Time (per task): {m.avg_inference_time:.3f}s")
    logger.info(f"Memory Utilization: {m.memory_utilization:.3f}")
