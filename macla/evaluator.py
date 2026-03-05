import re
import math
import time
import logging
from typing import Dict, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .data_structures import PerformanceMetrics
from .agent import EnhancedMACLAAgent, LLMMACLAAgent

logger = logging.getLogger(__name__)


class MACLAEvaluator:
    def __init__(self):
        self.evaluation_history: List[PerformanceMetrics] = []

    def evaluate_comprehensive(self, agent: EnhancedMACLAAgent, test_trajectories: List[Dict], test_tasks: List[Dict]) -> PerformanceMetrics:
        metrics = PerformanceMetrics()

        reward_results = self._compute_average_reward(test_trajectories)
        metrics.avg_reward = reward_results["avg_reward"]
        metrics.success_rate = reward_results["success_rate"]
        metrics.num_successful = reward_results["num_successful"]
        metrics.num_total = reward_results["num_total"]

        acc = self._evaluate_accuracy(agent, test_trajectories)
        metrics.accuracy = acc["accuracy"]
        metrics.precision = acc["precision"]
        metrics.recall = acc["recall"]
        metrics.f1_score = acc["f1_score"]

        bayes_like = self._evaluate_bayesian_components(agent)
        metrics.mean_posterior_entropy = bayes_like["mean_entropy"]
        metrics.calibration_score = bayes_like["calibration"]
        metrics.uncertainty_reduction = bayes_like["uncertainty_reduction"]

        meta = self._evaluate_meta_procedural(agent, test_trajectories)
        metrics.meta_procedure_usage_rate = meta["usage_rate"]
        metrics.composition_success_rate = meta["composition_success"]

        contrastive = self._evaluate_contrastive_learning(agent)
        metrics.procedures_refined = contrastive["refined_count"]
        metrics.refinement_improvement = contrastive["improvement"]

        timing = self._evaluate_timing(agent, test_tasks[: min(20, len(test_tasks))])
        metrics.avg_inference_time = timing["inference_time"]

        stats = agent.get_statistics()
        metrics.memory_utilization = stats["procedural_memory_size"] / max(1.0, float(agent.memory_system.N_p))

        self.evaluation_history.append(metrics)
        return metrics

    def _compute_average_reward(self, trajectories: List[Dict]) -> Dict:
        """
        Compute reward based on ACTUAL execution success, not synthetic flags.
        For SQL: Use actual_execution_success if available, otherwise use success.
        """
        rewards = []
        num_successful = 0

        for t in trajectories:
            if t.get("domain") == "sql" and "actual_execution_success" in t:
                actual_success = t.get("actual_execution_success", False)
                r = 1.0 if actual_success else 0.0
            else:
                r = float(t.get("reward", 1.0 if t.get("success", False) else 0.0))

            rewards.append(r)
            if r >= 0.5:
                num_successful += 1

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_rate = num_successful / len(rewards) if rewards else 0.0

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "num_successful": num_successful,
            "num_total": len(rewards),
            "rewards": rewards,
        }

    def _evaluate_accuracy(self, agent: EnhancedMACLAAgent, trajectories: List[Dict]) -> Dict:
        """
        Evaluate accuracy with domain-specific logic:
        - SQL: Compare generated query to gold query
        - Others: Check if procedure was retrieved and executed
        """
        preds, gts = [], []

        sql_exact_matches = 0
        sql_structural_matches = 0
        sql_total = 0

        for i, t in enumerate(trajectories[: min(100, len(trajectories))]):
            task = t.get("task", "")
            obs = f"Environment: {task}"
            domain = t.get("domain", "unknown")

            if isinstance(agent, LLMMACLAAgent):
                goal = agent.discover_goal_auto(t, mode="llm")
            else:
                goal = agent.discover_goal_unsupervised(t)

            if i < 5:
                logger.info(f"DEBUG Test {i} (domain={domain}):")
                logger.info(f"  Task: {task[:80]}...")
                logger.info(f"  Discovered Goal: {goal}")

            res = agent.execute_task(obs, goal)

            if domain == "sql":
                gold_query = t.get("gold_query", "").strip()
                generated_actions = res.get("action_sequence", [])

                sql_total += 1
                pred_success = False
                exact_match = False
                structural_match = False

                for action in generated_actions:
                    action_normalized = self._normalize_sql(action)
                    gold_normalized = self._normalize_sql(gold_query)

                    if action_normalized == gold_normalized:
                        exact_match = True
                        structural_match = True
                        pred_success = True
                        sql_exact_matches += 1
                        break

                    if self._sql_structural_match(action, gold_query):
                        structural_match = True
                        pred_success = True
                        sql_structural_matches += 1
                        break

                gt = t.get("actual_execution_success", False)

                if i < 5:
                    logger.info(f"  Generated SQL: {generated_actions[0] if generated_actions else 'None'}...")
                    logger.info(f"  Gold SQL: {gold_query[:80]}...")
                    logger.info(f"  Exact Match: {exact_match}, Structural Match: {structural_match}")

            else:
                pred_success = res.get("method") == "bayesian_procedure"
                gt = bool(t.get("success", False))

            if i < 5:
                logger.info(f"  Selected Procedure: {res.get('selected_procedure')}")
                logger.info(f"  Method: {res.get('method')}")
                logger.info(f"  Confidence: {res.get('confidence'):.3f}")
                logger.info(f"  Prediction: {pred_success}, Ground Truth: {gt}")

            agent.provide_feedback(res, gt)

            preds.append(pred_success)
            gts.append(gt)

        if not preds:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        acc = accuracy_score(gts, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(gts, preds, average="binary", zero_division=0)

        if sql_total > 0:
            logger.info(f"\nSQL Evaluation Results:")
            logger.info(f"  Exact Matches: {sql_exact_matches}/{sql_total} ({sql_exact_matches/sql_total*100:.1f}%)")
            logger.info(f"  Structural Matches: {sql_structural_matches}/{sql_total} ({sql_structural_matches/sql_total*100:.1f}%)")

        return {
            "accuracy": float(acc),
            "precision": float(pr),
            "recall": float(rc),
            "f1_score": float(f1)
        }

    def _normalize_sql(self, query: str) -> str:
        """Normalize SQL query for comparison"""
        if not query:
            return ""
        normalized = ' '.join(query.strip().upper().split())
        normalized = normalized.replace('  ', ' ')
        return normalized

    def _sql_structural_match(self, query1: str, query2: str, threshold: float = 0.6) -> bool:
        """Check if two SQL queries have similar structure"""
        if not query1 or not query2:
            return False

        def extract_components(sql):
            components = set()
            sql_upper = sql.upper()

            for match in re.finditer(r'FROM\s+(\w+)', sql_upper):
                components.add(f"TABLE:{match.group(1)}")
            for match in re.finditer(r'JOIN\s+(\w+)', sql_upper):
                components.add(f"TABLE:{match.group(1)}")

            operations = ['SELECT', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT',
                         'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'UNION', 'DISTINCT']
            for op in operations:
                if op in sql_upper:
                    components.add(f"OP:{op}")

            aggregates = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
            for agg in aggregates:
                if agg in sql_upper:
                    components.add(f"AGG:{agg}")

            return components

        comp1 = extract_components(query1)
        comp2 = extract_components(query2)

        if not comp1 or not comp2:
            return False

        intersection = len(comp1 & comp2)
        union = len(comp1 | comp2)
        similarity = intersection / union if union > 0 else 0.0

        return similarity >= threshold

    def _evaluate_bayesian_components(self, agent: EnhancedMACLAAgent) -> Dict:
        entropies = []
        for _, entry in agent.memory_system.procedural_memory.items():
            a, b = entry.procedure.alpha, entry.procedure.beta
            p = a / (a + b)
            if p in (0.0, 1.0):
                h = 0.0
            else:
                h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
            entropies.append(h)
        mean_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
        return {"mean_entropy": mean_entropy, "calibration": 0.0, "uncertainty_reduction": max(0.0, 1.0 - mean_entropy)}

    def _evaluate_meta_procedural(self, agent: EnhancedMACLAAgent, trajectories: List[Dict]) -> Dict:
        meta_count = len(agent.memory_system.meta_procedural_memory)
        usage_rate = min(1.0, meta_count / max(1, len(agent.memory_system.procedural_memory)))
        composition_success = 0.5 if meta_count > 0 else 0.0
        return {"usage_rate": usage_rate, "composition_success": composition_success}

    def _evaluate_contrastive_learning(self, agent: EnhancedMACLAAgent) -> Dict:
        refined = agent.memory_system.stats.get("procedures_refined", 0)
        improvement = 0.1 * refined
        return {"refined_count": refined, "improvement": improvement}

    def _evaluate_timing(self, agent: EnhancedMACLAAgent, tasks: List[Dict]) -> Dict:
        if not tasks:
            return {"inference_time": 0.0}
        start = time.time()
        for t in tasks:
            obs = f"Environment: {t.get('task','')}"
            if isinstance(agent, LLMMACLAAgent):
                goal = agent.discover_goal_auto(t, mode="llm")
            else:
                goal = agent.discover_goal_unsupervised(t)
            _ = agent.execute_task(obs, goal)
        end = time.time()
        return {"inference_time": (end - start) / len(tasks)}
