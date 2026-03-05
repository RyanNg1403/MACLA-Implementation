import re
import json
import logging
from typing import Dict, List, Optional, Tuple

from .data_structures import Procedure, ContrastiveContext
from .memory import EnhancedHierarchicalMemorySystem
from .bayesian_selector import BayesianProcedureSelector
from .contrastive import ContrastiveRefinementEngine
from .meta_learner import MetaProceduralLearner
from .llm_reasoner import FrozenLLMReasoner

logger = logging.getLogger(__name__)


class EnhancedMACLAAgent:
    """MACLA Agent with domain-agnostic learning"""

    def __init__(self, N_a: int = 1000, N_s: int = 100, N_p: int = 200, N_m: int = 50):
        self.memory_system = EnhancedHierarchicalMemorySystem(N_a, N_s, N_p, N_m)
        self.bayesian_selector = BayesianProcedureSelector(self.memory_system)
        self.contrastive_refiner = ContrastiveRefinementEngine()
        self.meta_learner = MetaProceduralLearner(self.memory_system)

        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "procedures_learned": 0,
            "meta_procedures_learned": 0,
            "procedures_refined": 0,
        }

    # ======== DOMAIN-AGNOSTIC GOAL DISCOVERY ========
    def discover_goal_unsupervised(self, traj: Dict) -> str:
        """
        Pure unsupervised goal discovery with TravelPlanner handling.
        """
        task = traj.get("task", "").lower()

        if "travel" in task or "trip" in task or "plan" in task:
            from_match = re.search(r'(?:from|originating from|starting (?:from|in))\s+(\w+)', task)
            to_match = re.search(r'(?:to|heading (?:to|for)|destination[:\s]+)(\w+)', task)

            if from_match and to_match:
                from_city = from_match.group(1).lower()
                to_city = to_match.group(1).lower()
                from_city = re.sub(r'^(new|south|north|east|west|saint|st)$', '', from_city).strip()
                to_city = re.sub(r'^(new|south|north|east|west|saint|st)$', '', to_city).strip()
                if from_city and to_city:
                    return f"plan_trip_{from_city}_{to_city}"

            if to_match:
                city = to_match.group(1).lower()
                if city and city not in ['march', 'april', 'may', 'june']:
                    return f"plan_trip_to_{city}"

        actions = " ".join(traj.get("actions", [])[:3]).lower()
        if "plan_departure" in actions:
            return "plan_travel_trip"

        return "travel_planning_task"

    # -------- Learning --------
    def learn_from_trajectories(self, trajectories: List[Dict]) -> Dict[str, int]:
        logger.info(f"Learning from {len(trajectories)} trajectories")
        results = {"procedures_extracted": 0, "meta_procedures_extracted": 0, "procedures_refined": 0}
        trajectory_procedures: Dict[str, str] = {}

        for traj in trajectories:
            self._learn_atomic_patterns(traj)
            proc_id = self._extract_and_add_procedure_universal(traj)
            if proc_id:
                results["procedures_extracted"] += 1
                trajectory_procedures[traj.get("id", "")] = proc_id

                ctx = ContrastiveContext(
                    observation_init=traj.get("task", ""),
                    action_sequence=traj.get("actions", []),
                    observation_term=(traj.get("observations", [""]) or [""])[-1],
                    cumulative_reward=1.0 if traj.get("success", False) else 0.0,
                    trajectory_id=traj.get("id", "unknown"),
                    success=traj.get("success", False),
                )
                self.memory_system.record_execution_outcome(proc_id, traj.get("success", False), ctx)

        for key, entry in list(self.memory_system.procedural_memory.items()):
            if self.contrastive_refiner.should_refine(entry):
                self.contrastive_refiner.refine_procedure(entry)
                results["procedures_refined"] += 1
                self.memory_system.stats["procedures_refined"] += 1

        for traj in trajectories:
            if not traj.get("success", False):
                continue
            actions = traj.get("actions", [])
            if len(actions) >= 3:
                sequence = self._segment_into_procedures(traj, trajectory_procedures)
                if len(sequence) >= 3:
                    meta = self.meta_learner.extract_meta_procedure(traj, sequence)
                    if meta:
                        self.memory_system.add_meta_procedure(meta)
                        results["meta_procedures_extracted"] += 1

        logger.info(f"Learning results: {results}")
        return results

    def _learn_atomic_patterns(self, trajectory: Dict):
        traj_id = trajectory.get("id", "unknown")
        task = trajectory.get("task", "")
        for step in trajectory.get("trajectory_path", []):
            self.memory_system.add_atomic_entry(
                action=step.get("action", ""),
                observation=step.get("observation", ""),
                reward=1.0 if trajectory.get("success") else 0.0,
                context=task,
                trajectory_id=traj_id,
                step_index=step.get("step", 0),
            )

    def _extract_and_add_procedure_universal(self, trajectory: Dict) -> Optional[str]:
        """Domain-agnostic procedure extraction"""
        task = trajectory.get("task", "")
        actions = trajectory.get("actions", [])
        if len(actions) < 1:
            return None

        if hasattr(self, 'discover_goal_auto'):
            goal = self.discover_goal_auto(trajectory, mode="llm")
        else:
            goal = self.discover_goal_unsupervised(trajectory)

        generalized_steps = [self._generalize_action(a) for a in actions]
        is_success = trajectory.get("success", False)

        for k, e in self.memory_system.procedural_memory.items():
            if e.procedure.goal == goal and e.procedure.steps == generalized_steps:
                if is_success:
                    e.procedure.alpha += 1
                else:
                    e.procedure.beta += 1
                return k

        proc = Procedure(
            goal=goal,
            preconditions=[],
            steps=generalized_steps,
            postconditions=["task_completed"] if is_success else ["task_incomplete"],
            concepts=set(re.findall(r"\b\w+\b", task.lower())),
            alpha=2 if is_success else 1,
            beta=1 if is_success else 2,
            source_trajectory=trajectory.get("id", ""),
        )
        contexts = {task, goal}
        goals = {goal}

        pk = self.memory_system.add_procedural_entry(proc, contexts, goals, performance=1.0 if is_success else 0.0)
        self.stats["procedures_learned"] += 1
        return pk

    def _segment_into_procedures(self, trajectory: Dict, trajectory_procedures: Dict[str, str]) -> List[str]:
        actions = trajectory.get("actions", [])
        n = len(actions)
        if n < 3:
            tid = trajectory.get("id", "")
            return [trajectory_procedures.get(tid)] if tid in trajectory_procedures else []
        seg = n // 3
        ids = []
        base_goal = self.discover_goal_unsupervised(trajectory)
        for i in range(3):
            start = i * seg
            end = (i + 1) * seg if i < 2 else n
            steps = [self._generalize_action(a) for a in actions[start:end]]
            mini = Procedure(goal=f"{base_goal}_part{i+1}", preconditions=[], steps=steps, postconditions=[], alpha=2, beta=1)
            mid = self.memory_system.add_procedural_entry(mini, {mini.goal}, {mini.goal}, performance=1.0)
            ids.append(mid)
        return ids

    # ======== DOMAIN-AGNOSTIC EXECUTION ========
    def execute_task(self, observation: str, goal: str) -> Dict:
        self.stats["total_executions"] += 1
        result = {
            "observation": observation,
            "goal": goal,
            "selected_procedure": None,
            "action_sequence": [],
            "confidence": 0.0,
            "method": "fallback",
        }

        use_bayesian = getattr(self, '_ablation_config', {}).get('bayesian', True)

        if use_bayesian:
            pk, conf = self.bayesian_selector.select_procedure(observation, goal)
            if pk and conf > 0.1:
                entry = self.memory_system.procedural_memory[pk]
                result.update({
                    "selected_procedure": pk,
                    "confidence": conf,
                    "method": "bayesian_procedure",
                    "action_sequence": entry.procedure.steps,
                })
                return result

        result["action_sequence"] = self._generate_fallback_actions(goal)
        result["confidence"] = 0.5
        return result

    def provide_feedback(self, execution_result: Dict, actual_success: bool):
        pk = execution_result.get("selected_procedure")
        if pk and pk in self.memory_system.procedural_memory:
            ctx = ContrastiveContext(
                observation_init=execution_result.get("observation", ""),
                action_sequence=execution_result.get("action_sequence", []),
                observation_term="",
                cumulative_reward=1.0 if actual_success else 0.0,
                trajectory_id=execution_result.get("trajectory_id", "unknown"),
                success=actual_success,
            )
            self.memory_system.record_execution_outcome(pk, actual_success, ctx)
            if actual_success:
                self.stats["successful_executions"] += 1

    # ======== DOMAIN-AGNOSTIC UTILITIES ========
    def _generalize_action(self, action: str) -> str:
        """Generalize actions by replacing specific entities with slots"""
        return re.sub(r"\b(\w+)\s+\d+\b", r"<\1>", action)

    def _generate_fallback_actions(self, goal: str) -> List[str]:
        """
        Domain-agnostic fallback using abstract role slots.
        """
        goal_lower = goal.lower()

        if any(v in goal_lower for v in ['find', 'locate', 'search', 'get', 'retrieve']):
            return [
                "perceive <environment>",
                "identify <target_entity>",
                "plan <approach>",
                "execute <primary_action>",
                "verify <outcome>"
            ]
        elif any(v in goal_lower for v in ['put', 'place', 'move', 'transfer']):
            return [
                "acquire <object>",
                "navigate to <destination>",
                "place <object> at <destination>",
                "verify placement"
            ]
        elif any(v in goal_lower for v in ['select', 'choose', 'buy', 'purchase', 'filter']):
            return [
                "browse <options>",
                "evaluate <criteria>",
                "select <choice>",
                "confirm selection"
            ]
        elif any(v in goal_lower for v in ['query', 'select', 'join', 'insert', 'update']):
            return [
                "parse <query>",
                "identify <tables>",
                "execute <operation>",
                "return <results>"
            ]
        elif any(v in goal_lower for v in ['heat', 'warm', 'cook']):
            return [
                "acquire <object>",
                "activate <heating_device>",
                "place <object> in <heating_device>",
                "monitor <process>",
                "retrieve <object>"
            ]
        elif any(v in goal_lower for v in ['cool', 'chill', 'refrigerate']):
            return [
                "acquire <object>",
                "open <cooling_device>",
                "place <object> in <cooling_device>",
                "verify storage"
            ]
        elif any(v in goal_lower for v in ['clean', 'wash', 'rinse']):
            return [
                "acquire <object>",
                "activate <cleaning_device>",
                "apply <cleaning_agent>",
                "rinse <object>",
                "dry <object>"
            ]

        return [
            "perceive <environment>",
            "locate <target>",
            "navigate to <location>",
            "interact with <object>",
            "verify <outcome>"
        ]

    def get_statistics(self) -> Dict[str, int]:
        return {
            "procedural_memory_size": len(self.memory_system.procedural_memory),
            "meta_procedural_size": len(self.memory_system.meta_procedural_memory),
            "total_executions": self.stats["total_executions"],
            "successful_executions": self.stats["successful_executions"],
        }


class LLMMACLAAgent(EnhancedMACLAAgent):
    """LLM-enhanced agent using Ollama for advanced reasoning"""

    def __init__(self, N_a=1000, N_s=100, N_p=200, N_m=50, llm_model: str = "llama2", use_llm: bool = True):
        super().__init__(N_a, N_s, N_p, N_m)
        self.use_llm = use_llm
        self.llm = FrozenLLMReasoner(llm_model) if use_llm else None
        self.llm_calls = {"segmentation": 0, "extraction": 0, "contrastive": 0, "goal_discovery": 0}

    # ======== LLM-ENHANCED GOAL DISCOVERY ========
    def discover_goal_auto(self, traj: Dict, mode: str = "llm") -> str:
        """
        Domain-agnostic goal discovery with LLM-first approach.
        """
        task = traj.get("task", "")
        actions = " | ".join(traj.get("actions", [])[:5])
        obs_init = (traj.get("observations") or [""])[0]
        obs_final = (traj.get("observations") or [""])[-1]

        if mode == "llm" and self.use_llm and self.llm:
            self.llm_calls["goal_discovery"] += 1
            raw = self.llm.discover_goal(task, actions, obs_init, obs_final)
            obj = self._safe_parse_json_dict(raw)
            goal = obj.get("goal", "").strip()
            if goal:
                logger.debug(f"LLM discovered goal: {goal}")
                return goal

        return self.discover_goal_unsupervised(traj)

    def _extract_and_add_procedure_universal(self, trajectory: Dict) -> Optional[str]:
        """Domain-agnostic procedure extraction with LLM goal discovery"""
        task = trajectory.get("task", "")
        actions = trajectory.get("actions", [])
        if len(actions) < 1:
            return None

        goal = self.discover_goal_auto(trajectory, mode="llm")
        generalized_steps = [self._generalize_action(a) for a in actions]
        is_success = trajectory.get("success", False)

        proc = Procedure(
            goal=goal,
            preconditions=[],
            steps=generalized_steps,
            postconditions=["task_completed"] if is_success else ["task_incomplete"],
            concepts=set(re.findall(r"\b\w+\b", task.lower())),
            alpha=2 if is_success else 1,
            beta=1 if is_success else 2,
            source_trajectory=trajectory.get("id", ""),
        )
        contexts = {task, goal}
        goals = {goal}

        for k, e in self.memory_system.procedural_memory.items():
            if e.procedure.steps == generalized_steps:
                return k

        pk = self.memory_system.add_procedural_entry(proc, contexts, goals, performance=1.0 if is_success else 0.0)
        self.stats["procedures_learned"] += 1
        return pk

    def segment_trajectory_llm_first(self, trajectory: Dict) -> List[Dict]:
        if self.use_llm and self.llm:
            self.llm_calls["segmentation"] += 1
            raw = self.llm.segment_trajectory(trajectory)
            steps = self._safe_parse_json_list(raw)
            if steps:
                return steps
        actions = trajectory.get("actions", [])
        return [{"step": i + 1, "action": a, "observation": ""} for i, a in enumerate(actions)]

    def extract_procedure_components_llm_first(self, segment: Dict) -> Dict:
        if self.use_llm and self.llm:
            self.llm_calls["extraction"] += 1
            raw = self.llm.extract_procedure_components(segment)
            obj = self._safe_parse_json_dict(raw)
            if obj and all(k in obj for k in ("goal", "preconditions", "steps", "postconditions")):
                return obj
        step = segment.get("action", segment.get("step", ""))
        return {"goal": "general_manipulation", "preconditions": [], "steps": [str(step)], "postconditions": []}

    def contrastive_analysis_llm_first(self, success_ctx: List[str], failure_ctx: List[str]) -> Dict:
        if self.use_llm and self.llm:
            self.llm_calls["contrastive"] += 1
            raw = self.llm.contrastive_analysis(success_ctx, failure_ctx)
            obj = self._safe_parse_json_dict(raw)
            if obj:
                return {
                    "preconditions_positive": obj.get("preconditions_positive", []),
                    "preconditions_negative": obj.get("preconditions_negative", []),
                    "postconditions": obj.get("postconditions", []),
                }
        s = set(" ".join(success_ctx).lower().split())
        f = set(" ".join(failure_ctx).lower().split())
        return {
            "preconditions_positive": list(s - f)[:3],
            "preconditions_negative": list(f - s)[:3],
            "postconditions": [],
        }

    @staticmethod
    def _safe_parse_json_list(text: str) -> List[Dict]:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    @staticmethod
    def _safe_parse_json_dict(text: str) -> Dict:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def learn_from_trajectories_ablation(
        self,
        trajectories: List[Dict],
        use_contrastive: bool = True,
        use_meta: bool = True
    ) -> Dict[str, int]:
        """
        Learning with ablation control flags.
        """
        logger.info(f"Learning from {len(trajectories)} trajectories (ablation mode)")
        results = {
            "procedures_extracted": 0,
            "meta_procedures_extracted": 0,
            "procedures_refined": 0
        }
        trajectory_procedures: Dict[str, str] = {}

        for traj in trajectories:
            self._learn_atomic_patterns(traj)
            proc_id = self._extract_and_add_procedure_universal(traj)
            if proc_id:
                results["procedures_extracted"] += 1
                trajectory_procedures[traj.get("id", "")] = proc_id

                ctx = ContrastiveContext(
                    observation_init=traj.get("task", ""),
                    action_sequence=traj.get("actions", []),
                    observation_term=(traj.get("observations", [""]) or [""])[-1],
                    cumulative_reward=1.0 if traj.get("success", False) else 0.0,
                    trajectory_id=traj.get("id", "unknown"),
                    success=traj.get("success", False),
                )
                self.memory_system.record_execution_outcome(proc_id, traj.get("success", False), ctx)

        if use_contrastive:
            for key, entry in list(self.memory_system.procedural_memory.items()):
                if self.contrastive_refiner.should_refine(entry):
                    self.contrastive_refiner.refine_procedure(entry)
                    results["procedures_refined"] += 1
        else:
            logger.info("Skipping contrastive refinement (ablation)")

        if use_meta:
            for traj in trajectories:
                if not traj.get("success", False):
                    continue
                actions = traj.get("actions", [])
                if len(actions) >= 3:
                    sequence = self._segment_into_procedures(traj, trajectory_procedures)
                    if len(sequence) >= 3:
                        meta = self.meta_learner.extract_meta_procedure(traj, sequence)
                        if meta:
                            self.memory_system.add_meta_procedure(meta)
                            results["meta_procedures_extracted"] += 1
        else:
            logger.info("Skipping meta-procedure extraction (ablation)")

        logger.info(f"Learning results: {results}")
        return results

    def configure_ablation(self,
                           use_bayesian: bool = True,
                           use_contrastive: bool = True,
                           use_meta: bool = True,
                           use_ontology: bool = True):
        """
        Configure which components are active for ablation studies.
        """
        self._ablation_config = {
            "bayesian": use_bayesian,
            "contrastive": use_contrastive,
            "meta": use_meta,
            "ontology": use_ontology
        }

        if not use_ontology:
            self.bayesian_selector.ontology = {}
            self.bayesian_selector.ontology_embeddings = {}

        logger.info(f"Ablation configured: Bayesian={use_bayesian}, "
                    f"Contrastive={use_contrastive}, Meta={use_meta}, Ontology={use_ontology}")
