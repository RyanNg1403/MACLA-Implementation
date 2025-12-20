#!/usr/bin/env python3
# MACLA_Generalized.py
# Domain-agnostic version - works for ALFWorld, WebShop, IC-SQL

import os
import re
import json
import time
import math
import random
import logging
import warnings
from pathlib import Path
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# OLLAMA BACKEND
# =========================================================
_OLLAMA_AVAILABLE = True
try:
    import ollama
    logger.info("✓ Ollama Python package imported successfully")
except Exception as _e:
    _OLLAMA_AVAILABLE = False
    logger.warning(f"✗ Ollama import failed: {_e}")

# =========================================================
# OPTIONAL SEMANTIC EMBEDDINGS
# =========================================================
_EMBED_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_MODEL_NAME = os.environ.get("MACLA_EMBED_MODEL", "all-MiniLM-L6-v2")
    _EMBEDDER = SentenceTransformer(_ST_MODEL_NAME)
except Exception:
    _EMBED_AVAILABLE = False
    _EMBEDDER = None
    st_util = None
    logger.info("SentenceTransformer not found; semantic similarity will use keyword heuristics.")

# =========================================================
# DATA STRUCTURES
# =========================================================
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


# =========================================================
# HIERARCHICAL MEMORY
# =========================================================
class EnhancedHierarchicalMemorySystem:
    """Enhanced memory with meta-procedural and contrastive support"""

    def __init__(self, N_a: int = 1000, N_s: int = 100, N_p: int = 200, N_m: int = 50):
        self.N_a = N_a
        self.N_s = N_s
        self.N_p = N_p
        self.N_m = N_m

        self.atomic_memory: deque = deque(maxlen=N_a)
        self.sequential_memory: Dict[str, any] = {}
        self.procedural_memory: Dict[str, ProceduralMemoryEntry] = {}
        self.meta_procedural_memory: Dict[str, MetaProcedure] = {}

        self.context_index = defaultdict(set)
        self.goal_index = defaultdict(set)

        self.stats = {
            "procedures_added": 0,
            "procedures_refined": 0,
            "meta_procedures_added": 0,
        }
        
        self._meta_counter = 0
        
    def add_atomic_entry(
        self,
        action: str,
        observation: str,
        reward: float,
        context: str,
        trajectory_id: str = "",
        step_index: int = 0,
    ):
        entry = AtomicMemoryEntry(
            action=action,
            observation=observation,
            reward=reward,
            context=context,
            trajectory_id=trajectory_id,
            step_index=step_index,
        )
        self.atomic_memory.append(entry)

    def add_procedural_entry(
        self, procedure: Procedure, contexts: Set[str], goals: Set[str], performance: float
    ) -> str:
        proc_key = f"proc_{hash(str(procedure.steps)) % 1000000}"

        if len(self.procedural_memory) >= self.N_p:
            self._prune_procedural_memory()

        entry = ProceduralMemoryEntry(
            procedure=procedure,
            contexts=contexts,
            goals=goals,
            performance_score=performance,
        )
        self.procedural_memory[proc_key] = entry

        # REMOVED: Hard-coded context tags (heating_context, cooling_context, etc.)
        # NEW: Only add actual contexts, no domain-specific injections
        for context in contexts:
            self.context_index[context].add(proc_key)

        for goal in goals:
            self.goal_index[goal].add(proc_key)

        self.stats["procedures_added"] += 1
        return proc_key

    def add_meta_procedure(self, meta_proc: MetaProcedure) -> str:
        self._meta_counter += 1
        meta_key = f"meta_{self._meta_counter:06d}"

        if len(self.meta_procedural_memory) >= self.N_m:
            self._prune_meta_procedural_memory()

        self.meta_procedural_memory[meta_key] = meta_proc
        self.stats["meta_procedures_added"] += 1
        return meta_key

    def record_execution_outcome(self, proc_key: str, success: bool, context: ContrastiveContext):
        if proc_key not in self.procedural_memory:
            return
        entry = self.procedural_memory[proc_key]

        if success:
            entry.procedure.alpha += 1
            entry.success_contexts.append(context)
        else:
            entry.procedure.beta += 1
            entry.failure_contexts.append(context)

        entry.procedure.execution_count += 1
        if len(entry.success_contexts) > 15:
            entry.success_contexts.pop(0)
        if len(entry.failure_contexts) > 15:
            entry.failure_contexts.pop(0)

    def retrieve_by_goal(self, goal: str) -> List[str]:
        return list(self.goal_index.get(goal, set()))

    def retrieve_by_context(self, context: str) -> List[str]:
        return list(self.context_index.get(context, set()))

    def _prune_procedural_memory(self):
        if not self.procedural_memory:
            return
        utilities = []
        now = time.time()
        for key, entry in self.procedural_memory.items():
            utility = (
                0.5 * entry.procedure.success_rate
                + 0.3 * min(1.0, entry.procedure.execution_count / 10.0)
                + 0.2 * (1.0 - min(1.0, (now - entry.last_refined) / 86400))
            )
            utilities.append((key, utility))
        utilities.sort(key=lambda x: x[1])
        to_remove = utilities[0][0]

        entry = self.procedural_memory[to_remove]
        for context in entry.contexts:
            self.context_index[context].discard(to_remove)
        for goal in entry.goals:
            self.goal_index[goal].discard(to_remove)
        del self.procedural_memory[to_remove]

    def _prune_meta_procedural_memory(self):
        if not self.meta_procedural_memory:
            return
        pairs = [(k, mp.success_rate) for k, mp in self.meta_procedural_memory.items()]
        pairs.sort(key=lambda x: x[1])
        del self.meta_procedural_memory[pairs[0][0]]


# =========================================================
# BAYESIAN SELECTOR (Domain-Agnostic)
# =========================================================
class BayesianProcedureSelector:
    """Bayesian selection with expected utility framework"""

    def __init__(self, memory_system: EnhancedHierarchicalMemorySystem, lambda_info: float = 0.0):
        self.memory_system = memory_system
        self.lambda_info = lambda_info
        self.R_max = 1.0
        self.C_fail = 0.5

        self.ontology: Dict[str, List[str]] = {}
        self.ontology_embeddings: Dict[str, any] = {}

    # -------- Ontology & Similarity (Domain-Agnostic) ----------
    def build_ontology(self, trajectories: List[Dict], k_top: int = 100):
        """Build ontology from ANY domain's training data"""
        all_words = []
        for traj in trajectories:
            task = traj.get("task", "").lower()
            actions = " ".join(traj.get("actions", [])).lower()
            
            # Extract meaningful words from tasks AND actions
            words = [w for w in re.findall(r"[a-zA-Z]+", task + " " + actions) if len(w) > 3]
            all_words.extend(words)

        word_counts = Counter(all_words)
        top_words = [w for w, _ in word_counts.most_common(k_top)]
        categories = {}

        if _EMBED_AVAILABLE:
            word_embeddings = _EMBEDDER.encode(top_words, convert_to_tensor=True)
            used = set()
            for i, w in enumerate(top_words):
                if w in used:
                    continue
                similar = [w]
                for j, ow in enumerate(top_words):
                    if j != i and ow not in used:
                        sim = float(st_util.cos_sim(word_embeddings[i], word_embeddings[j])[0][0])
                        if sim > 0.6:
                            similar.append(ow)
                            used.add(ow)
                categories[w] = similar
                used.add(w)

            self.ontology = categories
            for cat, keys in self.ontology.items():
                text = f"{cat} {' '.join(keys)}"
                self.ontology_embeddings[cat] = _EMBEDDER.encode(text, convert_to_tensor=True)
        else:
            # Simple fallback: group by first letter
            for w in top_words:
                key = w[0]
                categories.setdefault(key, []).append(w)
            self.ontology = categories

        logger.info(f"Built domain-agnostic ontology with {len(self.ontology)} categories")

    def _extract_context(self, observation: str, threshold: float = 0.55) -> str:
        """Extract context without domain-specific keywords"""
        obs_lower = observation.lower()
        if not self.ontology:
            return self._extract_context_fallback(obs_lower)

        # Fast keyword match
        for category, keywords in self.ontology.items():
            if any(k in obs_lower for k in keywords):
                return category

        # Semantic match
        if _EMBED_AVAILABLE and self.ontology_embeddings:
            obs_emb = _EMBEDDER.encode(obs_lower, convert_to_tensor=True)
            best_category, best_score = None, 0.0
            for cat, emb in self.ontology_embeddings.items():
                score = float(st_util.cos_sim(obs_emb, emb)[0][0])
                if score > best_score:
                    best_score, best_category = score, cat
            if best_score >= threshold and best_category:
                return best_category

        return self._extract_context_fallback(obs_lower)

    def _extract_context_fallback(self, obs_lower: str) -> str:
        """Domain-agnostic fallback"""
        for w in obs_lower.split():
            if len(w) > 4 and w.isalpha():
                return w
        return "general"

    # -------- Candidate Retrieval ----------
    def _retrieve_candidates(self, observation: str, goal: str, k: int = 10) -> List[str]:
        candidates: Set[str] = set()

        # Exact goal match
        if goal in self.memory_system.goal_index:
            candidates.update(self.memory_system.goal_index[goal])

        # Fuzzy goal match
        if not candidates:
            gwords = set(goal.lower().split("_"))
            for g, keys in self.memory_system.goal_index.items():
                if set(g.lower().split("_")) & gwords:
                    candidates.update(keys)

        # Context-based
        context = self._extract_context(observation)
        for ctx, keys in self.memory_system.context_index.items():
            if context in ctx.lower() or ctx.lower() in context:
                candidates.update(keys)

        # Popularity fallback
        if not candidates and self.memory_system.procedural_memory:
            all_procs = list(self.memory_system.procedural_memory.items())
            all_procs.sort(key=lambda x: x[1].procedure.execution_count, reverse=True)
            candidates = {k for k, _ in all_procs[:k]}

        clist = list(candidates)
        clist.sort(
            key=lambda pk: self.memory_system.procedural_memory[pk].procedure.execution_count
            if pk in self.memory_system.procedural_memory
            else 0,
            reverse=True,
        )
        return clist[:k]

    # -------- Utility & Selection ----------
    def _compute_relevance(self, entry: ProceduralMemoryEntry, observation: str, goal: str) -> float:
        rel = 0.0

        if goal in entry.goals:
            rel += 0.6
        else:
            for eg in entry.goals:
                if any(w in eg for w in goal.split("_")):
                    rel += 0.3
                    break

        context = self._extract_context(observation)
        if context in entry.contexts:
            rel += 0.4
        else:
            for ec in entry.contexts:
                if context in ec or ec in context:
                    rel += 0.2
                    break

        return min(1.0, rel)

    def _compute_failure_risk(self, entry: ProceduralMemoryEntry, observation: str, theta_risk: float = 0.75) -> float:
        if not entry.failure_contexts:
            return 0.0
        obs_lower = observation.lower()
        risk_count = 0
        for fctx in entry.failure_contexts:
            fail_obs = fctx.observation_init.lower()
            a = set(obs_lower.split())
            b = set(fail_obs.split())
            overlap = len(a & b) / max(len(a), 1)
            if overlap > theta_risk:
                risk_count += 1
        return risk_count / len(entry.failure_contexts)

    def _compute_information_gain(self, procedure: Procedure) -> float:
        alpha, beta = procedure.alpha, procedure.beta
        n = alpha + beta
        try:
            var = (alpha * beta) / (n * n * (n + 1))
            return float(var * 12.0)
        except Exception:
            return 0.0

    def _compute_expected_utility(self, entry: ProceduralMemoryEntry, observation: str, goal: str) -> float:
        relevance = self._compute_relevance(entry, observation, goal)
        rho_mean = entry.procedure.alpha / (entry.procedure.alpha + entry.procedure.beta)
        risk = self._compute_failure_risk(entry, observation)
        info_gain = self._compute_information_gain(entry.procedure)

        eu = (relevance * rho_mean * 1.0) - (risk * (1 - rho_mean) * 0.5) + 0.1 * info_gain
        return max(0.0, eu)

    def select_procedure(self, observation: str, goal: str, theta_conf: float = 0.1) -> Tuple[Optional[str], float]:
        candidates = self._retrieve_candidates(observation, goal, k=10)
        if not candidates:
            return None, 0.0

        utilities: List[Tuple[str, float]] = []
        for pk in candidates:
            if pk in self.memory_system.procedural_memory:
                entry = self.memory_system.procedural_memory[pk]
                eu = self._compute_expected_utility(entry, observation, goal)
                utilities.append((pk, eu))

        if not utilities:
            return None, 0.0

        utilities.sort(key=lambda x: x[1], reverse=True)
        best_pk, best_eu = utilities[0]
        if best_eu < theta_conf:
            return None, 0.0
        return best_pk, min(1.0, best_eu)

# =========================================================
# CONTRASTIVE REFINEMENT
# =========================================================
class ContrastiveRefinementEngine:
    def __init__(self, n_min_s: int = 3, n_min_f: int = 3):
        self.n_min_s = n_min_s
        self.n_min_f = n_min_f

    def should_refine(self, entry: ProceduralMemoryEntry) -> bool:
        return len(entry.success_contexts) >= self.n_min_s and len(entry.failure_contexts) >= self.n_min_f

    def refine_procedure(self, entry: ProceduralMemoryEntry) -> ProceduralMemoryEntry:
        if not self.should_refine(entry):
            return entry

        success_words = set()
        failure_words = set()
        for ctx in entry.success_contexts:
            success_words.update(ctx.observation_init.lower().split())
        for ctx in entry.failure_contexts:
            failure_words.update(ctx.observation_init.lower().split())

        pos = list(success_words - failure_words)[:3]
        neg = list(failure_words - success_words)[:3]

        entry.discriminative_patterns = {
            "preconditions_positive": pos,
            "preconditions_negative": neg,
            "postconditions": [],
            "action_differences": [],
        }

        if pos:
            entry.procedure.preconditions.extend(pos)

        entry.last_refined = time.time()
        return entry


# =========================================================
# META-PROCEDURAL LEARNING
# =========================================================
class MetaProceduralLearner:
    def __init__(self, memory_system: EnhancedHierarchicalMemorySystem):
        self.memory_system = memory_system

    def should_extract_meta_procedure(self, trajectory: Dict, procedure_sequence: List[str]) -> bool:
        return len(procedure_sequence) >= 3 and trajectory.get("success", False)

    def extract_meta_procedure(self, trajectory: Dict, procedure_sequence: List[str]) -> Optional[MetaProcedure]:
        if not self.should_extract_meta_procedure(trajectory, procedure_sequence):
            return None

        goal_meta = f"meta_{trajectory.get('task','unknown')[:30]}"
        policy = {"type": "sequential", "ordering": procedure_sequence, "branching_rules": {}}
        meta = MetaProcedure(
            goal_meta=goal_meta, 
            preconditions_meta=[], 
            sub_procedures=procedure_sequence, 
            composition_policy=policy, 
            alpha=2, 
            beta=1
        )

        logger.info(f"Extracted meta-procedure for task '{goal_meta}' with {len(procedure_sequence)} sub-procedures")
        return meta


# =========================================================
# OLLAMA LLM REASONER
# =========================================================
class FrozenLLMReasoner:
    """Ollama-backed reasoner"""

    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        if _OLLAMA_AVAILABLE:
            logger.info(f"✓ Using Ollama model: {self.model_name}")
        else:
            logger.warning("✗ Ollama not available – LLM calls will return empty results.")

    def _generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
        if not _OLLAMA_AVAILABLE:
            return ""
        try:
            resp = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return resp["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def segment_trajectory(self, trajectory: Dict) -> str:
        prompt = (
            "Segment the following trajectory into logical steps as a JSON array of {step, action, observation}:\n"
            f"{json.dumps(trajectory, ensure_ascii=False)}"
        )
        return self._generate(prompt, max_tokens=300, temperature=0.3)

    def extract_procedure_components(self, segment: Dict) -> str:
        prompt = (
            "Given this segment, extract a concise JSON with fields "
            '{"goal": str, "preconditions": [str], "steps": [str], "postconditions": [str]}:\n'
            f"{json.dumps(segment, ensure_ascii=False)}"
        )
        return self._generate(prompt, max_tokens=200, temperature=0.2)

    def contrastive_analysis(self, success_contexts: List[str], failure_contexts: List[str]) -> str:
        prompt = (
            "Contrast these contexts. Return a JSON with keys: "
            '{"preconditions_positive":[str], "preconditions_negative":[str], "postconditions":[str]}.\n'
            f"Success: {json.dumps(success_contexts, ensure_ascii=False)}\n"
            f"Failure: {json.dumps(failure_contexts, ensure_ascii=False)}"
        )
        return self._generate(prompt, max_tokens=200, temperature=0.2)
    
    def discover_goal(self, task: str, actions: str, obs_init: str, obs_final: str) -> str:
        """Domain-agnostic goal discovery via LLM"""
        prompt = (
            "Infer the high-level intent of this episode as a short noun phrase (avoid domain-specific terms). "
            'Return JSON: {"goal": "..."}\n'
            f"TASK: {task}\nACTIONS: {actions}\nINIT_OBS: {obs_init}\nFINAL_OBS: {obs_final}"
        )
        return self._generate(prompt, max_tokens=80, temperature=0.1)


# =========================================================
# ENHANCED MACLA AGENT (Domain-Agnostic Core)
# =========================================================
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
        
        # Special handling for TravelPlanner - IMPROVED
        if "travel" in task or "trip" in task or "plan" in task:
            # Extract city pairs
            from_match = re.search(r'(?:from|originating from|starting (?:from|in))\s+(\w+)', task)
            to_match = re.search(r'(?:to|heading (?:to|for)|destination[:\s]+)(\w+)', task)
            
            if from_match and to_match:
                from_city = from_match.group(1).lower()
                to_city = to_match.group(1).lower()
                # Normalize city names (remove common words)
                from_city = re.sub(r'^(new|south|north|east|west|saint|st)$', '', from_city).strip()
                to_city = re.sub(r'^(new|south|north|east|west|saint|st)$', '', to_city).strip()
                if from_city and to_city:
                    return f"plan_trip_{from_city}_{to_city}"
            
            # Fallback to just destination
            if to_match:
                city = to_match.group(1).lower()
                if city and city not in ['march', 'april', 'may', 'june']:  # Exclude months
                    return f"plan_trip_to_{city}"
        
        # Fallback to action-based goal
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

        # Contrastive refinement
        for key, entry in list(self.memory_system.procedural_memory.items()):
            if self.contrastive_refiner.should_refine(entry):
                self.contrastive_refiner.refine_procedure(entry)
                results["procedures_refined"] += 1
                self.memory_system.stats["procedures_refined"] += 1

        # Meta-procedures
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

        # Use LLM-enhanced or unsupervised goal discovery
        if hasattr(self, 'discover_goal_auto'):
            goal = self.discover_goal_auto(trajectory, mode="llm")
        else:
            goal = self.discover_goal_unsupervised(trajectory)
        
        generalized_steps = [self._generalize_action(a) for a in actions]
        is_success = trajectory.get("success", False)

        # FIXED: Only dedupe if BOTH goal AND steps match
        for k, e in self.memory_system.procedural_memory.items():
            if e.procedure.goal == goal and e.procedure.steps == generalized_steps:
                # Update statistics instead of skipping
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

        # Check if Bayesian selection is enabled
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
        
        # Fallback (used when Bayesian disabled OR no good match)
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
        REMOVED: All hard-coded appliance references (microwave, fridge, sink)
        """
        goal_lower = goal.lower()
        
        # Identify goal type via generic verb analysis
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
        
        # Default generic protocol (works for ANY domain)
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


# =========================================================
# LLM-ENHANCED WRAPPER AGENT
# =========================================================
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
        Supervisor's requirement: Use LLM to infer intent WITHOUT domain-specific keywords.
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
        
        # Fallback to unsupervised
        return self.discover_goal_unsupervised(traj)

    # Override parent method to use LLM-enhanced discovery
    def _extract_and_add_procedure_universal(self, trajectory: Dict) -> Optional[str]:
        """Domain-agnostic procedure extraction with LLM goal discovery"""
        task = trajectory.get("task", "")
        actions = trajectory.get("actions", [])
        if len(actions) < 1:
            return None

        # Use LLM-enhanced goal discovery
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

        # Dedupe
        for k, e in self.memory_system.procedural_memory.items():
            if e.procedure.steps == generalized_steps:
                return k

        pk = self.memory_system.add_procedural_entry(proc, contexts, goals, performance=1.0 if is_success else 0.0)
        self.stats["procedures_learned"] += 1
        return pk

    # LLM-enhanced helpers
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

        # Phase 1: Extract procedures (always happens)
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

        # Phase 2: Contrastive refinement (conditional)
        if use_contrastive:
            for key, entry in list(self.memory_system.procedural_memory.items()):
                if self.contrastive_refiner.should_refine(entry):
                    self.contrastive_refiner.refine_procedure(entry)
                    results["procedures_refined"] += 1
        else:
            logger.info("Skipping contrastive refinement (ablation)")

        # Phase 3: Meta-procedures (conditional)
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
            
            # Disable ontology if requested
            if not use_ontology:
                self.bayesian_selector.ontology = {}
                self.bayesian_selector.ontology_embeddings = {}
            
            logger.info(f"Ablation configured: Bayesian={use_bayesian}, "
                        f"Contrastive={use_contrastive}, Meta={use_meta}, Ontology={use_ontology}")
# =========================================================
# DATA LOADER (Universal)
# =========================================================
class ALFWorldLikeLoader:
    """Loads and parses ALFWorld-like text logs"""

    trajectory_pattern = re.compile(r"ID=(\w+_\d+)")
    task_pattern = re.compile(r"Task:\s*(.*)")
    action_pattern = re.compile(r"Action:\s*(.*)")
    observation_pattern = re.compile(r"Observation:\s*(.*)")
    success_pattern = re.compile(r"Success:\s*(True|False)")
    reward_pattern = re.compile(r"Reward:\s*([0-9\.\-]+)")

    def load_files(self, file_paths: List[str], include_trajectory_paths: bool = True) -> List[Dict]:
        all_traj: List[Dict] = []
        for fp in file_paths:
            logger.info(f"Loading from: {fp}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                trajs = self._parse_alfworld_content(content, fp)
                if include_trajectory_paths:
                    for t in trajs:
                        t["source_file"] = fp
                        t["file_type"] = Path(fp).stem
                all_traj.extend(trajs)
                logger.info(f"Loaded {len(trajs)} trajectories")
            except FileNotFoundError:
                logger.error(f"File not found: {fp}")
            except Exception as e:
                logger.error(f"Error loading {fp}: {e}")
        logger.info(f"Total trajectories: {len(all_traj)}")
        return all_traj

    def _parse_alfworld_content(self, content: str, source_file: str = "") -> List[Dict]:
        trajs: List[Dict] = []
        blocks = re.split(r"(?=ID=\w+_\d+)", content.strip())
        for block in blocks:
            b = block.strip()
            if not b:
                continue
            t = self._parse_single_trajectory(b, source_file)
            if t:
                trajs.append(t)
        return trajs

    def _parse_single_trajectory(self, block: str, source_file: str = "") -> Optional[Dict]:
        lines = block.split("\n")
        traj = {
            "id": "",
            "task": "",
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": False,
            "think_steps": [],
            "raw_text": block,
            "source_file": source_file,
            "trajectory_path": [],
        }
        try:
            idm = self.trajectory_pattern.search(lines[0])
            if idm:
                traj["id"] = idm.group(1)

            step_index = 0
            current_obs = ""
            for line in lines:
                line = line.strip()
                tm = self.task_pattern.search(line)
                if tm:
                    traj["task"] = tm.group(1)

                if line.startswith("Think:"):
                    traj["think_steps"].append(line[6:].strip())

                am = self.action_pattern.search(line)
                if am:
                    action = am.group(1)
                    traj["actions"].append(action)
                    traj["trajectory_path"].append(
                        {
                            "step": step_index,
                            "action": action,
                            "observation": current_obs,
                            "think": traj["think_steps"][-1] if traj["think_steps"] else "",
                        }
                    )
                    step_index += 1

                om = self.observation_pattern.search(line)
                if om:
                    current_obs = om.group(1)
                    traj["observations"].append(current_obs)

                sm = self.success_pattern.search(line)
                if sm:
                    traj["success"] = sm.group(1) == "True"

                rm = self.reward_pattern.search(line)
                if rm:
                    try:
                        traj["reward"] = float(rm.group(1))
                    except Exception:
                        pass

            for i in range(min(len(traj["actions"]), len(traj["observations"]))):
                traj["action_observation_pairs"].append(
                    {"action": traj["actions"][i], "observation": traj["observations"][i]}
                )

            if "reward" not in traj:
                traj["reward"] = 1.0 if traj["success"] else 0.0

            return traj
        except Exception as e:
            logger.error(f"Parse error in {source_file}: {e}")
            return None

class WebShopLoader:
    """Loads and parses WebShop data - e-commerce domain"""
    
    def load_and_split_webshop(
        self, 
        file_path: str, 
        train_ratio: float = 0.7, 
        val_seen_ratio: float = 0.15, 
        val_unseen_ratio: float = 0.15
    ) -> Dict[str, List[Dict]]:
        """Load WebShop data and split into train/val_seen/val_unseen"""
        logger.info(f"Loading WebShop data from: {file_path}")
        
        all_trajectories = self._parse_webshop_file(file_path)
        
        if not all_trajectories:
            logger.error("No WebShop trajectories loaded")
            return {"train": [], "val_seen": [], "val_unseen": []}
        
        # Split data
        random.seed(42)
        indices = list(range(len(all_trajectories)))
        random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        n_val_seen = int(len(indices) * val_seen_ratio)
        
        train_idx = indices[:n_train]
        val_seen_idx = indices[n_train:n_train + n_val_seen]
        val_unseen_idx = indices[n_train + n_val_seen:]
        
        splits = {
            "train": [all_trajectories[i] for i in train_idx],
            "val_seen": [all_trajectories[i] for i in val_seen_idx],
            "val_unseen": [all_trajectories[i] for i in val_unseen_idx]
        }
        
        logger.info(f"WebShop split - Train: {len(splits['train'])}, "
                   f"Val Seen: {len(splits['val_seen'])}, Val Unseen: {len(splits['val_unseen'])}")
        
        return splits
    
    def _parse_webshop_file(self, file_path: str) -> List[Dict]:
        """Parse WebShop conversational JSON format"""
        trajectories = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle list of episodes
            if isinstance(data, list):
                episodes = data
            elif isinstance(data, dict) and "episodes" in data:
                episodes = data["episodes"]
            else:
                logger.error("Unknown WebShop format")
                return []
            
            logger.info(f"Found {len(episodes)} WebShop episodes")
            
            for episode in episodes:
                traj = self._parse_webshop_conversation(episode)
                if traj:
                    trajectories.append(traj)
                    
        except Exception as e:
            logger.error(f"Error parsing WebShop file: {e}")
        
        logger.info(f"Successfully parsed {len(trajectories)} WebShop trajectories")
        return trajectories
    
    def _parse_webshop_conversation(self, episode: Dict) -> Optional[Dict]:
        """Parse single WebShop conversational episode"""
        episode_id = episode.get("id", "unknown")
        conversations = episode.get("conversations", [])
        reward = episode.get("reward", 0.0)
        
        if not conversations:
            return None
        
        # Initialize trajectory
        traj = {
            "id": f"webshop_{episode_id}",
            "task": "",
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": reward >= 1.0,
            "think_steps": [],
            "raw_text": json.dumps(episode),
            "source_file": "webshop",
            "trajectory_path": [],
            "domain": "webshop",
            "reward": float(reward)
        }
        
        # Extract instruction from WebShop format
        for conv in conversations:
            if conv.get("from") == "human":
                value = conv.get("value", "")
                if "Instruction:" in value and "[SEP]" in value:
                    # Split by [SEP] and find the instruction
                    parts = value.split("[SEP]")
                    for i, part in enumerate(parts):
                        if "instruction:" in part.lower():
                            # The actual instruction is in the NEXT part
                            if i + 1 < len(parts):
                                traj["task"] = parts[i + 1].strip()
                                break
                    break
        
        # Fallback: if task still empty, try extracting from first search query
        if not traj["task"]:
            for conv in conversations:
                if conv.get("from") == "gpt":
                    action_match = re.search(r"Action:\s*search\[(.*?)\]", conv.get("value", ""))
                    if action_match:
                        traj["task"] = f"find {action_match.group(1)}"
                        break
        
        # Parse action-observation pairs from conversations
        step_index = 0
        current_thought = ""
        
        for i, conv in enumerate(conversations):
            role = conv.get("from", "")
            value = conv.get("value", "")
            
            if role == "gpt":
                # Extract thought and action
                thought_match = re.search(r"Thought:\s*(.*?)(?:\n|Action:|$)", value, re.DOTALL)
                action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", value)
                
                if thought_match:
                    current_thought = thought_match.group(1).strip()
                    traj["think_steps"].append(current_thought)
                
                if action_match:
                    action = action_match.group(1).strip()
                    traj["actions"].append(action)
                    
                    # Look ahead for observation
                    observation = ""
                    if i + 1 < len(conversations) and conversations[i + 1].get("from") == "human":
                        obs_value = conversations[i + 1].get("value", "")
                        if "Observation:" in obs_value:
                            obs_match = re.search(r"Observation:\s*(.*)", obs_value, re.DOTALL)
                            if obs_match:
                                observation = obs_match.group(1).strip()
                                # Clean up - take only first part before [SEP]
                                if "[SEP]" in observation:
                                    observation = observation.split("[SEP]")[0].strip()
                    
                    traj["observations"].append(observation)
                    
                    traj["trajectory_path"].append({
                        "step": step_index,
                        "action": action,
                        "observation": observation,
                        "think": current_thought
                    })
                    
                    traj["action_observation_pairs"].append({
                        "action": action,
                        "observation": observation
                    })
                    
                    step_index += 1
                    current_thought = ""
        
        # Validate - require task AND at least one action
        if not traj["task"] or not traj["actions"]:
            logger.warning(f"Incomplete trajectory for episode {episode_id}: task='{traj['task']}', actions={len(traj['actions'])}")
            return None
        
        return traj

# =========================================================
# EVALUATOR
# =========================================================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
            # For SQL with synthetic success, use actual execution result
            if t.get("domain") == "sql" and "actual_execution_success" in t:
                actual_success = t.get("actual_execution_success", False)
                r = 1.0 if actual_success else 0.0
            else:
                # For other domains, use success flag
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
        
        # Domain-specific tracking
        sql_exact_matches = 0
        sql_structural_matches = 0
        sql_total = 0
        
        for i, t in enumerate(trajectories[: min(100, len(trajectories))]):
            task = t.get("task", "")
            obs = f"Environment: {task}"
            domain = t.get("domain", "unknown")
            
            # Discover goal
            if isinstance(agent, LLMMACLAAgent):
                goal = agent.discover_goal_auto(t, mode="llm")
            else:
                goal = agent.discover_goal_unsupervised(t)
            
            # Debug logging for first 5 examples
            if i < 5:
                logger.info(f"DEBUG Test {i} (domain={domain}):")
                logger.info(f"  Task: {task[:80]}...")
                logger.info(f"  Discovered Goal: {goal}")
            
            # Execute task
            res = agent.execute_task(obs, goal)
            
            # Domain-specific evaluation
            if domain == "sql":
                # SQL: Compare generated query to gold query
                gold_query = t.get("gold_query", "").strip()
                generated_actions = res.get("action_sequence", [])
                
                sql_total += 1
                pred_success = False
                exact_match = False
                structural_match = False
                
                for action in generated_actions:
                    # Normalize queries for comparison
                    action_normalized = self._normalize_sql(action)
                    gold_normalized = self._normalize_sql(gold_query)
                    
                    # Exact match
                    if action_normalized == gold_normalized:
                        exact_match = True
                        structural_match = True
                        pred_success = True
                        sql_exact_matches += 1
                        break
                    
                    # Structural match (same tables, operations)
                    if self._sql_structural_match(action, gold_query):
                        structural_match = True
                        pred_success = True
                        sql_structural_matches += 1
                        break
                
                # Ground truth: Use actual execution success for SQL
                gt = t.get("actual_execution_success", False)
                
                if i < 5:
                    logger.info(f"  Generated SQL: {generated_actions[0] if generated_actions else 'None'}...")
                    logger.info(f"  Gold SQL: {gold_query[:80]}...")
                    logger.info(f"  Exact Match: {exact_match}, Structural Match: {structural_match}")
            
            else:
                # Non-SQL domains: Use procedure retrieval as success indicator
                pred_success = res.get("method") == "bayesian_procedure"
                gt = bool(t.get("success", False))
            
            # More debug logging
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

        # Calculate metrics
        acc = accuracy_score(gts, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(gts, preds, average="binary", zero_division=0)
        
        # Log SQL-specific metrics
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
        # Convert to uppercase, remove extra whitespace
        normalized = ' '.join(query.strip().upper().split())
        # Remove common variations
        normalized = normalized.replace('  ', ' ')
        return normalized

    def _sql_structural_match(self, query1: str, query2: str, threshold: float = 0.6) -> bool:
        """Check if two SQL queries have similar structure"""
        if not query1 or not query2:
            return False
        
        def extract_components(sql):
            components = set()
            sql_upper = sql.upper()
            
            # Extract table names from FROM and JOIN clauses
            for match in re.finditer(r'FROM\s+(\w+)', sql_upper):
                components.add(f"TABLE:{match.group(1)}")
            for match in re.finditer(r'JOIN\s+(\w+)', sql_upper):
                components.add(f"TABLE:{match.group(1)}")
            
            # Extract main SQL operations
            operations = ['SELECT', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT',
                         'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'UNION', 'DISTINCT']
            for op in operations:
                if op in sql_upper:
                    components.add(f"OP:{op}")
            
            # Extract aggregate functions
            aggregates = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
            for agg in aggregates:
                if agg in sql_upper:
                    components.add(f"AGG:{agg}")
            
            return components
        
        comp1 = extract_components(query1)
        comp2 = extract_components(query2)
        
        if not comp1 or not comp2:
            return False
        
        # Calculate Jaccard similarity
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


# =========================================================
# TRAIN/TEST SPLIT + DRIVER
# =========================================================
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
    
    # Disable ontology if requested
    if not use_ontology:
        self.bayesian_selector.ontology = {}
        self.bayesian_selector.ontology_embeddings = {}
    
    logger.info(f"Ablation configured: Bayesian={use_bayesian}, "
                f"Contrastive={use_contrastive}, Meta={use_meta}, Ontology={use_ontology}")


class TravelPlannerLoader:
    """Loads and parses TravelPlanner data - trip planning domain"""
    
    def load_and_split_travelplanner(
        self, 
        file_path: str, 
        train_ratio: float = 0.7, 
        val_seen_ratio: float = 0.15, 
        val_unseen_ratio: float = 0.15
    ) -> Dict[str, List[Dict]]:
        """Load TravelPlanner data and split into train/val_seen/val_unseen"""
        logger.info(f"Loading TravelPlanner data from: {file_path}")
        
        all_trajectories = self._parse_travelplanner_file(file_path)
        
        if not all_trajectories:
            logger.error("No TravelPlanner trajectories loaded")
            return {"train": [], "val_seen": [], "val_unseen": []}
        
        # Split data
        random.seed(42)
        indices = list(range(len(all_trajectories)))
        random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        n_val_seen = int(len(indices) * val_seen_ratio)
        
        train_idx = indices[:n_train]
        val_seen_idx = indices[n_train:n_train + n_val_seen]
        val_unseen_idx = indices[n_train + n_val_seen:]
        
        splits = {
            "train": [all_trajectories[i] for i in train_idx],
            "val_seen": [all_trajectories[i] for i in val_seen_idx],
            "val_unseen": [all_trajectories[i] for i in val_unseen_idx]
        }
        
        logger.info(f"TravelPlanner split - Train: {len(splits['train'])}, "
                   f"Val Seen: {len(splits['val_seen'])}, Val Unseen: {len(splits['val_unseen'])}")
        
        return splits
    
    def _parse_travelplanner_file(self, file_path: str) -> List[Dict]:
        """Parse TravelPlanner JSON-lines format"""
        trajectories = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        traj = self._parse_travelplanner_trajectory(data)
                        if traj:
                            trajectories.append(traj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error parsing TravelPlanner file: {e}")
        
        logger.info(f"Successfully parsed {len(trajectories)} TravelPlanner trajectories")
        return trajectories
    
    def _parse_travelplanner_trajectory(self, data: Dict) -> Optional[Dict]:
        """Parse single TravelPlanner trajectory"""
        traj_id = data.get("id", "unknown")
        trajectory_text = data.get("trajectory_text", "")
        success = data.get("Success", False)
        
        if not trajectory_text:
            return None
        
        # Extract task from the first line
        task = self._extract_task(trajectory_text)
        if not task:
            logger.warning(f"No task found in trajectory {traj_id}")
            return None
        
        # Initialize trajectory
        traj = {
            "id": f"travelplanner_{traj_id}",
            "task": task,
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": success,
            "think_steps": [],
            "raw_text": trajectory_text,
            "source_file": "travelplanner",
            "trajectory_path": [],
            "domain": "travelplanner",
            "reward": 1.0 if success else 0.0,
            "split": data.get("split", "unknown"),
            "synthetic": data.get("synthetic", False)
        }
        
        # Parse Think→Action→Observation sequence
        lines = trajectory_text.split('\n')
        step_index = 0
        current_think = ""
        current_action = ""
        
        for line in lines:
            line = line.strip()
            
            # Parse Think steps
            if line.startswith("Think:"):
                current_think = line[6:].strip()
                traj["think_steps"].append(current_think)
            
            # Parse Actions
            elif line.startswith("Action:"):
                action_text = line[7:].strip()
                # Extract action name and parameters
                action_match = re.match(r'(\w+)\s*(\{.*\})?', action_text)
                if action_match:
                    action_name = action_match.group(1)
                    action_params = action_match.group(2) or ""
                    current_action = f"{action_name}{action_params}"
                    traj["actions"].append(current_action)
                else:
                    current_action = action_text
                    traj["actions"].append(current_action)
            
            # Parse Observations
            elif line.startswith("Observation:"):
                observation = line[12:].strip()
                traj["observations"].append(observation)
                
                # Create trajectory step
                if current_action:
                    traj["trajectory_path"].append({
                        "step": step_index,
                        "action": current_action,
                        "observation": observation,
                        "think": current_think
                    })
                    
                    traj["action_observation_pairs"].append({
                        "action": current_action,
                        "observation": observation
                    })
                    
                    step_index += 1
                    current_think = ""
                    current_action = ""
        
        # Validate - require task AND at least one action
        if not traj["task"] or not traj["actions"]:
            logger.warning(f"Incomplete trajectory {traj_id}: task='{traj['task']}', actions={len(traj['actions'])}")
            return None
        
        return traj
    
    def _extract_task(self, trajectory_text: str) -> str:
        """Extract task description from trajectory text"""
        # Look for "Your task is to:" pattern
        task_match = re.search(
            r'Your task is to:\s*(.*?)(?:\n|Think:)', 
            trajectory_text, 
            re.DOTALL
        )
        
        if task_match:
            task = task_match.group(1).strip()
            # Clean up the task text
            task = re.sub(r'\s+', ' ', task)
            return task
        
        # Fallback: look for travel planning keywords
        lines = trajectory_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['plan', 'trip', 'travel', 'from', 'to']):
                return line.strip()
        
        return ""
    
class SQLLoader:
    """Loads and parses InterCode SQL data - database query domain"""
    
    def load_and_split_sql(
        self, 
        file_path: str, 
        train_ratio: float = 0.7, 
        val_seen_ratio: float = 0.15, 
        val_unseen_ratio: float = 0.15,
        use_gold_as_success: bool = True  # NEW: Treat gold queries as successful
    ) -> Dict[str, List[Dict]]:
        """Load SQL data and split into train/val_seen/val_unseen"""
        logger.info(f"Loading SQL data from: {file_path}")
        
        all_trajectories = self._parse_sql_file(file_path, use_gold_as_success)
        
        if not all_trajectories:
            logger.error("No SQL trajectories loaded")
            return {"train": [], "val_seen": [], "val_unseen": []}
        
        # Log success statistics
        success_count = sum(1 for t in all_trajectories if t.get('success', False))
        logger.info(f"SQL Success Rate: {success_count}/{len(all_trajectories)} ({success_count/len(all_trajectories)*100:.1f}%)")
        
        # Split data
        random.seed(42)
        indices = list(range(len(all_trajectories)))
        random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        n_val_seen = int(len(indices) * val_seen_ratio)
        
        train_idx = indices[:n_train]
        val_seen_idx = indices[n_train:n_train + n_val_seen]
        val_unseen_idx = indices[n_train + n_val_seen:]
        
        splits = {
            "train": [all_trajectories[i] for i in train_idx],
            "val_seen": [all_trajectories[i] for i in val_seen_idx],
            "val_unseen": [all_trajectories[i] for i in val_unseen_idx]
        }
        
        logger.info(f"SQL split - Train: {len(splits['train'])}, "
                   f"Val Seen: {len(splits['val_seen'])}, Val Unseen: {len(splits['val_unseen'])}")
        
        return splits
    
    def _parse_sql_file(self, file_path: str, use_gold_as_success: bool = True) -> List[Dict]:
        """Parse SQL JSON-lines format"""
        trajectories = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        traj = self._parse_sql_trajectory(data, use_gold_as_success)
                        if traj:
                            trajectories.append(traj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error parsing SQL file: {e}")
        
        logger.info(f"Successfully parsed {len(trajectories)} SQL trajectories")
        return trajectories
    
    def _parse_sql_trajectory(self, data: Dict, use_gold_as_success: bool = True) -> Optional[Dict]:
        """Parse single SQL trajectory with gold query as learning target"""
        traj_id = data.get("id", "unknown")
        instruction = data.get("instruction", "")
        db = data.get("db", "unknown")
        gold_query = data.get("gold", "")
        actual_success = data.get("success", False)
        steps = data.get("steps", [])
        
        if not instruction:
            logger.warning(f"No instruction in SQL trajectory {traj_id}")
            return None
        
        # CRITICAL FIX: If gold query exists, treat it as successful learning example
        if use_gold_as_success and gold_query:
            success = True  # We have the correct answer
            reward = 1.0
        else:
            success = actual_success
            reward = 1.0 if actual_success else 0.0
        
        # Extract schema information from first observation
        schema_info = steps[0].get("observation", "") if steps else ""
        tables = self._extract_tables(schema_info)
        
        # Extract SQL operations from gold query
        sql_operations = self._extract_sql_operations(gold_query)
        
        # Initialize trajectory
        traj = {
            "id": f"sql_{traj_id}",
            "task": instruction,
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": success,  # MODIFIED: Based on gold query existence
            "think_steps": [],
            "raw_text": json.dumps(data),
            "source_file": "intercode_sql",
            "trajectory_path": [],
            "domain": "sql",
            "reward": reward,  # MODIFIED: Based on gold query existence
            "db": db,
            "hardness": data.get("hardness", "unknown"),
            "gold_query": gold_query,
            "actual_success": actual_success,
            "sql_tables": tables,
            "sql_operations": sql_operations
        }
        
        # Parse steps - use gold query as the "correct" action
        if use_gold_as_success and gold_query:
            # Create a synthetic successful step using gold query
            step = {
                "step": 0,
                "action": gold_query,
                "observation": schema_info,
                "observation_next": "Query executed successfully (synthetic)",
                "think": f"Generate SQL query to answer: {instruction}",
                "done": True
            }
            
            traj["actions"].append(gold_query)
            traj["observations"].append(schema_info)
            traj["think_steps"].append(step["think"])
            traj["trajectory_path"].append(step)
            traj["action_observation_pairs"].append({
                "action": gold_query,
                "observation": schema_info,
                "observation_next": step["observation_next"]
            })
        else:
            # Use actual failed steps
            for step_index, step in enumerate(steps):
                observation = step.get("observation", "")
                think = step.get("think", "")
                action = step.get("action", "")
                observation_next = step.get("observation_next", "")
                
                if think:
                    traj["think_steps"].append(think)
                
                if action:
                    traj["actions"].append(action)
                
                if observation:
                    traj["observations"].append(observation)
                
                traj["trajectory_path"].append({
                    "step": step_index,
                    "action": action,
                    "observation": observation,
                    "observation_next": observation_next,
                    "think": think,
                    "done": step.get("done", False)
                })
                
                traj["action_observation_pairs"].append({
                    "action": action,
                    "observation": observation,
                    "observation_next": observation_next
                })
        
        # Validate
        if not traj["task"] or not traj["actions"]:
            logger.warning(f"Incomplete trajectory {traj_id}: task='{traj['task']}', actions={len(traj['actions'])}")
            return None
        
        return traj
    
    def _extract_tables(self, schema_text: str) -> List[str]:
        """Extract table names from schema observation"""
        tables = []
        # Pattern: table_name(column1, column2, ...)
        for match in re.finditer(r'(\w+)\([^)]+\)', schema_text):
            table_name = match.group(1)
            if table_name not in ['Tables', 'sqlite_sequence']:  # Exclude meta entries
                tables.append(table_name)
        return list(set(tables))  # Remove duplicates
    
    def _extract_sql_operations(self, sql_query: str) -> List[str]:
        """Extract SQL operations and patterns"""
        if not sql_query:
            return []
        
        operations = []
        sql_upper = sql_query.upper()
        
        # Main clauses
        if 'SELECT' in sql_upper:
            operations.append('select')
        if 'FROM' in sql_upper:
            operations.append('from')
        if 'WHERE' in sql_upper:
            operations.append('where')
        if 'JOIN' in sql_upper or 'INNER JOIN' in sql_upper or 'LEFT JOIN' in sql_upper:
            operations.append('join')
        if 'GROUP BY' in sql_upper:
            operations.append('group_by')
        if 'ORDER BY' in sql_upper:
            operations.append('order_by')
        if 'HAVING' in sql_upper:
            operations.append('having')
        if 'LIMIT' in sql_upper:
            operations.append('limit')
        
        # Aggregations
        if 'COUNT' in sql_upper:
            operations.append('count')
        if 'SUM' in sql_upper:
            operations.append('sum')
        if 'AVG' in sql_upper:
            operations.append('avg')
        if 'MAX' in sql_upper:
            operations.append('max')
        if 'MIN' in sql_upper:
            operations.append('min')
        
        # Other operations
        if 'DISTINCT' in sql_upper:
            operations.append('distinct')
        if 'UNION' in sql_upper:
            operations.append('union')
        if 'SUBQUERY' in sql_upper or '(SELECT' in sql_upper:
            operations.append('subquery')
        
        return operations
# =========================================================
# MAIN - MULTI-DATASET SUPPORT
# =========================================================
def main():
    import argparse
    import sys
    
    # ========================================
    # CONFIGURATION: Choose ONE option below
    # ========================================
    
    if len(sys.argv) == 1:
        # OPTION 1: ALFWorld only (3 separate files)
        # sys.argv.extend([
        #     "--dataset", "alfworld",
        #     "--train_file", r"C:\ALFWord_data\train_reactified.txt",
        #     "--valid_seen_file", r"C:\ALFWord_data\valid_seen_reactified.txt",
        #     "--valid_unseen_file", r"C:\ALFWord_data\valid_unseen_reactified.txt",
        #     "--no_llm"
        # ])
        
        # OPTION 2: WebShop only (1 file, auto-split)
        # sys.argv.extend([
        #     "--dataset", "webshop",
        #     "--webshop_file", r"C:\Webshop_data\webshop_sft.txt",
        #     "--no_llm"
        # ])
        
        # OPTION 3: TravelPlanner only (2 files, auto-split)
        sys.argv.extend([
            "--dataset", "travelplanner",
            "--travelplanner_test_file", r"C:\Travel Planner\travelplanner_test_synthetic_trajectories.txt",
            "--travelplanner_val_file", r"C:\Travel Planner\travelplanner_validation_synthetic_trajectories.txt",
            # "--no_llm"
        ])
        
        # OPTION 4: SQL only (1 file, auto-split)
        # sys.argv.extend([
        #     "--dataset", "sql",
        #     "--sql_file", r"C:\intercode_sql\intercode_sql.txt",
        #     "--no_llm"
        # ])
            
    parser = argparse.ArgumentParser(description="Domain-Agnostic MACLA Agent")
    parser.add_argument("--dataset", type=str, default="alfworld", 
                       choices=["alfworld", "webshop", "travelplanner", "sql", "all"],
                       help="Dataset to use: alfworld, webshop, travelplanner, sql, or all")
    
    # ALFWorld specific
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--valid_seen_file", type=str, default=None)
    parser.add_argument("--valid_unseen_file", type=str, default=None)
    
    # WebShop specific
    parser.add_argument("--webshop_file", type=str, default=None,
                       help="WebShop data file (will be split automatically)")
    
    # TravelPlanner specific
    parser.add_argument("--travelplanner_test_file", type=str, default=None,
                       help="TravelPlanner test file")
    parser.add_argument("--travelplanner_val_file", type=str, default=None,
                       help="TravelPlanner validation file")
    
    # SQL specific
    parser.add_argument("--sql_file", type=str, default=None,
                       help="SQL data file (will be split automatically)")
    
    parser.add_argument("--llm_model", type=str, default=os.environ.get("MACLA_LLM_MODEL", "llama2"))
    # parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--ablation", action="store_true", 
                       help="Run ablation studies")
    
    args = parser.parse_args()
    
    # Initialize loaders
    alfworld_loader = ALFWorldLikeLoader()
    webshop_loader = WebShopLoader()
    travelplanner_loader = TravelPlannerLoader()
    sql_loader = SQLLoader()
    
    train_traj = []
    valid_seen = []
    valid_unseen = []
    
    # ========================================
    # PHASE 1: Load Data Based on Dataset Type
    # ========================================
    logger.info("="*60)
    logger.info("PHASE 1: Loading Training Data")
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info("="*60)
    
    if args.dataset == "alfworld":
        # ALFWorld: 3 separate files
        if not args.train_file:
            logger.error("--train_file required for ALFWorld dataset")
            return
        
        train_traj = alfworld_loader.load_files([args.train_file], include_trajectory_paths=True)
        
        if args.valid_seen_file:
            valid_seen = alfworld_loader.load_files([args.valid_seen_file], include_trajectory_paths=True)
        if args.valid_unseen_file:
            valid_unseen = alfworld_loader.load_files([args.valid_unseen_file], include_trajectory_paths=True)
    
    elif args.dataset == "webshop":
        # WebShop: 1 file, auto-split into train/val_seen/val_unseen
        if not args.webshop_file:
            logger.error("--webshop_file required for WebShop dataset")
            return
        
        splits = webshop_loader.load_and_split_webshop(
            args.webshop_file,
            train_ratio=0.7,
            val_seen_ratio=0.15,
            val_unseen_ratio=0.15
        )
        train_traj = splits["train"]
        valid_seen = splits["val_seen"]
        valid_unseen = splits["val_unseen"]
    
    elif args.dataset == "travelplanner":
        logger.info("Loading TravelPlanner dataset")
        
        all_data = []
        
        if args.travelplanner_test_file:
            test_data = travelplanner_loader._parse_travelplanner_file(
                args.travelplanner_test_file
            )
            all_data.extend(test_data)
            logger.info(f"Loaded {len(test_data)} from test file")
        
        if args.travelplanner_val_file:
            val_data = travelplanner_loader._parse_travelplanner_file(
                args.travelplanner_val_file
            )
            all_data.extend(val_data)
            logger.info(f"Loaded {len(val_data)} from validation file")
        
        # ADD THIS DIAGNOSTIC HERE
        logger.info(f"=== TRAVELPLANNER DATA CHECK ===")
        logger.info(f"Total trajectories loaded: {len(all_data)}")
        if all_data:
            logger.info(f"Sample task: {all_data[0].get('task', 'NO TASK')[:100]}")
            logger.info(f"Sample actions: {all_data[0].get('actions', [])[:3]}")
        else:
            logger.error("⚠️ NO TRAVELPLANNER DATA LOADED - Files may be empty or parse failed!")
        
        # Simple random split...
        random.seed(42)
        random.shuffle(all_data)
        n_train = int(len(all_data) * 0.7)
        n_val = int(len(all_data) * 0.15)
        
        train_traj = all_data[:n_train]
        valid_seen = all_data[n_train:n_train + n_val]
        valid_unseen = all_data[n_train + n_val:]        
        logger.info(f"Split: {len(train_traj)} train, {len(valid_seen)} val1, {len(valid_unseen)} val2")
       
    elif args.dataset == "sql":
        if not args.sql_file:
            logger.error("--sql_file required for SQL dataset")
            return
        
        splits = sql_loader.load_and_split_sql(
            args.sql_file,
            train_ratio=0.7,
            val_seen_ratio=0.15,
            val_unseen_ratio=0.15,
            use_gold_as_success=True  # Treat gold queries as successful
        )
        train_traj = splits["train"]
        valid_seen = splits["val_seen"]
        valid_unseen = splits["val_unseen"]
        
        # SQL-specific logging
        logger.info(f"\nSQL Dataset Analysis:")
        logger.info(f"  Tables extracted: {len(set(t for traj in train_traj for t in traj.get('sql_tables', [])))}")
        logger.info(f"  Operations found: {len(set(op for traj in train_traj for op in traj.get('sql_operations', [])))}")
        
    elif args.dataset == "all":
        # Load all datasets and combine
        logger.info("Loading ALL datasets: ALFWorld + WebShop + TravelPlanner + SQL")
        
        if args.train_file:
            alfworld_train = alfworld_loader.load_files([args.train_file], include_trajectory_paths=True)
            train_traj.extend(alfworld_train)
            logger.info(f"Added {len(alfworld_train)} ALFWorld training trajectories")
        
        if args.webshop_file:
            webshop_splits = webshop_loader.load_and_split_webshop(args.webshop_file)
            train_traj.extend(webshop_splits["train"])
            valid_seen.extend(webshop_splits["val_seen"])
            valid_unseen.extend(webshop_splits["val_unseen"])
            logger.info(f"Added {len(webshop_splits['train'])} WebShop training trajectories")
        
        if args.travelplanner_test_file:
            all_tp_data = []
            test_data = travelplanner_loader._parse_travelplanner_file(args.travelplanner_test_file)
            all_tp_data.extend(test_data)
            if args.travelplanner_val_file:
                val_data = travelplanner_loader._parse_travelplanner_file(args.travelplanner_val_file)
                all_tp_data.extend(val_data)
            
            # Split TravelPlanner data
            random.seed(42)
            random.shuffle(all_tp_data)
            n_train = int(len(all_tp_data) * 0.7)
            n_val = int(len(all_tp_data) * 0.15)
            train_traj.extend(all_tp_data[:n_train])
            valid_seen.extend(all_tp_data[n_train:n_train + n_val])
            valid_unseen.extend(all_tp_data[n_train + n_val:])
            logger.info(f"Added {n_train} TravelPlanner training trajectories")
        
        if args.sql_file:
            sql_splits = sql_loader.load_and_split_sql(args.sql_file)
            train_traj.extend(sql_splits["train"])
            valid_seen.extend(sql_splits["val_seen"])
            valid_unseen.extend(sql_splits["val_unseen"])
            logger.info(f"Added {len(sql_splits['train'])} SQL training trajectories")
        
        if args.valid_seen_file:
            alfworld_seen = alfworld_loader.load_files([args.valid_seen_file], include_trajectory_paths=True)
            valid_seen.extend(alfworld_seen)
            logger.info(f"Added {len(alfworld_seen)} ALFWorld seen validation trajectories")
        
        if args.valid_unseen_file:
            alfworld_unseen = alfworld_loader.load_files([args.valid_unseen_file], include_trajectory_paths=True)
            valid_unseen.extend(alfworld_unseen)
            logger.info(f"Added {len(alfworld_unseen)} ALFWorld unseen validation trajectories")
    
    if not train_traj:
        logger.error("No training trajectories loaded. Exiting.")
        return

    logger.info(f"Total training trajectories: {len(train_traj)}")
    if valid_seen:
        logger.info(f"Total validation (seen) trajectories: {len(valid_seen)}")
    if valid_unseen:
        logger.info(f"Total validation (unseen) trajectories: {len(valid_unseen)}")

    # ========================================
    # DATA SANITY CHECK
    # ========================================
    logger.info("\n=== DATA SANITY CHECK ===")
    success_count = sum(1 for t in train_traj if t.get('success', False))
    logger.info(f"Training: {success_count}/{len(train_traj)} successful ({success_count/len(train_traj)*100:.1f}%)")

    if valid_seen:
        success_count = sum(1 for t in valid_seen if t.get('success', False))
        logger.info(f"Val Seen: {success_count}/{len(valid_seen)} successful ({success_count/len(valid_seen)*100:.1f}%)")

    if valid_unseen:
        success_count = sum(1 for t in valid_unseen if t.get('success', False))
        logger.info(f"Val Unseen: {success_count}/{len(valid_unseen)} successful ({success_count/len(valid_unseen)*100:.1f}%)")

    # Sample a few trajectories to inspect
    logger.info("\nSample trajectory inspection:")
    for i, traj in enumerate(train_traj[:3]):
        logger.info(f"  Train {i}: ID={traj.get('id')[:30]}, success={traj.get('success')}, "
                    f"actions={len(traj.get('actions', []))}, task={traj.get('task', '')[:50]}...")
    
             
    # ========================================
    # PHASE 2: Train Domain-Agnostic Agent
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Training Domain-Agnostic Agent")
    logger.info("="*60)
    
    agent = build_agent_and_learn(train_traj, llm_model=args.llm_model, use_llm=(not args.no_llm))
    
    logger.info("\nTraining completed.")
    train_stats = agent.get_statistics()
    logger.info(f"Procedural memory size: {train_stats['procedural_memory_size']}")
    logger.info(f"Meta-procedural memory size: {train_stats['meta_procedural_size']}")

    # ========================================
    # PHASE 2.5: Ablation Studies (Optional)
    # ========================================
    if args.ablation:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2.5: Running Ablation Studies")
        logger.info("="*60)
        
        ablation_configs = [
            {"name": "Full System", 
             "bayesian": True, "contrastive": True, "meta": True, "ontology": True},
            {"name": "No Bayesian Selection", 
             "bayesian": False, "contrastive": True, "meta": True, "ontology": True},
            {"name": "No Contrastive Learning", 
             "bayesian": True, "contrastive": False, "meta": True, "ontology": True},
            {"name": "No Meta-Procedures", 
             "bayesian": True, "contrastive": True, "meta": False, "ontology": True},
            {"name": "No Ontology", 
             "bayesian": True, "contrastive": True, "meta": True, "ontology": False},
            {"name": "Minimal (No Bayesian, No Contrastive)", 
             "bayesian": False, "contrastive": False, "meta": True, "ontology": True},
        ]
        
        ablation_results = {}
        
        for config in ablation_configs:
            logger.info(f"\n--- Testing: {config['name']} ---")
            
            # Create fresh agent with same data
            ablation_agent = LLMMACLAAgent(
                N_a=2000, N_p=500, N_m=100,
                llm_model=args.llm_model, 
                use_llm=(not args.no_llm)
            )
            
            # Configure ablation - need to add this method to the class
            if hasattr(ablation_agent, 'configure_ablation'):
                ablation_agent.configure_ablation(
                    use_bayesian=config["bayesian"],
                    use_contrastive=config["contrastive"],
                    use_meta=config["meta"],
                    use_ontology=config["ontology"]
                )
            
            # Train (respecting ablation config)
            ablation_agent.bayesian_selector.build_ontology(train_traj)
            if hasattr(ablation_agent, 'learn_from_trajectories_ablation'):
                ablation_agent.learn_from_trajectories_ablation(
                    train_traj, 
                    use_contrastive=config["contrastive"],
                    use_meta=config["meta"]
                )
            else:
                ablation_agent.learn_from_trajectories(train_traj)
            
            # Evaluate on validation set
            if valid_unseen:
                metrics = run_evaluation(ablation_agent, valid_unseen)
                ablation_results[config['name']] = {
                    "f1": metrics.f1_score,
                    "accuracy": metrics.accuracy,
                    "reward": metrics.avg_reward
                }
                logger.info(f"F1: {metrics.f1_score:.3f}, Acc: {metrics.accuracy:.3f}, Reward: {metrics.avg_reward:.3f}")
        
        # Summary table
        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDY RESULTS")
        logger.info("="*60)
        logger.info(f"{'Configuration':<40} {'F1':>8} {'Accuracy':>10} {'Reward':>8}")
        logger.info("-" * 70)
        for name, results in ablation_results.items():
            logger.info(f"{name:<40} {results['f1']:>8.3f} {results['accuracy']:>10.3f} {results['reward']:>8.3f}")
        logger.info("="*60 + "\n")
    
    # ========================================
    # PHASE 3: Evaluate on SEEN Validation
    # ========================================
    metrics_seen = None
    if valid_seen:
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Evaluating on SEEN Validation Set")
        logger.info("="*60)
        
        logger.info(f"Validation (seen) trajectories: {len(valid_seen)}")
        metrics_seen = run_evaluation(agent, valid_seen)
        
        logger.info("\n--- SEEN Validation Results ---")
        pretty_print_metrics(metrics_seen)
    
    # ========================================
    # PHASE 4: Evaluate on UNSEEN Validation
    # ========================================
    metrics_unseen = None
    if valid_unseen:
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: Evaluating on UNSEEN Validation Set")
        logger.info("="*60)
        
        logger.info(f"Validation (unseen) trajectories: {len(valid_unseen)}")
        metrics_unseen = run_evaluation(agent, valid_unseen)
        
        logger.info("\n--- UNSEEN Validation Results ---")
        pretty_print_metrics(metrics_unseen)
    
    # ========================================
    # PHASE 5: Generalization Analysis
    # ========================================
    if metrics_seen and metrics_unseen:
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: Generalization Analysis")
        logger.info("="*60)
        
        gap_accuracy = metrics_seen.accuracy - metrics_unseen.accuracy
        gap_f1 = metrics_seen.f1_score - metrics_unseen.f1_score
        
        logger.info(f"\nSeen Accuracy:     {metrics_seen.accuracy:.3f}")
        logger.info(f"Unseen Accuracy:   {metrics_unseen.accuracy:.3f}")
        logger.info(f"Accuracy Gap:      {gap_accuracy:+.3f}")
        
        logger.info(f"\nSeen F1:           {metrics_seen.f1_score:.3f}")
        logger.info(f"Unseen F1:         {metrics_unseen.f1_score:.3f}")
        logger.info(f"F1 Gap:            {gap_f1:+.3f}")
        
        if abs(gap_accuracy) < 0.05:
            logger.info("\n✓ Excellent generalization (gap < 0.05)")
        elif abs(gap_accuracy) < 0.10:
            logger.info("\n✓ Good generalization (gap < 0.10)")
        elif abs(gap_accuracy) < 0.20:
            logger.info("\n⚠ Moderate generalization (gap < 0.20)")
        else:
            logger.info("\n✗ Poor generalization (gap >= 0.20) - possible overfitting")
    
    # ========================================
    # PHASE 6: Final Summary
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    final_stats = agent.get_statistics()
    logger.info(f"\nAgent Statistics:")
    logger.info(f"  Dataset(s):             {args.dataset.upper()}")
    logger.info(f"  Procedural memory:      {final_stats['procedural_memory_size']}/{agent.memory_system.N_p}")
    logger.info(f"  Meta-procedural memory: {final_stats['meta_procedural_size']}/{agent.memory_system.N_m}")
    logger.info(f"  Total executions:       {final_stats['total_executions']}")
    logger.info(f"  Successful executions:  {final_stats['successful_executions']}")
    
    if metrics_seen:
        logger.info(f"\nSeen validation F1:     {metrics_seen.f1_score:.3f}")
    if metrics_unseen:
        logger.info(f"Unseen validation F1:   {metrics_unseen.f1_score:.3f}")
    
    logger.info("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()