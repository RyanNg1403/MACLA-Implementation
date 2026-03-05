import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .backends import _EMBED_AVAILABLE, _EMBEDDER, st_util
from .data_structures import ProceduralMemoryEntry, Procedure
from .memory import EnhancedHierarchicalMemorySystem

logger = logging.getLogger(__name__)


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

        for category, keywords in self.ontology.items():
            if any(k in obs_lower for k in keywords):
                return category

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

        if goal in self.memory_system.goal_index:
            candidates.update(self.memory_system.goal_index[goal])

        if not candidates:
            gwords = set(goal.lower().split("_"))
            for g, keys in self.memory_system.goal_index.items():
                if set(g.lower().split("_")) & gwords:
                    candidates.update(keys)

        context = self._extract_context(observation)
        for ctx, keys in self.memory_system.context_index.items():
            if context in ctx.lower() or ctx.lower() in context:
                candidates.update(keys)

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
