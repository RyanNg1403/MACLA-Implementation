import time
import logging
from collections import defaultdict, deque
from typing import Dict, List, Set

from .data_structures import (
    AtomicMemoryEntry,
    Procedure,
    MetaProcedure,
    ContrastiveContext,
    ProceduralMemoryEntry,
)

logger = logging.getLogger(__name__)


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
