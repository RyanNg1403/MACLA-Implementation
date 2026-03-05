import time
import logging

from .data_structures import ProceduralMemoryEntry

logger = logging.getLogger(__name__)


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
