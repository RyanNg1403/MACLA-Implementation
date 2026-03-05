import logging
from typing import Dict, List, Optional

from .data_structures import MetaProcedure
from .memory import EnhancedHierarchicalMemorySystem

logger = logging.getLogger(__name__)


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
            beta=1,
        )

        logger.info(f"Extracted meta-procedure for task '{goal_meta}' with {len(procedure_sequence)} sub-procedures")
        return meta
