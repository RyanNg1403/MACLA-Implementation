import json
import logging
from typing import List

from .backends import _OLLAMA_AVAILABLE, ollama

logger = logging.getLogger(__name__)


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

    def segment_trajectory(self, trajectory: dict) -> str:
        prompt = (
            "Segment the following trajectory into logical steps as a JSON array of {step, action, observation}:\n"
            f"{json.dumps(trajectory, ensure_ascii=False)}"
        )
        return self._generate(prompt, max_tokens=300, temperature=0.3)

    def extract_procedure_components(self, segment: dict) -> str:
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
