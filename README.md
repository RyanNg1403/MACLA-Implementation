# MACLA: Memory-Augmented Continual Learning Agent

[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-blue)](https://www.aamas2026.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **This repository is a study fork of the original MACLA implementation.**
> Its purpose is to understand the architecture and learning mechanisms of MACLA by reading and modularising the codebase. All credit for the research, ideas, and original implementation belongs to the authors of the AAMAS 2026 paper.

---

## Original Paper & Authors

**"MACLA: Memory-Augmented Continual Learning Agent"**
*Oral Presentation — 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2026)*

**Authors:** Saman Forouzandeh, Wei Peng, Parham Moradi, Xinghuo Yu, Mahdi Jalili
**Institution:** RMIT University, Melbourne, Australia
**Contact:** saman.forouzandeh@rmit.edu.au

```bibtex
@inproceedings{macla2026,
  title     = {MACLA: Memory-Augmented Continual Learning Agent},
  author    = {Forouzandeh, Saman and Peng, Wei and Moradi, Parham and Yu, Xinghuo and Jalili, Mahdi},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multi-Agent Systems},
  year      = {2026},
  publisher = {IFAAMAS},
  note      = {Oral Presentation}
}
```

Original repository: https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-Conference

---

## What is MACLA?

MACLA is a domain-agnostic continual learning framework for LLM-based agents. Instead of calling an LLM on every step, it extracts **reusable procedural knowledge** from past trajectories and retrieves it via Bayesian selection — reducing LLM calls by >85% while matching or exceeding ReAct-style baselines on four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCode-SQL).

Core components:

| Component | Role |
|-----------|------|
| **Hierarchical Memory** | Stores atomic, sequential, procedural, and meta-procedural knowledge |
| **Bayesian Selector** | Ranks procedures by expected utility under uncertainty |
| **Contrastive Refiner** | Improves procedure quality by contrasting success vs. failure contexts |
| **Meta-Procedural Learner** | Composes sub-procedures into higher-level skills |
| **Frozen LLM Reasoner** | Uses an Ollama-backed LLM only for goal discovery and extraction |

---

## Repository Structure

This fork refactors the original single-file `MACLA.py` (~2,600 lines) into a Python package for readability.

```
MACLA-Implementation/
├── MACLA.py              # CLI entry point (imports from macla/)
├── requirements.txt      # Dependencies
├── MACLA.pdf             # Original paper
└── macla/                # Modular package
    ├── __init__.py           # Public API re-exports
    ├── backends.py           # Ollama + SentenceTransformer init
    ├── data_structures.py    # Dataclasses (Procedure, MetaProcedure, etc.)
    ├── memory.py             # EnhancedHierarchicalMemorySystem
    ├── bayesian_selector.py  # BayesianProcedureSelector
    ├── contrastive.py        # ContrastiveRefinementEngine
    ├── meta_learner.py       # MetaProceduralLearner
    ├── llm_reasoner.py       # FrozenLLMReasoner (Ollama)
    ├── agent.py              # EnhancedMACLAAgent + LLMMACLAAgent
    ├── loaders.py            # ALFWorldLikeLoader, WebShopLoader,
    │                         # TravelPlannerLoader, SQLLoader
    ├── evaluator.py          # MACLAEvaluator
    └── utils.py              # train_test_split, build_agent_and_learn, etc.
```

No behaviour was changed — only file organisation.

---

## Installation

```bash
git clone https://github.com/RyanNg1403/MACLA-Implementation.git
cd MACLA-Implementation
pip install -r requirements.txt

# Ollama backend (required for LLM features)
curl https://ollama.ai/install.sh | sh
ollama pull llama2
```

---

## Quick Start

```python
from macla import LLMMACLAAgent, build_agent_and_learn, run_evaluation

# Build and train
agent = build_agent_and_learn(train_trajectories, llm_model="llama2")

# Evaluate
metrics = run_evaluation(agent, test_trajectories)
print(f"F1: {metrics.f1_score:.3f}  Reward: {metrics.avg_reward:.3f}")
```

**CLI:**

```bash
# TravelPlanner
python MACLA.py --dataset travelplanner --travelplanner_test_file <path>

# ALFWorld
python MACLA.py --dataset alfworld --train_file <path> --valid_seen_file <path>

# Run ablation studies
python MACLA.py --dataset alfworld --train_file <path> --ablation
```

---

## Acknowledgements

All research, algorithms, experimental results, and the original implementation are the work of **Forouzandeh et al.** This fork exists solely as a learning exercise to understand the system by reading and reorganising the code.
