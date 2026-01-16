# MACLA: Memory-Augmented Continual Learning Agent

[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-blue)](https://www.aamas2026.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation of "MACLA: Memory-Augmented Continual Learning Agent"**  
*Accepted as Oral Presentation at The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2026)*

---

## 🎯 Overview

MACLA is a domain-agnostic continual learning framework that enables LLM-based agents to learn from experience through structured procedural memory. Unlike traditional approaches that require extensive fine-tuning or per-step LLM reasoning, MACLA achieves state-of-the-art performance by extracting, refining, and composing reusable procedural knowledge from successful and failed trajectories.

### Key Features

- **🧠 Bayesian Selection**: Uncertainty-aware procedure ranking that balances exploitation and exploration
- **🔄 Procedural Memory**: Learns reusable skills with automatic extraction and parameterization
- **🏗️ Meta-Procedures**: Hierarchical composition for complex multi-step reasoning
- **⚖️ Contrastive Learning**: Quality refinement through discriminative pattern extraction
- **🚀 Minimal LLM Usage**: Reduces LLM calls by >85% compared to ReAct (2 vs 16-20 calls per episode)
- **📊 Superior Generalization**: Achieves positive generalization gap (+3.1%) on unseen tasks

---

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8 or higher required
python --version
```

### Option 1: Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/MACLA.git
cd MACLA

# Install dependencies
pip install -r requirements.txt

# Install Ollama for LLM backend (if not already installed)
curl https://ollama.ai/install.sh | sh
```

### Option 2: Development Setup

```bash
# Create virtual environment
python -m venv macla_env
source macla_env/bin/activate  # On Windows: macla_env\Scripts\activate

# Install in editable mode
pip install -e .

# Install optional dependencies for visualization
pip install matplotlib seaborn
```

### Dependencies

**Core Requirements:**
```txt
ollama>=0.1.23
sentence-transformers>=2.2.0
numpy>=1.21.0
scipy>=1.7.0
```

**Optional:**
```txt
matplotlib>=3.5.0  # For visualization
seaborn>=0.12.0    # For advanced plotting
```

### Ollama Model Setup

```bash
# Pull the Llama-2-7B model
ollama pull llama2:7b

# Verify installation
ollama list
```

---

## 🚀 Quick Start

### Basic Usage

```python
from MACLA import LLMMACLAAgent, load_trajectories

# Initialize agent
agent = LLMMACLAAgent(
    N_a=2000,  # Atomic memory capacity
    N_p=200,   # Procedural memory capacity
    N_m=50,    # Meta-procedural memory capacity
    llm_model="llama2:7b",
    use_llm=True
)

# Load training data
train_trajectories = load_trajectories("data/alfworld_train.json")

# Learn from trajectories
agent.learn_from_trajectories(train_trajectories)

# Evaluate on new tasks
validation_data = load_trajectories("data/alfworld_valid_unseen.json")
metrics = agent.evaluate(validation_data)

print(f"Success Rate: {metrics.accuracy:.1%}")
print(f"F1 Score: {metrics.f1_score:.3f}")
print(f"Avg Reward: {metrics.avg_reward:.3f}")
```

### Command-Line Interface

```bash
# Train on ALFWorld dataset
python MACLA.py --dataset alfworld --llm-model llama2:7b

# Train with ablation studies
python MACLA.py --dataset alfworld --ablation

# Train on multiple datasets
python MACLA.py --dataset alfworld,webshop --llm-model llama2:7b

# Run without LLM (testing mode)
python MACLA.py --dataset alfworld --no-llm
```

### Datasets

MACLA supports multiple benchmarks out-of-the-box:

- **ALFWorld**: Embodied household tasks (pick-and-place, cooling, heating, etc.)
- **WebShop**: E-commerce product search and selection
- **TravelPlanner**: Multi-constraint travel planning
- **InterCodeSQL**: Database query generation

Place your data files in the `data/` directory with the following structure:

```
data/
├── alfworld_train.json
├── alfworld_valid_seen.json
├── alfworld_valid_unseen.json
├── webshop_train.json
└── ...
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     MACLA Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   LLM Core   │───▶│   Bayesian   │───▶│  Procedural  │  │
│  │  (Frozen)    │    │   Selector   │    │    Memory    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         │                    ▼                    │          │
│         │            ┌──────────────┐            │          │
│         └───────────▶│  Contrastive │◀───────────┘          │
│                      │   Refiner    │                        │
│                      └──────────────┘                        │
│                             │                                │
│                             ▼                                │
│                      ┌──────────────┐                        │
│                      │Meta-Procedure│                        │
│                      │   Learner    │                        │
│                      └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Memory Hierarchy

1. **Atomic Memory** (𝑁_𝑎 = 2000): Raw action-observation pairs
2. **Sequential Memory** (𝑁_𝑠 = 100): Trajectory segments
3. **Procedural Memory** (𝑁_𝑝 = 200): Parameterized, reusable skills
4. **Meta-Procedural Memory** (𝑁_𝑚 = 50): Hierarchical compositions

### Learning Pipeline

```
Trajectory → LLM Segmentation → Procedure Extraction → 
Bayesian Update → Contrastive Refinement → Meta-Procedure Formation
```

---

## 📊 Experimental Results

### Ablation Studies

| Configuration | Seen | Unseen | Proc. Count | Reuse Rate | LLM Calls |
|--------------|------|--------|-------------|------------|-----------|
| **Full MACLA** | 87.2 | 90.3 | 187 | 78% | 6.2 |
| w/o Bayesian Selection | 79.4 | 81.2 | 189 | 62% | 8.4 |
| w/o Contrastive | 83.6 | 85.7 | 201 | 71% | 6.8 |
| w/o Meta-Procedures | 81.2 | 78.4 | 193 | 65% | 9.1 |
| w/o Ontology | 82.8 | 84.1 | 185 | 74% | 6.5 |

**Key Findings:**
- Bayesian selection provides largest gain (–9.1% without)
- Meta-procedures critical for unseen tasks (–11.9% without)
- All components contribute synergistically

### Memory Capacity Analysis

| Capacity | Actual Proc. | Seen | Unseen | Avg α/(α+β) |
|----------|--------------|------|--------|-------------|
| 25 / 5 | 25 | 68.3 | 64.1 | 0.61 |
| 50 / 10 | 50 | 76.5 | 74.2 | 0.68 |
| 100 / 20 | 98 | 83.1 | 85.6 | 0.74 |
| **200 / 50** | **187** | **87.2** | **90.3** | **0.79** |
| 300 / 75 | 203 | 87.1 | 90.1 | 0.79 |

*Optimal capacity: 150-200 procedures with diminishing returns beyond*

### Learning Dynamics

**Three Emergent Phases:**
1. **Exploration** (0-570 trajectories): 15% → 45% success, 70 procedures extracted
2. **Consolidation** (571-1,425 trajectories): 45% → 82% success, contrastive refinement activates
3. **Exploitation** (1,426-2,851 trajectories): 82% → 87.2% success, memory saturation

---

## 🔬 Advanced Usage

### Custom Bayesian Priors

```python
# Use informed priors based on domain knowledge
agent = LLMMACLAAgent(N_p=200, N_m=50)
agent.bayesian_selector.set_prior(alpha_0=3.2, beta_0=1.8)
```

### Contrastive Learning Configuration

```python
# Adjust refinement thresholds
agent.contrastive_refiner.set_threshold(
    min_successes=5,    # Minimum successful examples
    min_failures=5,     # Minimum failed examples
    similarity_threshold=0.85
)
```

### Meta-Procedure Policies

```python
# Custom composition policies
meta_proc = MetaProcedure(
    goal_meta="compound_task",
    preconditions_meta=["chilled", "object"],
    sub_procedures=["cooling", "placement"],
    composition_policy={
        "trigger": "chilled_modifier",
        "order": "sequential",
        "dependencies": {"placement": ["cooling"]}
    }
)
agent.memory_system.add_meta_procedure(meta_proc)
```

### Export and Import Memory

```python
# Save learned procedures
agent.save_memory("checkpoints/alfworld_memory.pkl")

# Load pre-trained memory
agent.load_memory("checkpoints/alfworld_memory.pkl")
```

---

## 📝 Citation

If you use MACLA in your research, please cite our paper:

```bibtex
@inproceedings{macla2026,
  title={MACLA: Memory-Augmented Continual Learning Agent},
  author={Forouzandeh, Saman and Peng, Wei and Moradi, Parham and  Yu, Xinghuo and Jalili, Mahdi},
  booktitle={Proceedings of the 25th International Conference on Autonomous Agents and Multi-Agent Systems},
  year={2026},
  publisher={IFAAMAS},
  note={Oral Presentation}
}
```

---

## 📂 Project Structure

```
MACLA/
├── MACLA.py                 # Main implementation
├── README.md                # This file
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
├── data/                    # Dataset directory
│   ├── alfworld_train.json
│   ├── webshop_train.json
│   └── ...
├── checkpoints/             # Saved models
├── results/                 # Experimental outputs
├── docs/                    # Additional documentation
│   ├── MACLA_Appendix.pdf  # Detailed analyses
│   └── API_Reference.md    # API documentation
└── tests/                   # Unit tests
    ├── test_memory.py
    ├── test_bayesian.py
    └── test_contrastive.py
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_memory.py -v

# Run with coverage
pytest --cov=MACLA tests/
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
sudo systemctl restart ollama
```

**2. Memory Overflow**
```python
# Reduce memory capacities
agent = LLMMACLAAgent(N_a=1000, N_p=100, N_m=25)
```

**3. Slow Performance**
```python
# Disable semantic embeddings for faster startup
import os
os.environ["MACLA_EMBED_MODEL"] = "none"
```

**4. CUDA Out of Memory**
```bash
# Use 4-bit quantization with Ollama
ollama run llama2:7b --gpu-layers 32 --context-size 2048
```

---

## 🛣️ Roadmap

- [ ] Support for GPT-4 and Claude-3 backends
- [ ] Multi-modal procedure learning (vision + language)
- [ ] Distributed memory sharing across agents
- [ ] Real-time learning in embodied environments
- [ ] Neural-symbolic hybrid memory
- [ ] Automated hyperparameter tuning
- [ ] Web interface for visualization
- [ ] Docker containerization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AAMAS 2026** reviewers for valuable feedback
- **ALFWorld**, **WebShop**, **TravelPlanner**, and **InterCodeSQL** benchmark teams
- **Anthropic** for Claude API access during development
- **Meta AI** for Llama-2 model
- **Sentence-Transformers** team for embedding models

---

## 📧 Contact

For questions, issues, or collaboration:

- **Primary Contact**: [Saman Forouzandeh] - [saman.forouzandeh@rmit.edu.au]
- **Institution**: RMIT University, Melbourne, Australia
---


