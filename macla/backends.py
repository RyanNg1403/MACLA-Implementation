import os
import logging

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
    ollama = None
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
