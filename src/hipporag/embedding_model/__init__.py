from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .BgeSmallEnV15 import BgeSmallEnV15Model
from .AllMiniLML6V2 import AllMiniLML6V2EmbeddingModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "BAAI/bge-small-en-v1.5" in embedding_model_name:
        return BgeSmallEnV15Model
    elif "sentence-transformers/all-MiniLM-L6-v2" in embedding_model_name:
        return AllMiniLML6V2EmbeddingModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
