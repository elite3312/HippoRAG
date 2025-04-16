from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "voyage-multilingual-2" in embedding_model_name:
        return VoyageMultilingual2EmbeddingModel
    elif "jinaai/jina-embeddings-v2-small-en" in embedding_model_name:
        return JinaEmbeddingsV2SmallEnEmbeddingModel
    elif "BAAI/bge-en-icl" in embedding_model_name:
        return BgeEnIclEmbeddingModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
