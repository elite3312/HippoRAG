from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .BgeSmallEnV15 import BgeSmallEnV15Model
from .SfrEmbeddingMistral import SfrEmbeddingMistralModel
from .E5Mistral7B import E5Mistral7BInstructEmbeddingModel
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
    elif "Salesforce/SFR-Embedding-Mistral" in embedding_model_name:
        return SfrEmbeddingMistralModel
    elif "intfloat/e5-mistral-7b-instruct" in embedding_model_name:
        return E5Mistral7BInstructEmbeddingModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
