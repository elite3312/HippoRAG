from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class JinaEmbeddingsV2SmallEnModel(BaseEmbeddingModel):
    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.embedding_model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
        self.embedding_model.eval()
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        with torch.no_grad():
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        return embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                results.append(self.encode(texts[i:i + batch_size]))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()

        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
