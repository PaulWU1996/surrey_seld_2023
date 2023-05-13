from typing import Any, Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn

from .modules import FeatureExtraction


class T3Model(pl.LightningModule):
    def __init__(self,
                 embedding_dim: int = 51,
                 num_sources_output: int = 3,
                 feature_extraction_dropout: float = 0.0,
                 transformer_num_heads: int = 8,
                 transformer_num_layers: int = 3,
                 transformer_feedforward_dim: int = 1024,
                 transformer_dropout: float = 0.1,
                 learning_rate: float = 0.05,
                 num_epochs_warmup: int = 5,
                 ) -> None:
        # add the args later
        super(T3Model, self).__init__()

        self.num_sources_output = num_sources_output
        self.embedding_dim = embedding_dim

        self.feature_extraction = FeatureExtraction(embedding_dim=self.embedding_dim,
                                                    num_sources_output=self.num_sources_output,
                                                    dropout_rate=feature_extraction_dropout)
        
        transformer_layers = nn.TransformerEncoderLayer(
            embedding_dim, transformer_num_heads, transformer_feedforward_dim, transformer_dropout
        )
        self.transformer = nn.TransformerEncoder(transformer_layers, transformer_num_layers)

    def forward(self, audio_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Comments later
        """
        embeddings = self.feature_extraction(audio_features)

        for src_dix in range(self.num_sources_output):
            src_embeddiings = self.transformer(embeddings[:, 1, ...].permute(1,0,2)).permute(1,0,2) 

        return 0