from typing import Any, Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn
import math

from .modules import FeatureExtraction, LinearGaussianSystem


class T3Model(pl.LightningModule):
    def __init__(self,
                 embedding_dim: int = 32,
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

        self.linear_gaussian_system = LinearGaussianSystem(state_dim=3, observation_dim=embedding_dim)
        self.prior_mean = torch.cat((
            torch.linspace(-math.pi, math.pi - 2 * math.pi / num_sources_output, num_sources_output).unsqueeze(-1),
            torch.zeros(num_sources_output).unsqueeze(-1)
        ), dim=-1)
        self.prior_covariance = torch.eye(2).unsqueeze(0).repeat((num_sources_output, 1, 1))



    def forward(self, audio_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Comments later
        """
        embeddings, observation_noise = self.feature_extraction(audio_features)

        posterior_mean = []
        posterior_covariance = []

        # for src_dix in range(self.num_sources_output):
        #     src_embeddiings = self.transformer(embeddings[:, 1, ...].permute(1,0,2)).permute(1,0,2) 

        for src_idx in range(self.num_sources_output):
            # Permutation is needed here, because the Transformer class requires sequence length in dimension zero.
            src_embeddings = self.transformer(embeddings[:, src_idx, ...].permute(1, 0, 2)).permute(1, 0, 2)
            src_observation_noise_covariance = torch.diag_embed(observation_noise[:, src_idx, ...])

            posterior_distribution = self.linear_gaussian_system(src_embeddings,
                                                                 src_observation_noise_covariance,
                                                                 prior_mean=self.prior_mean[src_idx, ...].to(self.device),
                                                                 prior_covariance=self.prior_covariance[src_idx, ...].to(self.device))

            posterior_mean.append(posterior_distribution[0])
            posterior_covariance.append(posterior_distribution[1].unsqueeze(-1))

        posterior_mean = torch.cat(posterior_mean, dim=-1).permute(0, 3, 1, 2)
        posterior_covariance = torch.cat(posterior_covariance, dim=-1).permute(0, 4, 1, 2, 3)

        return posterior_mean, posterior_covariance