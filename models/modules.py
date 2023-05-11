import functools
import operator
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class FeatureExtraction(nn.Module):
    def __init__(self,
                 embedding_dim: int = 32,
                 num_sources_output: int = 3,
                 dropout_rate: float = 0.0) -> None:
        """
            FeatureExtraaction is for extracting features from audio features,
            which includes mel spectrogram and pha map
        """

        super(FeatureExtraction, self).__init__()

        self.conv_pha

        self.conv_mel

    # def forward(self, audio_features):
        
    #     batch_size, 
