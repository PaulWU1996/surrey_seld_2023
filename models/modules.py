import functools
import operator
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class FeatureExtraction(nn.Module):
    def __init__(self,
                 embedding_dim: int = 51,
                 num_sources_output: int = 3,
                 dropout_rate: float = 0.0) -> None:
        """
            FeatureExtraaction is for extracting features from audio features,
            which includes mel spectrogram and pha map
        """

        super(FeatureExtraction, self).__init__()

        self.conv_network = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )

        flatten_dim = functools.reduce(operator.mul, list(
            self.conv_network(torch.rand(1, 4, 577, 50)).shape[slice(1, 4, 2)]))

        self.fc_network = nn.Sequential(
            nn.Linear(flatten_dim, 128), # flatten_dim has been replaced by the out
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.embedding_output = nn.Linear(128, (embedding_dim - 1) * num_sources_output)
        # self.observation_noise_output = nn.Linear(128, embedding_dim * num_sources_output) # Not sure this is essential or not

        self.embedding_dim = embedding_dim
        self.num_sources_output = num_sources_output

    def forward(self, audio_features: Tensor):
        
        batch_size, frames, mel_pha, channel = audio_features.shape 
        audio_features = audio_features.permute(0,3,2,1)

        output = self.conv_network(audio_features) # [batch,channel_n, frames, mel_pha_n] =[128,64,50,10]
        output = output.permute(0,2,1,3).flatten(2) # [128,50,640]

        output = self.fc_network(output)

        embeddings = self.embedding_output(output)
        embeddings = embeddings.view(batch_size, frames, self.num_sources_output, -1)

        positional_encodings = torch.linspace(0, 1, frames, device=audio_features.device)[None, ..., None, None]
        positional_encodings = positional_encodings.repeat((batch_size, 1, self.num_sources_output, 1))
        
        

        embeddings = torch.cat((embeddings, positional_encodings), dim=-1)

        return embeddings

import  data_module

data = data_module.T3DataModule(batch_size=1)
data.setup()
dataloader = data.train_dataloader()

samples = next(iter(dataloader))
feats, labels = samples

model = FeatureExtraction()
output = model(feats)
print(output.shape)