from data_handler import data_module
from models import FeatureExtraction

import torch.nn as nn

data = data_module.T3DataModule(batch_size=1)
data.setup()
dataloader = data.train_dataloader()

samples = next(iter(dataloader))
feats, labels = samples

model = FeatureExtraction()
output = model(feats)
print(output.shape)

transformer_layers = nn.TransformerEncoderLayer(51,8,1024,0.1)
transformer = nn.TransformerEncoder(transformer_layers,3)

print(1)