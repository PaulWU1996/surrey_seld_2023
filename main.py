from data_handler import data_module
from models import T3Model

import torch.nn as nn

data = data_module.T3DataModule(batch_size=1)
data.setup()
dataloader = data.train_dataloader()

samples = next(iter(dataloader)) 
feats, labels = samples

# feats, labels = data.train_set.__getitem__(150)

model = T3Model()
target_cls,doa,noise = model(feats)

print(1)