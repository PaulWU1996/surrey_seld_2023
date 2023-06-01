# from data_handler import data_module
# from models import T3Model

# import torch.nn as nn

# data = data_module.T3DataModule(batch_size=1)
# data.setup()
# dataloader = data.train_dataloader()

# samples = next(iter(dataloader)) 
# feats, labels = samples

# # feats, labels = data.train_set.__getitem__(150)

# model = T3Model()
# target_cls,doa,noise = model(feats)

# from utls import sedl_loss

# print(sedl_loss((target_cls,doa,noise),labels))


# from models import T3Model
from data_handler import data_module

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")
import torch

import torch.nn as nn

import numpy as np
from models import EINPLCST

data = np.load('/vol/research/VS-Work/PW00391/surrey_seld_2023/audio.npy').astype(np.float32)
data = data.reshape(1,7,160,256)
data = torch.from_numpy(data)

model = EINPLCST().float()
output = model(data)
print(output)

# class ClassificationNet(nn.Module):
#     def __init__(self):
#         super(ClassificationNet, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(50 * 32, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 13)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x

# if __name__ == "__main__":

#     torch.set_float32_matmul_precision("medium")

#     model = T3Model().load_from_checkpoint("/mnt/fast/nobackup/users/pw00391/dcase/512.ckpt")

#     new_classifier = ClassificationNet()
#     model.classifier = new_classifier

#     data = data_module.T3DataModule(batch_size=512)
#     checkpoint_callback = ModelCheckpoint(monitor="val_loss")

#     trainer = pl.Trainer(
#         default_root_dir='/mnt/fast/nobackup/users/pw00391/dcase',
#         min_epochs=0,
#         max_epochs=200,
#         accelerator="auto",
#         enable_checkpointing=True,
#         # fast_dev_run=True
#     )

#     trainer.tune(model=model,datamodule=data)
#     trainer.fit(model=model,datamodule=data)
