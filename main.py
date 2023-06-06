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
from utls import Losses #, Metrics

import torch.optim as optim

from pytorch_lightning.loggers import WandbLogger

# if __name__ == "__main__":
#     torch.autograd.set_detect_anomaly(True)
#     torch.set_float32_matmul_precision("medium")
#     wandb_logger = WandbLogger(project='EINPLCST_without_Matric', entity='')

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = EINPLCST(target_num=6).to(device)


#     data = data_module.UserDataModule(batch_size=64)
#     checkpoint_callback = ModelCheckpoint(monitor="val_loss")

#     trainer = pl.Trainer(
#         default_root_dir='/mnt/fast/nobackup/users/pw00391/dcase',
#         min_epochs=0,
#         max_epochs=90,
#         accelerator="auto",
#         enable_checkpointing=True,
#         logger=wandb_logger,
#         # fast_dev_run=True
#     )

#     # trainer.tune(model=model,datamodule=data)
#     trainer.fit(model=model,datamodule=data)






# audio = np.load('/mnt/fast/nobackup/users/pw00391/dcase/audio.npy').astype(np.float32)
# target = np.load('/mnt/fast/nobackup/users/pw00391/dcase/target.npy').astype(np.float32)
# # audio = audio.reshape(1,7,160,256)
# audio = torch.from_numpy(audio)
# # target = target.shape(1,40,6,4,13)
# target = torch.from_numpy(target)

# model = EINPLCST().float()
# output = model(audio)

# SED = np.load('/mnt/fast/nobackup/users/pw00391/dcase/SED.npy').astype(np.float32)
# DOA = np.load('/mnt/fast/nobackup/users/pw00391/dcase/DOA.npy').astype(np.float32)

# audio = torch.from_numpy(audio)

audio = torch.from_numpy(np.load('/mnt/fast/nobackup/users/pw00391/dcase/audio.npy').astype(np.float32))
target = torch.from_numpy(np.load('/mnt/fast/nobackup/users/pw00391/dcase/target.npy').astype(np.float32))

model = EINPLCST()

output = model(audio)

criterion = Losses(mode='train')
val_loss = Losses(mode='val')
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.autograd.set_detect_anomaly(True):
    optimizer.zero_grad()  
    loss_dict = criterion.calculate(pred=output,target=target)
    loss_dict['all'].backward()

    # val_loss_dct = val_loss.eval_calculate(pred=output,target=target)

    # print(1)

# try:
#     loss_dict['all'].backward()
# except Exception as e:
#     print(e)


# loss_dict = criterion.eval_calculate(pred=output,target=target)
# # loss_dict['all'].backward()  # 计算梯度

# metric = Metrics()

# metric.calculate(loss_dict['updated_target'],target)

# # optimizer.step()

# print(1)

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
