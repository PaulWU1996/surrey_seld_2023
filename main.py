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


from models import T3Model
from data_handler import data_module

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")
import torch

if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    model = T3Model()
    data = data_module.T3DataModule(batch_size=256)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        default_root_dir='/mnt/fast/nobackup/users/pw00391/',
        min_epochs=0,
        max_epochs=200,
        accelerator="auto",
        enable_checkpointing=True,
        # fast_dev_run=True
    )

    trainer.tune(model=model,datamodule=data)
    trainer.fit(model=model,datamodule=data)
