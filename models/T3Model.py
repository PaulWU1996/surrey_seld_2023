from typing import Any
import numpy as np
import torch
import pytorch_lightning as pl
import Tensor
import torch.nn as nn


class T3Model(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)