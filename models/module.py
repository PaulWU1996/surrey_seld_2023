import functools
import operator
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple
import numpy as np


# define the conv network sharing its params
class ConvNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.in_channels = 7 # log_mel + intensity


        # SED Conv Block
        self.sed_conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        # DOA conv block
        self.doa_conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=7,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        # cross-stitch to share params
        self.stitch = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(64, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
        ])

    def forward(self, x):
        """
        x: input audio feature, the input dim (nctf)
            n - batch
            c - channel
            t - time_step
            f - freq_bins
        """
        
        # different input for different tasks
        x_sed = x[:,:4]
        x_doa = x

        # conv and cross-stitch
        x_sed = self.sed_conv_block1(x_sed)
        x_doa = self.doa_conv_block1(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block4(x_sed)
        x_doa = self.doa_conv_block4(x_doa)
        # (nctf) = (n,512,10,16)

        x_sed = x_sed.mean(dim=3) # (N, C, T)
        x_doa = x_doa.mean(dim=3) # (N, C, T)
        # (nct) = (n,512,10)
        
        return x_sed, x_doa
    

class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()
        
        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead= 8, dim_feedforward=1024,dropout=0.2
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,num_layers=2
        )

    def forward(self, x):
        return self.encoder(x)

class CrossStitchTransFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.sed_transformer = TransformerEncoder()
        self.doa_transformer = TransformerEncoder()

        self.stitch = nn.ParameterList(
            nn.Parameter(
                torch.FloatTensor(2,2).uniform_(0.1,0.9)
            )
        )

    def forward(self, x):
        """
            x: List of positional embeddings [sed, doa]
        """
        x_sed = x[0]
        x_doa = x[1]

        x_sed = self.sed_transformer(x_sed)
        x_doa = self.doa_transformer(x_doa)



class LinearGaussianSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class EIN_PLCST(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.convs = ConvNetwork()
        self.positional_encoder = PositionalEncoding()
        self.cross_transformer = CrossStitchTransFormer()

    def forward(self, x):
        """
        x: audio feature
        """
        x_sed, x_doa = self.convs(x)
        