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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=(1, 2)),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=(1, 2)),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=(1, 2)),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1),
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=(1, 2)),
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
    
class Feat2Track(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 target_num: int = 6) -> None:
        super().__init__()
        self.target_num = target_num
        self.linear = nn.Linear(embedding_dim, embedding_dim * target_num)

    def forward(self, x_doa, x_sed):
        x_sed = x_sed.permute(2,0,1) # t,N,dim
        x_doa = x_doa.permute(2,0,1)

        time_step, batch, channel = x_doa.shape

        x_sed = self.linear(x_sed).view(time_step,batch,self.target_num,-1).permute(2,1,3,0)
        x_doa = self.linear(x_doa).view(time_step,batch,self.target_num,-1).permute(2,1,3,0)
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
                x = x + pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x = x + pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x = x + self.pe[:, :, :x.shape[2]]
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
        output = self.encoder(x)
        return output

class CrossStitchTransFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.sed_transformer = TransformerEncoder()
        self.doa_transformer = TransformerEncoder()

        self.stitch = nn.Parameter(
                torch.FloatTensor(512,2,2).uniform_(0.1,0.9)
            )

    def forward(self, x_sed, x_doa):
        """
            x: List of positional embeddings [sed, doa]
        """
        # x_sed = x[0]
        # x_doa = x[1]

        x_sed = self.sed_transformer(x_sed)
        x_doa = self.doa_transformer(x_doa)

        # cross-stitch
        x_sed = torch.einsum('c,tnc->tnc',self.stitch[:,0,0],x_sed) + torch.einsum('c,tnc->tnc',self.stitch[:,0,1],x_doa)

        x_doa = torch.einsum('c,tnc->tnc',self.stitch[:,1,0],x_sed) + torch.einsum('c,tnc->tnc',self.stitch[:,1,1],x_doa)

        return x_sed, x_doa

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_1 = CrossStitchTransFormer()
        self.encoder_2 = CrossStitchTransFormer()
        self.encoder_3 = CrossStitchTransFormer()

    def forward(self, x_sed, x_doa):
        x_sed, x_doa = self.encoder_1(x_sed, x_doa)
        x_sed, x_doa = self.encoder_2(x_sed,x_doa)
        x_sed, x_doa = self.encoder_3(x_sed,x_doa)

        return x_sed, x_doa

"""
    SEDPrediction version 1 is off
"""
# class SedPrediction(nn.Module):
#     def __init__(self,
#                 embedding_dim: int = 512,
#                 target_num: int = 5,
#                 cls_dim: int = 13) -> None:
#         super().__init__()
#         self.target_num = target_num
#         self.cls_num = cls_dim
#         self.linear = nn.Linear(embedding_dim, embedding_dim*target_num)
#         self.fc = nn.Linear(embedding_dim,cls_dim,bias=True)
#         self.final_act_sed = nn.Sequential() # nn.Sigmoid() # 

#     def forward(self, x):
#         batch, time_step, channel = x.shape
#         x = self.linear(x)
#         x = x.view(batch,time_step,self.target_num,-1)
#         # output = torch.Tensor(batch,time_step,self.target_num,self.cls_num) # possible error?

#         output = []
#         for src_id in range(self.target_num):
#             src = x[:,:,src_id]
#             src = self.fc(src)
#             src = self.final_act_sed(src)
#             # output[:,:,src_id] = src
#             output.append(src)

#         output = torch.stack(output).permute(1,2,0,3)
#         return output
class SedPrediction(nn.Module):
    def __init__(self,
                embedding_dim: int = 512,
                cls_dim: int = 13) -> None:
        super().__init__()
        self.cls_num = cls_dim
        self.fc = nn.Linear(embedding_dim,cls_dim,bias=True)
        self.final_act_sed = nn.Sequential() # nn.Sigmoid() # 

    def forward(self, x):
        batch, time_step, channel = x.shape

        output = self.fc(x)
        output = self.final_act_sed(output)
        return output
    

class ObservationNoise(nn.Module):
    def __init__(self,
                target_num: int = 6,
                embedding_dim: int = 512,
                state_dim: int = 3) -> None:
        super().__init__()
        self.target_num = target_num
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        # self.linear_1 = nn.Linear(embedding_dim, self.embedding_dim*self.target_num)
        self.linear_2 = nn.Linear(embedding_dim, state_dim)

    def forward(self,x):
        x = x.permute(2,0,1)

        # time_step, batch, channel = x.shape
        # x = self.linear_1(x).view(time_step,batch,self.target_num,-1)
        noise = torch.exp(x)
        # noise = noise.permute(1,2,0,3) # (n,num,t,c)
        noise = self.linear_2(noise) # (t,n,state_dim)
        noise = noise.permute(1,0,2)
        return noise #(n,t,state_dim)

class LinearGaussianSystem(nn.Module):
    def __init__(self,
                state_dim: int = 3,
                observation_dim: int = 512,
                prior_mean = None,
                prior_covariance = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim

        self.linear = nn.Linear(observation_dim,state_dim)

        # Observation model params
        self.observation_matrix = nn.Parameter(torch.randn(state_dim,state_dim),requires_grad=True)
        self.observation_bias = nn.Parameter(torch.randn(state_dim),requires_grad=True)

        # Prior information
        if prior_mean is None:
            prior_mean = torch.zeros(state_dim)

        if prior_covariance is None:
            prior_covariance = torch.eye(state_dim)

        self.prior_mean = nn.Parameter(prior_mean, requires_grad=True)
        self.prior_covariance = nn.Parameter(prior_covariance,requires_grad=True)

    def forward(self,observation, prior_mean, prior_covariance, observation_noise):
         
            # Prior informaton
            if prior_mean is not None:
                self.prior_mean = nn.Parameter(prior_mean.to(observation.device),requires_grad=True)

            if prior_covariance is not None:
                self.prior_covariance = nn.Parameter(prior_covariance.to(observation.device),requires_grad=True)
            
            if observation_noise is None:
                observation_noise = 1e-6 * torch.eye(self.state_dim,requires_grad=True).to(observation.device)


            innovation_covariance = self.observation_matrix @ self.prior_covariance @ self.observation_matrix.t() + observation_noise
            posterior_covariance = torch.inverse(torch.inverse(self.prior_covariance) + innovation_covariance)

            tmp = self.linear(observation)
            residual = self.observation_matrix @ (tmp - self.observation_bias).unsqueeze(-1)
            posterior_mean = posterior_covariance @ (torch.inverse(self.prior_covariance) @ self.prior_mean.unsqueeze(-1) + residual)

            return posterior_mean #, posterior_covariance


"""
    Version 1 of DoaPrediction is off
"""
# class DoaPrediction(nn.Module):
#     def __init__(self,
#                 target_num: int = 10,
#                 embedding_dim: int = 512,
#                 state_dim: int = 3,
#                 ) -> None:
#         super().__init__()
#         self.target_num = target_num
#         self.embedding_dim = embedding_dim
#         self.state_dim = state_dim
#         # self.observation_noise = observation_noise

#         self.linear = nn.Linear(512, self.embedding_dim*self.target_num)

#         self.linear_gaussian_system = LinearGaussianSystem()

#         self.prior_mean = torch.clamp(torch.randn(target_num,3),-0.75,0.75)
#         self.prior_covariance = torch.eye(3).unsqueeze(0).repeat((target_num,1,1))

#     def forward(self,x,noise):
#         batch, time_step, channel = x.shape

#         x = self.linear(x).view(batch,time_step,self.target_num,-1) # (N, T, target,C)
        
#         # posterior_mean = []
#         # posterior_covariance = []

#         # output = torch.Tensor(batch,time_step,self.target_num,self.state_dim)

#         output = []

#         for src_idx in range(self.target_num):
#             src = x[:,:,src_idx]
#             obs_noise_cov = torch.diag_embed(noise[:,src_idx].detach())

#             # self.linear_gaussian_system = (prior_mean=self.prior_mean[src_idx,...],
#             #                         prior_covariance=self.prior_covariance[src_idx,...])
#             posterior_distribution = self.linear_gaussian_system(observation=src, 
#                                                                  prior_mean=self.prior_mean[src_idx,...], 
#                                                                  prior_covariance=self.prior_covariance[src_idx,...], 
#                                                                  observation_noise=obs_noise_cov)

#             # output[:,:,src_idx] = posterior_distribution[0].squeeze(-1)
#             output.append(posterior_distribution[0].squeeze(-1))

#             # posterior_mean.append(posterior_distribution[0])
#             # posterior_covariance.append(posterior_distribution[1].unsqueeze(-1))

#         output = torch.stack(output).permute(1,2,0,3)
#         return output # (N,T,num,state_d)
class DoaPrediction(nn.Module):
    def __init__(self,
                embedding_dim: int = 512,
                state_dim: int = 3,
                ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.linear_gaussian_system = LinearGaussianSystem()

        # self.prior_mean = torch.clamp(torch.randn(1,3),-0.75,0.75)
        # self.prior_covariance = torch.eye(3).unsqueeze(0).repeat((1,1,1))

    def forward(self,x,noise=None):

 
            batch, time_step, channel = x.shape

            if noise is None:
                obs_noise_cov = 1e-6 * torch.eye(self.state_dim, device=x.device,requires_grad=False)
            else:
                obs_noise_cov = 1e-3 * torch.diag_embed(noise)
            output = self.linear_gaussian_system(observation=x, 
                                                prior_mean=None, 
                                                prior_covariance=None, 
                                                observation_noise=obs_noise_cov)

            # output = torch.stack(output).permute(1,2,0,3)
            output = output.transpose(2,3).squeeze(2)
            return output # (N,T,num,state_d)