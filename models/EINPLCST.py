from typing import Any, Optional
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from .module import *
from utls import Losses
import pytorch_lightning as pl

class EINPLCST(pl.LightningModule):
    def __init__(self,
                target_num: int = 10,) -> None:
        super().__init__()

        self.target_num = target_num


        self.convs = ConvNetwork()
        self.observation_noise = ObservationNoise()
        
        self.positional_encoder = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=0.0)
        self.transformer = Transformer()
        self.sed_output = SedPrediction(target_num=self.target_num)
        self.doa_output = DoaPrediction(target_num=self.target_num)

    def forward(self, x):
        """
        x: audio feature
        """
        # Extracting embeddings by convs
        x_sed, x_doa = self.convs(x) # (N,C,T)

        # Extracting observation noise
        noise = self.observation_noise(x_doa) # (n,num,t,c)

        # Encoding positional information
        x_sed = self.positional_encoder(x_sed)
        x_doa = self.positional_encoder(x_doa) # (N,C,T)
        x_sed = x_sed.permute(2,0,1)
        x_doa = x_doa.permute(2,0,1) # (T,N,C)

        # Applying cross-stitch transformer's encoder
        x_sed, x_doa = self.transformer(x_sed,x_doa)
        x_sed = x_sed.transpose(0,1)
        x_doa = x_doa.transpose(0,1)

        """
            SED
        """
        sed_output =  self.sed_output(x_sed) # (N,T,num,Cls)


        """
            DOA
        """
        #
        doa_output = self.doa_output(x_doa,noise)  # (N,T,num,xyz)

        output = {
            'sed': sed_output,
            'doa': doa_output
        }
        
        return output
    
    def training_step(self, batch):

        audio_features, targets = batch
        predicton_dict = self(audio_features)

        loss_fcn = Losses()
        loss_dict = loss_fcn.calculate(pred=predicton_dict, target=targets)

        self.log('train_loss', loss_dict['all'],logger=True, prog_bar=True)
        
        return {'loss':loss_dict['all'], 'loss_dict': loss_dict}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        epoch_sed_loss = torch.stack([x['loss_dict']['sed'] for x in outputs]).mean()
        epoch_doa_loss = torch.stack([x['loss']['doa'] for x in outputs]).mean()
        
        
        self.log('Epoch_ALL_train_loss', epoch_loss)
        self.log('Epoch_SED_train_loss', epoch_sed_loss)
        self.log('Epoch_DOA_train_loss', epoch_doa_loss)
    
    

    def validation_step(self, batch):
        audio_features, targets = batch
        predicton_dict = self(audio_features)

        loss_fcn = Losses(mode='val')
        loss_dict = loss_fcn.eval_calculate(pred=predicton_dict, target=targets)

        self.log('val_loss', loss_dict['all'],logger=True, prog_bar=True)
        self.log('Epoch_ALL_val_loss', loss_dict['all'], logger=True, on_epoch=True)
        self.log('Epoch_SED_val_loss', loss_dict['sed'], logger=True, on_epoch=True)
        self.log('Epoch_DOA_val_loss', loss_dict['doa'], logger=True, on_epoch=True)


        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()
    
    