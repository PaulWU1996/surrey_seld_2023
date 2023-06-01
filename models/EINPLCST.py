from .module import *

class EINPLCST(nn.Module):
    def __init__(self,
                target_num: int = 3,) -> None:
        super().__init__()

        self.target_num = target_num


        self.convs = ConvNetwork()
        self.observation_noise = ObservationNoise()
        
        self.positional_encoder = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=0.0)
        self.transformer = Transformer()
        self.sed_output = SedPrediction()
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
        x_sed, x_doa = self.transformer([x_sed,x_doa])
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