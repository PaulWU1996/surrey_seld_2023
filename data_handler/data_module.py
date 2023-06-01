import numpy as np
from numpy import ndarray
import os
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class UserDataset(Dataset):
    def __init__(self,
                 path_to_dataset: str = '/mnt/fast/nobackup/scratch4weeks/pw00391/noiseless_seld_feat_label',
                 split: Tuple[int, ...] = [[1]], # [1] train [2] eval or test
                 chunk_length: float = 4.0,
                 hop_length: float = 0.025) -> None:
        """
        Args:
            path_to_dataset (str): Root directory of the dataset.
            split (tuple): Splits info for spliting train/val/test dataset.
            chunk_length (float): The length of the chunk in seconds.
            hop_length (float): The length of the hop in seconds.
        """
        super().__init__()

        self.split = split[0]

        self.chunks = {}

        self.feat_dir = path_to_dataset + '/' + 'foa_dev_norm'
        self.label_dir = path_to_dataset + '/' + 'foa_dev_adpit_label'

        for file_name in os.listdir(self.feat_dir):
            if int(file_name[4]) in self.split:
                feat_file = self.feat_dir + '/' + file_name
                label_file = self.label_dir + '/' + file_name
                self._append_chunks(feat_file=feat_file,label_file=label_file,chunk_length=chunk_length,hop_length=hop_length)

    def _append_chunks( self,
                        feat_file: str,
                        label_file: str,
                        chunk_length: float,
                        hop_length: float):
        num_chunks = int(60 / chunk_length) # 60s / 4s
        num_hop_in_chunk = int(chunk_length / hop_length)
        num_label_in_chunk = int(chunk_length / 0.1)

        for chunk_idx in range(num_chunks):
            sequence_idx = len(self.chunks)

            feat_start_loc = chunk_idx * num_hop_in_chunk
            feat_end_loc = feat_start_loc + num_hop_in_chunk

            label_start_loc = chunk_idx * num_label_in_chunk
            label_end_loc = label_start_loc + num_label_in_chunk

            self.chunks[sequence_idx] = {
                'feat_file': feat_file,
                'label_file': label_file,
                'chunk_loc': chunk_idx,
                'feat_start_loc': feat_start_loc,
                'feat_end_loc':  feat_end_loc,
                'label_start_loc': label_start_loc,
                'label_end_loc': label_end_loc
            }

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        sequence = self.chunks[index]

        audio = self._get_audio_feat(feat_file=sequence['feat_file'],
                                     start_loc=sequence['feat_start_loc'],
                                     end_loc=sequence['feat_end_loc'])
        label = self._get_label(label_file=sequence['label_file'],
                                     start_loc=sequence['label_start_loc'],
                                     end_loc=sequence['label_end_loc'])
        return audio, label

    def _get_audio_feat(self,
                        feat_file,
                        start_loc,
                        end_loc):
        # feat = [num_steps, freq_bins, channels]
        feat = np.load(feat_file)
        audio_features = feat[start_loc:end_loc,:,:]
        audio_features = np.transpose(audio_features, (2,0,1))
        return audio_features
    
    def _get_label(self,
                   label_file,
                   start_loc,
                   end_loc):
        # label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        label = np.load(label_file)
        label = label[start_loc:end_loc,:,:,:]
        return label

class UserDataModule(pl.LightningDataModule):
    def __init__(self,
                 path_to_dataset: str = '/mnt/fast/nobackup/scratch4weeks/pw00391/noiseless_seld_feat_label',
                 batch_size: int = 8) -> None:
        super().__init__()
        self.data_path = path_to_dataset
        self.batch_size = batch_size

        self.train_split = [[1]]
        self.val_split = [[2]]
        self.test_split = [[2]]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = UserDataset(path_to_dataset=self.data_path,
                                       split=self.train_split)
            self.val_set = UserDataset(path_to_dataset=self.data_path,
                                       split=self.val_split)
            
        if stage == "test" or stage is None:
            self.test_set = UserDataset(path_to_dataset=self.data_path,
                                       split=self.test_split)
        return super().setup(stage)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          num_workers=12)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=12)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=12)


class T3Dataset(Dataset):
    def __init__(self,
                 path_to_dataset: str,
                 split: Tuple[int, ...] = (1, 2, 3),
                #  num_overlapping_sources: Tuple[int, ...] = (1, 2, 3),
                 chunk_length: int = 50,
                #  frame_length: float = 0.04, # hop = 0.02
                #  num_fft_bins: int = 1024,
                 num_sources_output: int = 3) -> None:
        """
        Args:
            path_to_dataset (str): Root directory of the downloaded dataset.
            train (bool): If True, creates dataset from training set, otherwise creates from test set.
            split (tuple): Indices of the splits the dataset will be created from.
            num_overlapping_sources: Number of overlapping sources that the dataset will be created from.
            chunk_length (int): Length of one chunk (signal block) in hops.
            frame_length (float): Frame length (within one chunk) in seconds.
            num_fft_bins (int): Number of frequency bins used in the fast Fourier transform (FFT).
            num_sources_output (int): Number of sources represented in the targets.
        """


        super().__init__()

        # if isinstance(split, int):
        #     split = tuple([split])

        # if isinstance(num_overlapping_sources, int):
        #     split = tuple([num_overlapping_sources])

        self.split = split[0]
        # self.num_overlapping_sources = num_overlapping_sources
        # self.frame_length = frame_length
        # self.num_fft_bins = num_fft_bins
        # self.num_sources_output = num_sources_output

        self.chunks = {}

        self.feat_dir = path_to_dataset + "foa_dev_norm/"
        self.label_dir = path_to_dataset + "foa_dev_label/"
        

        for filename in os.listdir(self.feat_dir):
            if int(filename[4]) in self.split:
                # which mean this is what we need file
                
                feat_name = self.feat_dir+filename
                label_name = self.label_dir+filename
                self._append_chunks(feat_file=feat_name,label_file=label_name,chunk_length=chunk_length)
                    
    def _append_chunks(self,
                       feat_file: str,
                       label_file: str,
                       chunk_length: int) -> None:

        num_chunks = int(np.load(label_file).shape[0]/chunk_length)
        for chunk_idx in range(num_chunks):
            sequence_idx = len(self.chunks)

            start_loc = chunk_idx * chunk_length
            end_loc = start_loc + chunk_length

            self.chunks[sequence_idx] = {
                'feat_file': feat_file,
                'label_file': label_file,
                'chunk_loc': chunk_idx,
                'start_loc': start_loc,
                'end_loc':  end_loc
            }

    def _get_audio_features(self,
                            feat_file: str,
                            start_loc: int = None,
                            end_loc: int = None) -> None:
        """
            audio_features.shape = [50,577,4] -> 513 (pha) + 64 (mel)
        """

        feat = np.load(feat_file)
        audio_features = feat[start_loc:end_loc,:,:]
        return audio_features.astype(np.float32)
    
    def _get_label(self,
                   label_file: str,
                   chunk_idx: int = None):
        """
            label.shape = [1,52] = [SED[1,13], x_axis[1,13], y_axis[1,13], z_axis[1,13]]
            SED.shape = [1,13]
            DOA.shape = [3,13]
        """

        label = np.load(label_file)[chunk_idx,:]
        SED = label[0:13]
        DOA = label[13:].reshape(3,13)

        # return SED.astype(np.float32), DOA.astype(np.float32)

        cls_id = np.zeros((5,13))
        loc = np.zeros((5,2))
        # mask = [False,False,False]

        idx = 0
        t_id = 0
        for i in range(len(SED)):
            if SED[i] == 1:
                xy = [DOA[0,idx],DOA[1,idx]] #,DOA[2,idx]]
                cls_id[t_id,i] = 1
                loc[t_id,:] = xy
                # mask[t_id] = True
                t_id += 1

                # cls_id.append(idx)
                # loc.append(xyz)
            idx += 1

        cls_id = np.array(cls_id).astype(np.float32)
        loc = np.array(loc).astype(np.float32)
        
        return cls_id, loc
        

    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, index: int) -> Tuple[ndarray,ndarray]:
        sequence = self.chunks[index]

        audio_features = self._get_audio_features(sequence['feat_file'],
                                                  sequence['start_loc'],
                                                  sequence['end_loc'])
        SED, DOA = self._get_label(sequence['label_file'],
                                   sequence['chunk_loc'])

        return audio_features, (SED, DOA)





class T3DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str = "/mnt/fast/nobackup/scratch4weeks/pw00391/DCASE2023_SELD_dataset/surrey_seld_feat_label/",
                 batch_size: int = 256) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        # DCASE Settings
        self.test_splits = [[4]]
        self.val_splits = [[4]]
        self.train_splits = [[1, 2, 3]] 

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = T3Dataset(path_to_dataset=self.data_path,
                                       split=self.train_splits)
            self.val_set = T3Dataset(path_to_dataset=self.data_path,
                                       split=self.val_splits)
            
        if stage == "test" or stage is None:
            self.test_set = T3Dataset(path_to_dataset=self.data_path,
                                       split=self.test_splits)
        return super().setup(stage)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          num_workers=1)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=12)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=12)
