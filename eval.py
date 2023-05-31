import torch
from data_handler import data_module, T3Dataset
from models import T3Model
from utls import sedl_loss

import os
import numpy as np


# feat = np.load("/mnt/fast/nobackup/scratch4weeks/pw00391/DCASE2023_SELD_dataset/surrey_seld_feat_label/foa_dev_norm/fold4_room24_mix016.npy")

# data = data_module.T3DataModule(batch_size=1)
# data.setup()

# model = T3Model.load_from_checkpoint('512.ckpt')
# val_dataloader = data.val_dataloader()

# model.eval()

# print(model(feat))

# RESULT = []

# chunk_id = 0

# with torch.no_grad():
#     for batch in val_dataloader:
#         feats, labels = batch
#         outputs = model(feats)
#         loss = sedl_loss(outputs,labels)

#         cls_id_pred, doa, _ = outputs

#         max_val, max_loc = cls_id_pred.max(dim=-1)
        
#         i = 0
#         for value in max_val:
#             if value > 0.5:
#                 RESULT.append([chunk_id, max_loc[i], doa[i,:][0], doa[i,:][1]])
#             i += 1
#         chunk_id += 1

#         # predictions.append(outputs)

# filename = "64.txt"  # Specify the output file name

# with open(filename, "w") as file:
#     for item in RESULT:
#         file.write(str(item) + "\n")

# print("Eval is completed!")


import torch
from data_handler import data_module, T3Dataset
from models import T3Model
from utls import sedl_loss

import os
import numpy as np

model = T3Model.load_from_checkpoint('/mnt/fast/nobackup/users/pw00391/dcase/512.ckpt')

path_to_feat = "/mnt/fast/nobackup/scratch4weeks/pw00391/DCASE2023_SELD_dataset/surrey_seld_feat_label/foa_dev_norm/"

for file_name in enumerate(os.listdir(path_to_feat)):
    npy_path = path_to_feat+file_name[1]

    feat = torch.from_numpy(np.load(npy_path).astype(np.float32))

    if feat.shape[0] % 50:
        mod = feat.shape[0] % 50

        new_shape = (feat.shape[0]//50,50,577,4)
        feat_1 = feat[:-mod,:,:].view(*new_shape)

        feat_2 = feat[-mod:,:,:].view(1,mod,577,4)
    else:
        new_shape = (feat.shape[0]//50,50,577,4)
        feat_1 = feat[:-mod,:,:].view(*new_shape)

    if feat.shape[0] % 50:
        target_cls, posterior_mean, posterior_covariance = model(feat_1)
        print("pause")
    else:
        target_cls, posterior_mean, posterior_covariance = model(feat_1)

    print("pause")

    
