import torch
import torch.nn as nn
import torch.nn.functional as F
eps = torch.finfo(torch.float32).eps

import numpy as np

from scipy.optimize import linear_sum_assignment


class BCEWithLogitsLoss:
    def __init__(self,reduction='mean') -> None:
        super().__init__()
        self.name = 'loss_BCEWithLogits'
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def calculate(self, pred, target):
        batch, time_step, n_track, dim = pred.shape
        _, _, n_gt, _ = target.shape
        COST = []
        
        for i in range(n_track):
            cost = []
            for j in range(n_gt):
                cost.append(self.loss(pred[:,:,i],target[:,:,j]).mean(dim=2))
            COST.append(torch.stack(cost).permute(1,2,0)) # (N,T,n)
        COST = torch.concat(COST,dim=2)

        return COST

class MSELoss:
    def __init__(self,reduction='mean') -> None:
        super().__init__()
        self.name = 'loss_MSE'
        self.loss = nn.MSELoss(reduction=reduction)

    def calculate(self, pred, target):
        batch, time_step, n_track, dim = pred.shape
        _, _, n_gt, _ = target.shape
        COST = []
        for i in range(n_track):
            cost = []
            for j in range(n_gt):
                cost.append(self.loss(pred[:,:,i],target[:,:,j]).mean(dim=2))
            COST.append(torch.stack(cost).permute(1,2,0)) # (N,T,n)
        COST = torch.concat(COST,dim=2)

        return COST


class Losses:
    def __init__(self,mode:str = 'train') -> None:
        super().__init__()
        if mode == 'train':
            reduction = 'mean'
            self.losses = [nn.BCEWithLogitsLoss(reduction=reduction), nn.MSELoss(reduction=reduction)]
        else:
            reduction = 'none'
            self.losses = [BCEWithLogitsLoss(reduction=reduction), MSELoss(reduction=reduction)]
        

        self.names = ['loss_all'] + ['loss.name' in self.losses]

    def calculate(self, pred, target):
        p_sed = pred['sed']
        p_doa = pred['doa']
        t_sed = target[...,:13]
        t_doa = target[...,13:]

       
        sed_loss = self.losses[0](p_sed,t_sed)#.mean()
        doa_loss = self.losses[1](p_doa,t_doa)#.mean()

        all_loss = sed_loss * 0.4 + doa_loss * 0.6

        loss_dict = {
            'sed': sed_loss,
            'doa': doa_loss,
            'all': all_loss
        }

        return loss_dict

    def eval_calculate(self, pred, target):
        """
        pred: (dict) {'sed','doa'} {(N,t,n,13), (N,t,n,3)}
        target: (tensor) (N,T,n,16) -> n = num_target, 16 = (13 + 3)
        """
        # 
        p_sed = pred['sed']
        p_doa = pred['doa']
        t_sed = target[...,:13]
        t_doa = target[...,13:]

        N, t, n, _ = p_sed.shape

        # obtain the cost matrix from sed and doa
        # we will leave the computation graph

        sed_cost = self.losses[0].calculate(pred=p_sed,target=t_sed).cpu().detach().numpy().reshape(N,t,n,n)
        doa_cost = self.losses[1].calculate(pred=p_doa,target=t_doa).cpu().detach().numpy().reshape(N,t,n,n)

        sum_cost = (sed_cost + doa_cost).reshape(N,t,n,n)
        SED_loss = []
        DOA_loss = []

        new_sed = torch.zeros_like(p_sed)
        new_doa = torch.zeros_like(p_doa)

        for i in range(N):
            sed_loss = []
            doa_loss = []
            for j in range(t):
                frame_cost = sum_cost[i,j]
                row_ind, col_ind = linear_sum_assignment(frame_cost)

                sed_t = sed_cost[i,j][row_ind, col_ind].mean()
                doa_t = doa_cost[i,j][row_ind, col_ind].mean()

                sed_loss.append(sed_t)
                doa_loss.append(doa_t)

                # re-range the output
                new_sed[i,j] = p_sed[i,j,col_ind]
                new_doa[i,j] = p_doa[i,j,col_ind]

            SED_loss.append(np.array(sed_loss))
            DOA_loss.append(np.array(doa_loss))   

        SED_loss = torch.from_numpy(np.stack(SED_loss)).mean()
        DOA_loss = torch.from_numpy(np.stack(DOA_loss)).mean()

        ALL_loss = 0.5 * (SED_loss + DOA_loss)

        ALL_loss = ALL_loss.mean()

        loss_dict = {
            'all': ALL_loss,
            'sed': SED_loss,
            'doa': DOA_loss,
            'updated_target': {'sed': new_sed, 'doa':new_doa}
        }

        return loss_dict

        






# audio = torch.from_numpy(np.load('/mnt/fast/nobackup/users/pw00391/dcase/audio.npy').astype(np.float32))
# target = torch.from_numpy(np.load('/mnt/fast/nobackup/users/pw00391/dcase/target.npy').astype(np.float32))
# SED = torch.from_numpy(np.load('/mnt/fast/nobackup/users/pw00391/dcase/SED.npy').astype(np.float32))
# DOA = torch.from_numpy(np.load('/mnt/fast/nobackup/users/pw00391/dcase/DOA.npy').astype(np.float32))

# t_sed = target[...,:13]
# t_doa = target[...,13:]

# sed_loss = BCEWithLogitsLoss()
# doa_loss = MSELoss()

# sed_cost = sed_loss.calculate(pred=SED,target=t_sed) # (N,t, 10*10)
# doa_cost = doa_loss.calculate(pred=DOA,target=t_doa)

# loss = Losses()

# output = loss.calculate(pred={'sed':SED,'doa':DOA},target=target)

# print(1)


# class Losses:
#     def __init__(self, cfg):
        
#         self.cfg = cfg
#         self.beta = cfg['training']['loss_beta']

#         self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
#         self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]

#         self.names = ['loss_all'] + [loss.name for loss in self.losses]
    
#     def calculate(self, pred, target, epoch_it=0):

#         if 'PIT' not in self.cfg['training']['PIT_type']:
#             updated_target = target
#             loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
#             loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
#         elif self.cfg['training']['PIT_type'] == 'tPIT':
#             loss_sed, loss_doa, updated_target = self.tPIT(pred, target)

#         loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
#         losses_dict = {
#             'all': loss_all,
#             'sed': loss_sed,
#             'doa': loss_doa,
#             'updated_target': updated_target
#         }
#         return losses_dict

#     def tPIT(self, pred, target):
#         """Frame Permutation Invariant Training for 2 possible combinations

#         Args:
#             pred: {
#                 'sed': [batch_size, T, num_tracks=2, num_classes], 
#                 'doa': [batch_size, T, num_tracks=2, doas=3]
#             }
#             target: {
#                 'sed': [batch_size, T, num_tracks=2, num_classes], 
#                 'doa': [batch_size, T, num_tracks=2, doas=3]            
#             }
#         Return:
#             updated_target: updated target with the minimum loss frame-wisely
#                 {
#                     'sed': [batch_size, T, num_tracks=2, num_classes], 
#                     'doa': [batch_size, T, num_tracks=2, doas=3]            
#                 }
#         """
#         target_flipped = {
#             'sed': target['sed'].flip(dims=[2]),
#             'doa': target['doa'].flip(dims=[2])
#         }

#         loss_sed1 = self.losses_pit[0].calculate_loss(pred['sed'], target['sed'])
#         loss_sed2 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped['sed'])
#         loss_doa1 = self.losses_pit[1].calculate_loss(pred['doa'], target['doa'])
#         loss_doa2 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped['doa'])

#         loss1 = loss_sed1 + loss_doa1
#         loss2 = loss_sed2 + loss_doa2

#         loss_sed = (loss_sed1 * (loss1 <= loss2) + loss_sed2 * (loss1 > loss2)).mean()
#         loss_doa = (loss_doa1 * (loss1 <= loss2) + loss_doa2 * (loss1 > loss2)).mean()
#         updated_target_sed = target['sed'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
#             target_flipped['sed'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
#         updated_target_doa = target['doa'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
#             target_flipped['doa'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
#         updated_target = {
#             'sed': updated_target_sed,
#             'doa': updated_target_doa
#         }
#         return loss_sed, loss_doa, updated_target


# class MSELoss:
#     def __init__(self, reduction='mean'):
#         self.reduction = reduction
#         self.name = 'loss_MSE'
#         if self.reduction != 'PIT':
#             self.loss = nn.MSELoss(reduction='mean')
#         else:
#             self.loss = nn.MSELoss(reduction='none')
    
#     def calculate_loss(self, pred, target):
#         if self.reduction != 'PIT':
#             return self.loss(pred, target)
#         else:
#             return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


# class BCEWithLogitsLoss:
#     def __init__(self, reduction='mean', pos_weight=None):
#         self.reduction = reduction
#         self.name = 'loss_BCEWithLogits'
#         if self.reduction != 'PIT':
#             self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
#         else:
#             self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
#     def calculate_loss(self, pred, target):
#         if self.reduction != 'PIT':
#             return self.loss(pred, target)
#         else:
#             return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))



# audio = np.load('/mnt/fast/nobackup/users/pw00391/dcase/audio.npy').astype(np.float32)
# target = np.load('/mnt/fast/nobackup/users/pw00391/dcase/label.npy').astype(np.float32)
# SED = np.load('/mnt/fast/nobackup/users/pw00391/dcase/SED.npy').astype(np.float32)
# DOA = np.load('/mnt/fast/nobackup/users/pw00391/dcase/DOA.npy').astype(np.float32)

# target = torch.from_numpy(target)
# SED = torch.from_numpy(SED)
# DOA = torch.from_numpy(DOA)


# losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
# updated_target = target
# loss_sed = losses[0].calculate_loss(pred['sed'], updated_target['sed'])
# loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        