
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

# def compute_spherical_distamce_3D(y_pred: Tensor, y_true: Tensor, radius=1.0) -> Tensor:
#     """
#         Computes the distance between two points (given as angles) on a sphere in 3D.
        
#         Args:
#             y_pred: shape(N,3)
#             y_true: shape(N,3)
#             radius (float) = 1.0
#     """

    

    # dot_product = torch.sum(y_pred * y_true, dim=1)
    # dot_product = torch.clamp(dot_product, -1.0, 1.0)
    # angle = torch.acos(dot_product)
    # s_distance = radius * angle
    # return s_distance


def compute_spherical_distance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (Tensor): Tensor of predicted azimuth and elevation angles.
        y_true (Tensor): Tensor of ground-truth azimuth and elevation angles.

    Returns:
        Tensor: Tensor of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2): # we need change to dim 3
        assert RuntimeError('Input tensors require a dimension of two.')

    sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
    cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

    return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))


def compute_kld_to_standard_norm(covariance_matrix: Tensor) -> Tensor:
    """Computes the Kullback-Leibler divergence between two multivariate Gaussian distributions with identical mean,
    where the second distribution has an identity covariance matrix.

    Args:
        covariance_matrix (Tensor): Covariance matrix of the first distribution.

    Returns:
        Tensor: Tensor of KLD values.
    """
    matrix_dim = covariance_matrix.shape[-1]

    covariance_trace = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).sum(-1)

    return 0.5 * (covariance_trace - matrix_dim - torch.logdet(covariance_matrix.contiguous()))


def sedl_loss(predictions: Tuple[Tensor, Tensor, Tensor],
              targets: Tuple[Tensor, Tensor],
              alpha: float = 1.,
              beta: float = 1.) -> Tensor:        
    """ Returns the sound event detection and localization loss
    
    Args:
        predictions (tuple): Predicted source activity, direction-of-arrival and posterior covariance matrix.
        targets (Tensor): Ground-truth source activity and direction-of-arrival.
        alpha (float): Weighting factor for direction-of-arrival loss component.
        beta (float): Weighting factor for KLD loss component.

    Returns:
        Tensor: Scalar probabilistic SEL loss value.
    """

    source_cls_pred, post_mean, post_cov = predictions
    source_cls_true, direction_of_arrival = targets

    source_masks = source_cls_true.max(dim=2).values.bool()

    # detection loss
    source_cls_loss = F.binary_cross_entropy_with_logits(source_cls_pred, source_cls_true)

    # doa loss
    spherical_distance = compute_spherical_distance(post_mean.mean(dim=2)[source_masks],direction_of_arrival[source_masks])
    spherical_distance_update = torch.where(torch.isnan(spherical_distance),0.0,spherical_distance)
    doa_loss = torch.mean(spherical_distance)

    kld_loss = compute_kld_to_standard_norm(post_cov)
    kld_loss = torch.mean(kld_loss)

    final_loss = source_cls_loss + alpha * doa_loss + beta * kld_loss

    # final_loss = torch.clamp(final_loss,min=0,max=100)
    return final_loss





def psel_loss(predictions: Tuple[Tensor, Tensor, Tensor],
              targets: Tuple[Tensor, Tensor],
              alpha: float = 1.,
              beta: float = 1.) -> Tensor:
    """Returns the probabilistic sound event localization loss, as described in Eq. (5) in the paper.

    Args:
        predictions (tuple): Predicted source activity, direction-of-arrival and posterior covariance matrix.
        targets (Tensor): Ground-truth source activity and direction-of-arrival.
        alpha (float): Weighting factor for direction-of-arrival loss component.
        beta (float): Weighting factor for KLD loss component.

    Returns:
        Tensor: Scalar probabilistic SEL loss value.
    """
    source_activity, posterior_mean, posterior_covariance = predictions
    source_activity_target, direction_of_arrival_target = targets

    source_activity_loss = F.binary_cross_entropy(source_activity, source_activity_target.squeeze(0))
    source_activity_mask = source_activity_target.bool()

    spherical_distance = compute_spherical_distance(posterior_mean[source_activity_mask],
                                                    direction_of_arrival_target[source_activity_mask])
    direction_of_arrival_loss = torch.mean(spherical_distance)

    kld_loss = compute_kld_to_standard_norm(posterior_covariance)
    kld_loss = torch.mean(kld_loss)

    return source_activity_loss + alpha * direction_of_arrival_loss + beta * kld_loss
