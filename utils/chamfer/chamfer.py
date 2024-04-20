import chamfer_cuda
import torch
from torch.cuda.amp import custom_bwd, custom_fwd


def safe_sqrt(x, eps=1e-12):
    return torch.sqrt(torch.clamp(x, eps))


"""Chamfer distance module
"""


def bpdist2(feature1, feature2, data_format="NWC"):
    """This version has a high memory usage but more compatible(accurate) with optimized Chamfer Distance."""
    if data_format == "NCW":
        diff = feature1.unsqueeze(3) - feature2.unsqueeze(2)
        distance = torch.sum(diff**2, dim=1)
    elif data_format == "NWC":
        diff = feature1.unsqueeze(2) - feature2.unsqueeze(1)
        distance = torch.sum(diff**2, dim=3)
    else:
        raise ValueError("Unsupported data format: {}".format(data_format))
    return distance


def nn_distance_torch(xyz1, xyz2, data_format="NWC"):
    assert torch.is_tensor(xyz1) and xyz1.dim() == 3
    assert torch.is_tensor(xyz2) and xyz2.dim() == 3
    if data_format == "NCW":
        assert xyz1.size(1) == 3 and xyz2.size(1) == 3
    elif data_format == "NWC":
        assert xyz1.size(2) == 3 and xyz2.size(2) == 3
    distance = bpdist2(xyz1, xyz2, data_format)
    dist1, idx1 = distance.min(2)
    dist2, idx2 = distance.min(1)
    return dist1, idx1, dist2, idx2


def test_chamfer_cpu(xyz1, xyz2):
    # ---------------------------------------------------------------------------- #
    # NWC format
    # ---------------------------------------------------------------------------- #

    # check forward
    (
        dist1_desired,
        idx1_desired,
        dist2_desired,
        idx2_desired,
    ) = nn_distance_torch(xyz1, xyz2, "NWC")
    print(dist1_desired.device, dist1_desired.shape)
    print(dist2_desired.device, dist2_desired.shape)

    return dist1_desired, dist2_desired


# class ChamferDistanceFunction(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, xyz1, xyz2):
#         xyz1 = xyz1.contiguous()
#         xyz2 = xyz2.contiguous()
#         assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."

#         dist1, idx1, dist2, idx2 = chamfer_cuda.chamfer_forward(xyz1, xyz2)
#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
#         return dist1, dist2

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_dist1, grad_dist2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#         grad_dist1 = grad_dist1.contiguous()
#         grad_dist2 = grad_dist2.contiguous()
#         assert grad_dist1.is_cuda and grad_dist2.is_cuda, "Only support cuda currently."
#         grad_xyz1, grad_xyz2 = chamfer_cuda.chamfer_backward(
#             grad_dist1, grad_dist2, xyz1, xyz2, idx1, idx2
#         )
#         return grad_xyz1, grad_xyz2


def chamfer_distance(xyz1, xyz2, transpose=False, sqrt=False, eps=1e-12):
    """Chamfer distance

    Args:
        xyz1 (torch.Tensor): (b, n1, 3)
        xyz2 (torch.Tensor): (b, n1, 3)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
        sqrt (bool): whether to square root distance
        eps (float): to safely sqrt

    Returns:
        dist1 (torch.Tensor): (b, n1)
        dist2 (torch.Tensor): (b, n2)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)

    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    # dist1, dist2 = ChamferDistanceFunction.apply(xyz1, xyz2)
    dist1, dist2 = test_chamfer_cpu(xyz1, xyz2)
    if sqrt:
        dist1 = safe_sqrt(dist1, eps)
        dist2 = safe_sqrt(dist2, eps)
    return dist1, dist2


def nn_distance(xyz1, xyz2, transpose=True):
    """The interface to infer rather than train"""
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2).contiguous()
        xyz2 = xyz2.transpose(1, 2).contiguous()
    return chamfer_cuda.chamfer_forward(xyz1, xyz2)
