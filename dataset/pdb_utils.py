import numpy as np
import torch
from pyuul import VolumeMaker  # the main PyUUL module
from pyuul import utils  # the PyUUL utility module


def farthest_point_sampling(points, num_samples):
    num_points = points.size(0)
    centroids = torch.zeros(num_samples, dtype=torch.long)
    distance = torch.ones(num_points) * 1e10
    farthest = torch.randint(0, num_points, (1,)).item()
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = torch.sum((points - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1].item()
    return points[centroids]


def convert_pdb_to_np_array_point_cloud(filepath_pdb, num_samples=1000):
    coords, atname = utils.parsePDB(filepath_pdb)
    atoms_channel = utils.atomlistToChannels(atname)
    radius = utils.atomlistToRadius(atname)

    device = "cpu"
    PointCloudVolumeObject = VolumeMaker.PointCloudSurface(device=device)

    coords = coords.to(device)
    radius = radius.to(device)
    atoms_channel = atoms_channel.to(device)

    VolumePointCloud = PointCloudVolumeObject(coords, radius)
    pc_array = np.array(VolumePointCloud)

    # Apply Farthest Point Sampling
    coords_tensor = torch.from_numpy(pc_array).float()
    sampled_coords_tensor = farthest_point_sampling(coords_tensor, num_samples)
    sampled_coords = sampled_coords_tensor.numpy()

    return sampled_coords

