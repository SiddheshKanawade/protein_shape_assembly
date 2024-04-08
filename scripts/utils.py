from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils # the PyUUL utility module
import numpy as np

def convert_pdb_to_np_array_point_cloud(filepath_pdb):
    coords, atname = utils.parsePDB(filepath_pdb) # get coordinates and atom names
    atoms_channel = utils.atomlistToChannels(atname) # calculates the corresponding channel of each atom
    radius = utils.atomlistToRadius(atname) # calculates the radius of each atom

    device = "gpu" #gpu runs the volumes on GPU (you need a cuda-compatible GPU computer for this)
    PointCloudVolumeObject = VolumeMaker.PointCloudSurface(device=device)

    coords = coords.to(device)
    radius = radius.to(device)
    atoms_channel = atoms_channel.to(device)

    VolumePointCloud = PointCloudVolumeObject(coords, radius)
    pc_array = np.array(VolumePointCloud)
    return pc_array