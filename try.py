#import trimesh
import numpy as np
import matplotlib.pyplot as plt

pc = "protein_extracted/1a6f/1a6f_94_108.npy"
# Load your point cloud data
point_cloud = np.load(pc)
print(point_cloud)
#Create a Trimesh object
# mesh = trimesh.Trimesh(vertices=point_cloud)

# mesh.export("output_model.obj")

# mesh = trimesh.load('output_model.obj')

# # Extract vertices 
# vertices = mesh.vertices

# # Basic 3D Scatter Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:,2])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:,2])
# plt.show()


import open3d as o3d


# Create Open3D point cloud object 
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Visualization
o3d.visualization.draw_geometries([pcd]) 
