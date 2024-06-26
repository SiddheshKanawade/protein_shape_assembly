import os
import numpy as np
from dataset.pdb_utils import convert_pdb_to_np_array_point_cloud

pdb_data_dir = "../protein_data/"
pc_data_dir ="../protein_extracted_new/"

if not os.path.exists(pc_data_dir):
    os.makedirs(pc_data_dir)

print(len(os.listdir(pdb_data_dir)))

for filename in os.listdir(pdb_data_dir):
    if filename[-4:] == ".tcl":
        continue
    file_path = os.path.join(pdb_data_dir, filename)
    pc_file_path = os.path.join(pc_data_dir, filename)
    if not os.path.exists(pc_file_path):
        os.makedirs(pc_file_path)
    for file in os.listdir(file_path):
        if file[-4:] != ".pdb":
            continue
        print(f"Current PDB file name: {file}")
        pdb_file = os.path.join(file_path,file)
        pc_array = convert_pdb_to_np_array_point_cloud(pdb_file)
        print(pc_array.shape)
        numpy_pc_path = os.path.join(pc_file_path, file[:-4]+".npy")
        np.save(numpy_pc_path, pc_array)
