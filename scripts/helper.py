import numpy as np
import os
import math

# file_path = '/home/siddhesh/Desktop/Siddhesh/CS_499/protein_shape_assembly/protein_extracted/7auo/7auo_24_24.npy'  # Make sure this path is correct

# # Check if the file exists
# if os.path.exists(file_path):
#     # Load the array from the .npy file
#     array = np.load(file_path)
    
#     # Print the contents of the array
#     print("Array contents:")
#     print(array)
# else:
#     print(f"File not found: {file_path}")

protein_extracted = '../protein_extracted'
file_path = '../protein_data/data_split'

if not os.path.exists(file_path):
    os.makedirs(file_path)

train_file = os.path.join(file_path, 'train.txt')
val_file = os.path.join(file_path, 'val.txt')

num_samples = len(os.listdir(protein_extracted))

num_train_samples = math.floor(0.8 * num_samples)
num_val_samples = num_samples - num_train_samples

for i, filename in enumerate(os.listdir(protein_extracted)):
    if i < num_train_samples:
        with open(train_file, 'a') as f:
            f.write(filename + '\n')
    else:
        with open(val_file, 'a') as f:
            f.write(filename + '\n')