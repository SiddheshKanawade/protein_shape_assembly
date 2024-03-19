import numpy as np
import os

file_path = '/home/siddhesh/Desktop/Siddhesh/CS_499/protein_shape_assembly/protein_extracted/7auo/7auo_24_24.npy'  # Make sure this path is correct

# Check if the file exists
if os.path.exists(file_path):
    # Load the array from the .npy file
    array = np.load(file_path)
    
    # Print the contents of the array
    print("Array contents:")
    print(array)
else:
    print(f"File not found: {file_path}")
