import scipy.io
import os

# Path to a sample .mat file
file_path = r'c:\Users\M\Desktop\GLM版本\cwru_data\12k Drive End Bearing Fault Data\Ball\0007\B007_0.mat'

try:
    mat_data = scipy.io.loadmat(file_path)
    print(f"Keys in {os.path.basename(file_path)}:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(mat_data[key])} - Shape: {mat_data[key].shape}")
except Exception as e:
    print(f"Error reading .mat file: {e}")
