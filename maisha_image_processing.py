import h5py
import numpy as np
import os
import cv2

# Define the path to the data folder and the output folder for processed files
data_folder = '/kaggle/input/brats2020-training-data/BraTS2020_training_data/content/data/'
processed_folder = '/kaggle/working/preprocessed_images/'

# Create the output folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

# Loop through all H5 files in the data folder
for file_name in os.listdir(data_folder):
    if file_name.endswith('.h5'):  # Only process .h5 files
        file_path = os.path.join(data_folder, file_name)
        
        # Open the H5 file
        with h5py.File(file_path, 'r') as h5_file:
            if 'image' in h5_file:  # Ensure 'image' dataset exists
                image_data = h5_file['image'][:]
                
                # Normalize the image data (assuming it's a 3D volume)
                image_data_normalized = image_data / np.max(image_data)
                
                # Resize each 2D slice to a fixed size (e.g., 128x128)
                resized_slices = [cv2.resize(slice, (128, 128)) for slice in image_data_normalized]
                resized_slices = np.array(resized_slices)
                
                # Ensure the shape is correct after resizing (add channel dimension if needed)
                resized_slices = np.expand_dims(resized_slices, axis=-1)
                
                # Save the processed image data into a new H5 file
                save_image_path = os.path.join(processed_folder, file_name.replace('.h5', '_processed_image.h5'))
                with h5py.File(save_image_path, 'w') as save_image_file:
                    save_image_file.create_dataset('image', data=resized_slices)
                
                print(f"Processed and saved: {file_name}")

