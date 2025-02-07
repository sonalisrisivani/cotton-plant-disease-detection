import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Path to the base directory containing your subfolders
base_directory = r'E:\git\cotton-plant-disease-detection\content\cotton_new\bud\train'

# Create a data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through all subfolders in the base directory
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)
    
    if os.path.isdir(folder_path):  # Only process directories
        print(f"Processing folder: {folder_name}")

        # Loop through all the images in the folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            if img_name.endswith('.jpg'):  # Process only .jpg images
                print(f'  Processing image: {img_name}')

                # Load the image
                img = load_img(img_path)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Augment the image and save in the same folder
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_path, save_prefix='aug_', save_format='jpeg'):
                    i += 1
                    if i >= 10:  # Number of augmented images per original image
                        break

print("Augmentation complete!")
