import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Path to the directory containing your images
image_directory = r'E:\git\cotton-plant-disease-detection\content\cotton_new\stem\train\fresh_stem'

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

# Loop through all the images in the directory
for img_name in os.listdir(image_directory):
    img_path = os.path.join(image_directory, img_name)

    if img_name.endswith('.jpg'):  # Process only .jpg images
        print(f'Processing image: {img_name}')

        # Load the image
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Augment the image and save in the same directory
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=image_directory, save_prefix='aug_', save_format='jpeg'):
            i += 1
            if i >= 10:  # Number of augmented images per original image
                break

print("Augmentation complete!")
