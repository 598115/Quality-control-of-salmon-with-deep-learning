
import os
from PIL import Image

class Util:
    """
    A utility class for image augmentation operations. Specifically for the binary model dataset.
    This class provides functionality to create augmented versions of images in a specified directory structure.
    It performs horizontal and vertical flips of images and saves them in the same directory as the original images.
    Attributes:
        path (str): Base directory path containing the image dataset structure.
    Methods:
        create_augmentations(): Creates horizontally and vertically flipped versions of all images in the dataset.
    Directory Structure Expected:
        path/
        ├── train/
        │   ├── good/
        │   └── bad/
        └── validate/
            ├── good/
            └── bad/
    Supported Image Formats:
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        - WebP (.webp)
    """
    def __init__(self, path):
        self.path = path
        
    def create_augmentations(self):
            """
            Creates augmented versions of images in the dataset by applying horizontal and vertical flips.
            The function processes all images in the train and validate directories for both good and bad classes.
            For each image, it creates:
            - A horizontal flip version (prefixed with 'horiz_flip_')
            - A vertical flip version (prefixed with 'vert_flip_')
            Supported image formats: PNG, JPG, JPEG, WEBP
            """ 
            DATA_DIR = self.path
            output_images_train_good = os.path.join(DATA_DIR, 'train', 'good')
            output_images_validate_bad = os.path.join(DATA_DIR, 'validate', 'bad')
            output_images_train_bad = os.path.join(DATA_DIR, 'train', 'bad')
            output_images_validate_good = os.path.join(DATA_DIR, 'validate', 'good')

            dirs = [output_images_train_good, output_images_validate_good, output_images_train_bad, output_images_validate_bad]
            
            for dir_path in dirs:
                for filename in os.listdir(dir_path):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_path = os.path.join(dir_path, filename)
                        img = Image.open(img_path)

                        # Horizontal flip
                        img_h = img.transpose(Image.FLIP_LEFT_RIGHT)

                        # Vertical flip  
                        img_v = img.transpose(Image.FLIP_TOP_BOTTOM)

                        # Save horizontal flip
                        img_h.save(os.path.join(dir_path, 'horiz_flip_' + filename))

                        # Save vertical flip
                        img_v.save(os.path.join(dir_path, 'vert_flip_' + filename))
            print(f"Augmentation complete. Check the directories for augmented images.")

    
