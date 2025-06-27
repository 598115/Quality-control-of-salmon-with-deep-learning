import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import pandas as pd
from .Config import Config

class ImageResizer:
    """
    A class for resizing images and their corresponding zone and spot masks to a specified target size.
    The ImageResizer class reads image paths from processed CSV files, processes both training/validation and test 
    images along with their masks, and saves the resized versions back to disk in the same directories.
    Attributes:
        conf (Config): Configuration object containing data directory settings
        data_dir (str): Path to the data directory
        df (pandas.DataFrame): DataFrame containing training/validation data paths
        test_df (pandas.DataFrame): DataFrame containing test data paths
    Methods:
        resize(): Resizes all images and masks in both training/validation and test sets to 512x256 pixels.
                 Uses bilinear interpolation for images and nearest neighbor interpolation for masks to 
                 preserve label values. Overwrites original files with resized versions.
    Notes:
        - Images are resized to 512x256 (height x width)
        - Images are converted to RGB format
        - Masks are preserved as numpy arrays
        - Uses torchvision transforms for resizing
        - Handles both zone and spot masks separately
        - Preserves label values in masks using nearest neighbor interpolation
    """

    def __init__(self, test_csv):
        self.conf = Config()
        self.data_dir = self.conf.get_data_dir()
        self.df = pd.read_csv(os.path.join(self.data_dir, 'processed_data.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, test_csv))
        
    def resize(self, resize_test=True):
        """
        Resizes images and their corresponding zone and spot masks to a target size.
        This method processes both training/validation and test datasets by:
        1. Loading images and their corresponding masks
        2. Resizing masks using nearest neighbor interpolation to preserve label values
        3. Saving the resized images and masks by overwriting the original files
        The target size is fixed at (512, 256) pixels (height x width).
        Images are processed as RGB using PIL Image format.
        Masks are processed as numpy arrays, converted to PIL Image for resizing,
        then back to numpy arrays for saving.
        Uses:
            - PIL Image for image processing
            - torchvision.transforms for resizing operations
            - numpy for mask array operations
            - tqdm for progress tracking
        The method processes training/validation data first, followed by test data,
        and provides progress updates for both sets.
        Returns:
            None. Files are saved directly to disk, overwriting original files.
        Prints:
            - Number of images processed
            - Target size used
            - Save locations for images and masks
        """

        
        image_paths = self.df['image_path'].tolist()
        zone_mask_paths = self.df['zone_label_path'].tolist()
        spot_mask_paths = self.df['spot_label_path'].tolist()

        target_size = (512, 256) #height, width

        # Define transforms for images
        img_transform = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)

        # Process each image-mask pair
        for (img_path, zone_mask_path, spot_mask_path) in tqdm(zip(image_paths, zone_mask_paths, spot_mask_paths), total=len(image_paths)):
             
            # Load image
            img = Image.open(img_path).convert("RGB")
            
            # Load mask (numpy array)
            zone_mask = np.load(zone_mask_path)
            spot_mask = np.load(spot_mask_path)
            
            # Apply transform to image
            resized_img = img_transform(img)
            
            # For numpy mask: convert to PIL, resize, then back to numpy
            # First convert numpy array to PIL Image
        
            zone_mask_pil = Image.fromarray(zone_mask.astype(np.uint8))
            spot_mask_pil = Image.fromarray(spot_mask.astype(np.uint8))
            
            # Resize mask using nearest neighbor to preserve label values
            resized_zone_mask_pil = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)(zone_mask_pil)
            resized_spot_mask_pil = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)(spot_mask_pil)
            
            # Convert back to numpy array
            resized_zone_mask = np.array(resized_zone_mask_pil)
            resized_spot_mask = np.array(resized_spot_mask_pil)
            
            # Save resized image, overwriting the original
            resized_img.save(img_path)

            zone_output_path = os.path.normpath(zone_mask_path)
            spot_output_path = os.path.normpath(spot_mask_path)
            
            # Save resized masks as numpy file, overwriting the original
            np.save(zone_output_path.replace('.npy', ''), resized_zone_mask)
            np.save(spot_output_path.replace('.npy', ''), resized_spot_mask)

        print(f"Processed {len(image_paths)} training/validation images and masks")
        print(f"Resized to {target_size[0]}x{target_size[1]} (height x width)")
        print(f"Images saved to {os.path.join(self.conf.get_data_dir(), 'images')}")
        print(f"Masks saved to {os.path.join(self.conf.get_data_dir(), 'combined_labels')}")

        if resize_test:
            image_paths = self.test_df['image_path'].tolist()
            zone_mask_paths = self.test_df['zone_label_path'].tolist()
            spot_mask_paths = self.test_df['spot_label_path'].tolist()

            # Process each image-mask pair
            for (img_path, zone_mask_path, spot_mask_path) in tqdm(zip(image_paths, zone_mask_paths, spot_mask_paths), total=len(image_paths)):
                
                # Load image
                img = Image.open(img_path).convert("RGB")
                
                # Load mask (numpy array)
                zone_mask = np.load(zone_mask_path)
                spot_mask = np.load(spot_mask_path)
                
                # Apply transform to image
                resized_img = img_transform(img)
                
                # For numpy mask: convert to PIL, resize, then back to numpy
                # First convert numpy array to PIL Image
            
                zone_mask_pil = Image.fromarray(zone_mask.astype(np.uint8))
                spot_mask_pil = Image.fromarray(spot_mask.astype(np.uint8))
                
                # Resize mask using nearest neighbor to preserve label values
                resized_zone_mask_pil = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)(zone_mask_pil)
                resized_spot_mask_pil = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)(spot_mask_pil)
                
                # Convert back to numpy array
                resized_zone_mask = np.array(resized_zone_mask_pil)
                resized_spot_mask = np.array(resized_spot_mask_pil)
                
                # Save resized image, overwriting the original
                resized_img.save(img_path)

                zone_output_path = os.path.normpath(zone_mask_path)
                spot_output_path = os.path.normpath(spot_mask_path)
                
                # Save resized masks as numpy file, overwriting the original
                np.save(zone_output_path.replace('.npy', ''), resized_zone_mask)
                np.save(spot_output_path.replace('.npy', ''), resized_spot_mask)

            print(f"Processed {len(image_paths)} test images and masks")
            print(f"Resized to {target_size[0]}x{target_size[1]} (height x width)")
            print(f"Images saved to {os.path.join(self.conf.get_data_dir(), 'images')}")
            print(f"Masks saved to {os.path.join(self.conf.get_data_dir(), 'combined_labels')}")