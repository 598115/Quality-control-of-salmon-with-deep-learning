from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .Config import Config
import numpy as np
import os
from torchvision.transforms import functional as TF
import random
from tqdm import tqdm

class MaskDataset(Dataset):
    """
    This class handles loading and preprocessing of image data and their corresponding segmentation masks.
    It supports data augmentation through rotation transformations.
    Attributes:
        data (pandas.DataFrame): DataFrame containing the paths to the images and their corresponding masks.
            Expected columns: 'image_path', 'zone_label_path', 'spot_label_path'.
        dir (str): Base directory path for the dataset files.
        transform (bool): If True, applies data augmentation transformations to the images and masks.
        p_rotate (float): Probability of applying rotation augmentation (default: 0.5).
        rotate_range (tuple): Range of angles for rotation in degrees as (min_angle, max_angle) (default: (-30, 30)).
    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns a tuple of (image, zone_mask, spot_mask) for the given index.
        _apply_transforms(image, zone_mask, spot_mask): Applies data augmentation transformations.
    Returns:
        tuple: A tuple containing:
            - image (torch.Tensor): RGB image tensor of shape (3, H, W), normalized to [0, 1]
            - zone_mask (torch.Tensor): Zone segmentation mask of shape (H, W) with integer labels
            - spot_mask (torch.Tensor): Spot segmentation mask of shape (H, W) with integer labels
    """
    
    def __init__(self, dataframe, transform=False,
                 p_rotate=0.5, rotate_range=(-30, 30)):
        self.data = dataframe
        conf = Config()
        self.dir = conf.get_data_dir()
        self.transform = transform
        # Augmentation probabilities
        self.p_rotate = p_rotate
        self.rotate_range = rotate_range
    
    def __len__(self):
        return len(self.data)

    def _apply_transforms(self, image, zone_mask, spot_mask):
        # Convert tensors to PIL images for transforms
        image_np = image.permute(1, 2, 0).numpy() * 255.0
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        
        # For masks: Convert from tensor to PIL
        zone_mask_pil = Image.fromarray(zone_mask.numpy().astype(np.uint8))
        spot_mask_pil = Image.fromarray(spot_mask.numpy().astype(np.uint8))

        # Geometric transformations (apply to both image and masks)
        if random.random() < self.p_rotate:
            angle = random.uniform(*self.rotate_range)
            # Use different interpolation methods for image and masks
            image_pil = TF.rotate(image_pil, angle, interpolation=Image.BILINEAR, fill=0)
            zone_mask_pil = TF.rotate(zone_mask_pil, angle, interpolation=Image.NEAREST, fill=0)
            spot_mask_pil = TF.rotate(spot_mask_pil, angle, interpolation=Image.NEAREST, fill=0)
        
        # Convert back to tensors
        image = TF.to_tensor(image_pil)
        zone_mask = torch.from_numpy(np.array(zone_mask_pil)).long()
        spot_mask = torch.from_numpy(np.array(spot_mask_pil)).long()
        
        return image, zone_mask, spot_mask

    def __getitem__(self, idx):
        # Get file paths from dataframe
        img_path = os.path.join(self.dir, self.data.iloc[idx]['image_path'])
        zone_mask_path = os.path.join(self.dir, self.data.iloc[idx]['zone_label_path'])
        spot_mask_path = os.path.join(self.dir, self.data.iloc[idx]['spot_label_path'])
        
        # Load image and convert to tensor
        image = Image.open(img_path).convert('RGB')
        image = torch.FloatTensor(np.array(image)) / 255.0
        image = image.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
        
        # Load mask directly as numpy array since it's .npy format
        zone_mask = np.load(zone_mask_path)
        spot_mask = np.load(spot_mask_path)
        # Ensure all expected classes (0, 1, 2) are preserved during conversion
        zone_mask = torch.from_numpy(zone_mask.astype(np.int64)).long()  # Convert to tensor with long dtype for class labels
        spot_mask = torch.from_numpy(spot_mask.astype(np.int64)).long()

        # Apply transformations if enabled (only for training)
        if self.transform:
            image, zone_mask, spot_mask = self._apply_transforms(image, zone_mask, spot_mask)

        return image, zone_mask, spot_mask


class DatasetLoader():
    """A class for loading and preparing image datasets for the segmentation model.
    This class handles the loading, splitting, and validation of image datasets,
    Args:
        dataFrame (pandas.DataFrame): DataFrame containing training/validation data paths and information.
        test_dataframe (pandas.DataFrame): DataFrame containing test data paths and information.
        batch_size (int): The size of batches to use in data loaders.
        augmentation (bool, optional): Whether to apply data augmentation to training data. Defaults to False.
    Attributes:
        df (pandas.DataFrame): Stored training/validation DataFrame.
        test_df (pandas.DataFrame): Stored test DataFrame.
        bsize (int): Stored batch size.
        augmentation (bool): Stored augmentation flag.
    Methods:
        load_data(num_workers=4, pin_memory=True):
            Loads and prepares the datasets with multi-worker support.
            Args:
                num_workers (int, optional): Number of worker processes. Defaults to 4.
                pin_memory (bool, optional): Whether to pin memory in data loading. Defaults to True.
            Returns:
                tuple: Contains three DataLoader objects (train_loader, val_loader, test_loader).
            Features:
                - Splits data into training and validation sets
                - Creates data loaders with specified batch size and workers
                - Validates data shapes and unique values
                - Checks for empty batches
                - Provides progress bars for loading
                - Prints comprehensive dataset statistics
    """

    def __init__(self, dataFrame, test_dataframe, batch_size, augmentation=False):
        self.df = dataFrame
        self.test_df = test_dataframe
        self.bsize = batch_size
        self.augmentation = augmentation

    def load_data(self, num_workers=4, pin_memory=True):
            """Load and prepare data into pytorch dataloader objects with multi-worker support and progress bars
            Args:
                num_workers (int, optional): Number of worker processes. Defaults to 4.
                pin_memory (bool, optional): Whether to pin memory in data loading. Defaults to True.
            Returns:
                tuple: Contains three DataLoader objects (train_loader, val_loader, test_loader).
            """
            train_df, val_df = train_test_split(self.df, test_size=0.2, random_state=42)

            # Create dataset instances
            train_dataset = MaskDataset(train_df, transform=self.augmentation,)
            val_dataset = MaskDataset(val_df, transform=False)
            test_dataset = MaskDataset(self.test_df, transform=False)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.bsize, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=False
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.bsize, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                drop_last=False
            )

            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.bsize, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                drop_last=False
            )


            # Initialize tracking variables
            train_zone_values = set()
            train_spot_values = set()
            val_zone_values = set()
            val_spot_values = set()
            test_zone_values = set()
            test_spot_values = set()
            image_shapes = set()
            zone_mask_shapes = set()
            spot_mask_shapes = set()

            print("\nLoading and verifying datasets...")

            
            # Process training data with single progress bar
            with tqdm(total=len(train_loader), desc="Training data", unit="batch") as pbar:
                for images, zone_masks, spot_masks in train_loader:
                    train_zone_values.update(torch.unique(zone_masks).cpu().numpy())
                    train_spot_values.update(torch.unique(spot_masks).cpu().numpy())
                    image_shapes.add(tuple(images.shape[1:]))
                    zone_mask_shapes.add(tuple(zone_masks.shape[1:]))
                    spot_mask_shapes.add(tuple(spot_masks.shape[1:]))
                    pbar.update(1)

            # Process validation data with single progress bar
            with tqdm(total=len(val_loader), desc="Validation data", unit="batch") as pbar:
                for images, zone_masks, spot_masks in val_loader:
                    val_zone_values.update(torch.unique(zone_masks).cpu().numpy())
                    val_spot_values.update(torch.unique(spot_masks).cpu().numpy())
                    image_shapes.add(tuple(images.shape[1:]))
                    zone_mask_shapes.add(tuple(zone_masks.shape[1:]))
                    spot_mask_shapes.add(tuple(spot_masks.shape[1:]))
                    pbar.update(1)

            # Process test data with single progress bar
            with tqdm(total=len(test_loader), desc="Test data", unit="batch") as pbar:
                for images, zone_masks, spot_masks in test_loader:
                    val_zone_values.update(torch.unique(zone_masks).cpu().numpy())
                    val_spot_values.update(torch.unique(spot_masks).cpu().numpy())
                    image_shapes.add(tuple(images.shape[1:]))
                    zone_mask_shapes.add(tuple(zone_masks.shape[1:]))
                    spot_mask_shapes.add(tuple(spot_masks.shape[1:]))
                    pbar.update(1)

            # Print summary statistics
            print("\n=== Dataset Statistics ===")
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            print(f"\nUnique zone values (train): {sorted(train_zone_values)}")
            print(f"Unique spot values (train): {sorted(train_spot_values)}")
            print(f"Unique zone values (val): {sorted(val_zone_values)}")
            print(f"Unique spot values (val): {sorted(val_spot_values)}")
            print(f"Unique zone values (test): {sorted(test_zone_values)}")
            print(f"Unique spot values (test): {sorted(test_spot_values)}")
            print(f"\nImage dimensions: {image_shapes}")
            print(f"Zone mask dimensions: {zone_mask_shapes}")
            print(f"Spot mask dimensions: {spot_mask_shapes}\n")
            

            # Check for empty batches
            print("Checking for empty batches...")
            for loader, name in [(train_loader, "Training"), (val_loader, "Validation"), (test_loader, "Test")]:
                empty_batches = 0
                with tqdm(total=len(loader), desc=f"Checking {name} batches", unit="batch") as pbar:
                    for images, zone_masks, spot_masks in loader:
                        if images.size(0) == 0 or zone_masks.size(0) == 0 or spot_masks.size(0) == 0:
                            empty_batches += 1
                        pbar.update(1)
                if empty_batches > 0:
                    print(f"WARNING: Found {empty_batches} empty batches in {name} dataset!")
                else:
                    print(f"No empty batches found in {name} dataset")

            return train_loader, val_loader, test_loader