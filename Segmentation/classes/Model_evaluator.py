import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from .UnetPlusPlus import UNetPlusPlus
import numpy as np


class ModelEvaluator:
    """
    A class for evaluating segmentation models that predict zones and spots of salmon filets.
    This class loads a trained model and provides methods for visualizing predictions,
    calculating metrics, and evaluating performance on validation data.
    Attributes:
        device (torch.device): Device to run the model on (CPU or CUDA)
        model (UNetPlusPlus): The loaded segmentation model
        model_path (str): Path to the saved model checkpoint
    Methods:
        view_single_result(val_loader, batch_idx, image_idx):
            Visualizes the model predictions alongside ground truth for a single image.
        predict(image):
            Makes zone and spot predictions on a single input image.
        viewLoaderItems(loader, idx): 
            Displays information about items from a data loader at specified index.
        calculate_dice_score(val_loader):
            Calculates mean Dice scores for zone and spot predictions.
    Parameters:
        model_path (str): Path to the model checkpoint file to load
    Raises:
        Exception: If there is an error loading the model checkpoint
    """

    def __init__(self, model_path):
    
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetPlusPlus(n_classes_zone=4, n_classes_spot=3).to(self.device)
        self.model_path = model_path
        
        try:
            checkpoint = torch.load(self.model_path, weights_only=True)
            
            # Load the model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # Set model to evaluation mode
        self.model.eval()

        # Extract and print epoch from checkpoint if available
        try:
            if 'epoch' in checkpoint:
                print(f"Model checkpoint from epoch: {checkpoint['epoch']}")
            else:
                print("Epoch information not available in checkpoint")
        except:
            print("Could not access epoch information")


    def view_single_result(self, val_loader, batch_idx, image_idx):
        """
        View a single result from the validation loader with both ground truth and predictions.
        
        Args:
            val_loader: DataLoader for validation data (or test data)
            batch_idx: Index of the batch to visualize
            image_idx: Index of the image within the batch to visualize
        Returns:
            None: Displays the image and masks using matplotlib
        """
        # Define class names for zones and spots
        zone_class_names = {
            0: 'Bakgrunn',
            1: 'Rygg',
            2: 'Buk1',
            3: 'Buk2'
        }
        spot_class_names = {
            0: 'Bakgrunn',
            1: 'Blod',
            2: 'Melanin'
        }
        
        # Create custom colormaps
        zone_colors = ['black', 'blue', 'green', 'purple']  # One color for each zone + background
        spot_colors = ['black', 'red', 'yellow']  # One color for each spot type + background
        zone_n_classes = 4  # Background + 3 zones
        spot_n_classes = 3  # Background + 2 spot types
        
        zone_cmap = mcolors.ListedColormap(zone_colors)
        spot_cmap = mcolors.ListedColormap(spot_colors)
        zone_norm = mcolors.BoundaryNorm(boundaries=range(zone_n_classes + 1), ncolors=zone_n_classes)
        spot_norm = mcolors.BoundaryNorm(boundaries=range(spot_n_classes + 1), ncolors=spot_n_classes)

        # Iterate through loader to find the specified batch
        for i, (images, zone_masks, spot_masks) in enumerate(val_loader):
            if i == batch_idx:
                # Ensure image_idx is within batch size
                batch_size = images.size(0)
                if image_idx >= batch_size:
                    print(f"Image index {image_idx} out of range. Batch size is {batch_size}")
                    return
                
                # Move to device and make prediction
                images = images.to(self.device)
                with torch.no_grad():
                    zone_output, spot_output = self.model(images)
                
                # Get predicted masks
                predicted_zone = torch.argmax(zone_output, dim=1)
                predicted_spot = torch.argmax(spot_output, dim=1)
                
                # Move tensors to CPU and select specific image
                img = images[image_idx].cpu()
                true_zone_mask = zone_masks[image_idx].cpu()
                true_spot_mask = spot_masks[image_idx].cpu()
                pred_zone_mask = predicted_zone[image_idx].cpu()
                pred_spot_mask = predicted_spot[image_idx].cpu()

                # Create figure with specific size and spacing
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

                # Original image
                ax1 = fig.add_subplot(gs[0, 0])
                if img.shape[0] == 3:  # RGB image
                    # Convert from (C, H, W) to (H, W, C)
                    img_display = img.permute(1, 2, 0)
                    # Normalize if needed
                    if img_display.max() > 1.0:
                        img_display = img_display / 255.0
                    ax1.imshow(img_display)
                elif img.shape[0] == 1:  # Grayscale image with channel dimension
                    ax1.imshow(img.squeeze(0), cmap='gray')
                else:  # Grayscale without channel dimension
                    ax1.imshow(img, cmap='gray')
                ax1.set_title('Original Image')
                ax1.axis('off')

                # True zone mask
                ax2 = fig.add_subplot(gs[0, 1])
                im_zone = ax2.imshow(true_zone_mask, cmap=zone_cmap, norm=zone_norm, interpolation='nearest')
                ax2.set_title('True Zone Mask')
                ax2.axis('off')

                # Predicted zone mask
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(pred_zone_mask, cmap=zone_cmap, norm=zone_norm, interpolation='nearest')
                ax3.set_title('Predicted Zone Mask')
                ax3.axis('off')

                # True spot mask
                ax4 = fig.add_subplot(gs[1, 1])
                im_spot = ax4.imshow(true_spot_mask, cmap=spot_cmap, norm=spot_norm, interpolation='nearest')
                ax4.set_title('True Spot Mask')
                ax4.axis('off')

                # Predicted spot mask
                ax5 = fig.add_subplot(gs[1, 2])
                ax5.imshow(pred_spot_mask, cmap=spot_cmap, norm=spot_norm, interpolation='nearest')
                ax5.set_title('Predicted Spot Mask')
                ax5.axis('off')

                # Add colorbars with class names
                zone_cax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
                spot_cax = fig.add_axes([0.92, 0.1, 0.02, 0.35])

                # Zone colorbar with class names
                cbar_zone = plt.colorbar(im_zone, cax=zone_cax)
                cbar_zone.set_ticks(np.arange(zone_n_classes) + 0.5)  # Center ticks
                cbar_zone.set_ticklabels([zone_class_names[i] for i in range(zone_n_classes)])

                # Spot colorbar with class names
                cbar_spot = plt.colorbar(im_spot, cax=spot_cax)
                cbar_spot.set_ticks(np.arange(spot_n_classes) + 0.5)  # Center ticks
                cbar_spot.set_ticklabels([spot_class_names[i] for i in range(spot_n_classes)])

                # Adjust layout
                plt.subplots_adjust(right=0.9)
                
                plt.show()
                return  # Exit after showing the requested image
    
        # If we get here, the batch_idx was out of range
        print(f"Batch index {batch_idx} out of range. Dataset has {len(val_loader)} batches.")
        
    def predict(self, image):
        """
        Predict on a single image using the saved segmentation model.
        Args:
            image (numpy.ndarray): Input image to predict on. Should be a 3D array (H, W, C).
        Returns:
            tuple: Predicted zone and spot masks as 2D numpy arrays.
        """ 
        # Convert numpy array to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            
        # Move to device and ensure model is in eval mode
        image_tensor = image_tensor.to(self.device)
        self.model.eval()

        # Make prediction
        with torch.no_grad():
            zone_output, spot_output = self.model(image_tensor)
            predicted_zone = torch.argmax(zone_output, dim=1)
            predicted_spot = torch.argmax(spot_output, dim=1)

        # Move predictions back to CPU and convert to numpy
        predicted_zone = predicted_zone.cpu().numpy()
        predicted_spot = predicted_spot.cpu().numpy()

        return predicted_zone[0], predicted_spot[0]
        
    
    def viewLoaderItems(self, loader, idx):
        """
        View items from a data loader at a specific index.
        Args:
            loader: DataLoader object containing the dataset
            idx: Index of the item to view
        Returns:
            tuple: Image tensor, zone masks, and spot masks as numpy arrays.
        """
        for i, (images, zone_masks, spot_masks) in enumerate(loader):
            if i == idx:
                images = images.cpu().numpy()
                print(f"Image tensor type: {type(images)}, shape: {images.shape}, dtype: {images.dtype}")
                zone_masks = zone_masks.cpu().numpy()
                spot_masks = spot_masks.cpu().numpy()
                # Print unique values in image and masks
                print("Unique values in zone masks:", np.unique(zone_masks))
                print("Unique values in spot masks:", np.unique(spot_masks))
                return images, zone_masks, spot_masks
        return None, None, None  # Return None if idx is out of range
    
    def calculate_dice_score(self, val_loader):
        """
        Calculate the Dice similarity coefficient for zone and spot segmentation masks.
        This method evaluates the model's segmentation performance by computing Dice scores
        for both zone and spot predictions. The Dice score is calculated per class,
        excluding the background class (class_id 0).
        Parameters:
        -----------
        val_loader : torch.utils.data.DataLoader
            DataLoader containing validation or test data in the format of (images, zone_masks, spot_masks)
        Returns:
        --------
        tuple(float, float)
            A tuple containing:
            - zone_dice: Mean Dice score for zone segmentation (3 classes)
            - spot_dice: Mean Dice score for spot segmentation (2 classes)
        Notes:
        ------
        - Even though the paramaeter is called val_loader, it can be used for test data as well (test_loader) or any other data loader of the same format.
        - Zone segmentation has 3 classes (excluding background)
        - Spot segmentation has 2 classes (excluding background)
        - The method uses no_grad() for memory efficiency during validation
        - A small epsilon (1e-8) is added to prevent division by zero
        """

        zone_dice_scores = []
        spot_dice_scores = []

        with torch.no_grad():
            for images, zone_masks, spot_masks in val_loader:
                # Move to device and make predictions
                images = images.to(self.device)
                zone_output, spot_output = self.model(images)
                
                # Get predicted masks
                predicted_zone = torch.argmax(zone_output, dim=1)
                predicted_spot = torch.argmax(spot_output, dim=1)
                
                # Calculate Dice score for each class in zone masks (excluding background)
                for class_id in range(1, 4):  # 3 zone classes (excluding background)
                    true_class = (zone_masks == class_id)
                    pred_class = (predicted_zone.cpu() == class_id)
                    intersection = (true_class & pred_class).sum().item()
                    total = true_class.sum().item() + pred_class.sum().item()
                    dice = (2 * intersection) / (total + 1e-8)  # add small epsilon to avoid division by zero
                    zone_dice_scores.append(dice)

                # Calculate Dice score for each class in spot masks (excluding background)
                for class_id in range(1, 3):  # 2 spot classes (excluding background)
                    true_class = (spot_masks == class_id)
                    pred_class = (predicted_spot.cpu() == class_id)
                    intersection = (true_class & pred_class).sum().item()
                    total = true_class.sum().item() + pred_class.sum().item()
                    dice = (2 * intersection) / (total + 1e-8)
                    spot_dice_scores.append(dice)

        # Calculate average Dice scores
        zone_dice = np.mean(zone_dice_scores)
        spot_dice = np.mean(spot_dice_scores)
        
        print("Mean dice score zones:", zone_dice)
        print("Mean dice score spots:", spot_dice)



