
from .Model_evaluator import ModelEvaluator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import matplotlib.colors as mcolors

class ResultUtil:
    """
    A utility class for handling and visualizing segmentation model predictions.
    This class provides functionality to process images through a segmentation model,
    combine different prediction types, and visualize results.
    (blood, melanin) in images.
    Attributes:
        model_path (str): Path to the trained model file.
    Methods:
        combineResults(image):
            Combines zone and spot predictions into a single segmentation mask.
        displayNumpy(image1, image2_path):
            Displays segmentation results alongside the original image.
        getPredictedZone(image):
            Analyzes zone predictions and returns dictionaries of blood and melanin overlaps.
        pipeline_prediction(image_path):
            Runs the complete prediction pipeline on an input image.
    Example:
        >>> util = ResultUtil("path/to/model")
        >>> result = util.pipeline_prediction("path/to/image.jpg")
        >>> print(result)
        {
            'melanin': True,
            'melanin_location': ['Rygg'],
            'blood': False,
            'blood_location': []
    Notes:
        - Input images are automatically resized to (512, 256)
        - Supports various image formats including WebP
        - Output segmentation uses the following class mapping:
            0: background
            1: Back
            2: Belly forward
            3: Belly rear
            4: blood
            5: melanin
    """

    def __init__(self, model_path):
        self.model_path = model_path
        
    def combineResults(self, image):
        """
        Combines predictions of zones and spots to create a final segmentation mask.
        This method processes an image through the saved segmentation model and 
        combines the dual output into a single segmentation mask. The zone predictions are used as a base,
        with spot predictions (blood and melanin) overlaid on top.
        Parameters
        ----------
        image :
            Input image to process. Can be:
            - Path to image file (str)
            - PIL Image object
            - Numpy array of shape (H,W,C) or (B,H,W,C)
        Returns
        -------
        numpy.ndarray
            Combined segmentation mask where:
            - Original zone predictions (0-3) are preserved
            - Blood spots are marked as 4
            - Melanin spots are marked as 5
        Notes
        -----
        - If input is 4D with batch dimension, only first image is processed
        - Input image is automatically converted to numpy array if needed
        - Original zone predictions are overwritten where spot predictions exist
        """

        if isinstance(image, str):
            image = Image.open(image)

        # Ensure input is numpy array
        image = np.asarray(image)
        
        # Print shape for debugging
        print(f"Input shape: {image.shape}")
        
        # Handle batch dimension if present
        if image.ndim == 4:  # [B,H,W,C]
            image = image[0]  # Remove batch dimension
        
        print(f"Preprocessed shape: {image.shape}")
        
        # Get predictions
        evaluator = ModelEvaluator(self.model_path)
        zone_prediction, spot_prediction = evaluator.predict(image)

        print("Unique values in spot_prediction in combineResults:", np.unique(spot_prediction))
        print("Number of pixels classified as blood in spot_prediction mask:", np.sum(spot_prediction == 1))
        print("Number of pixels classified as melanin in spot_prediction mask:", np.sum(spot_prediction == 2))
        
        # Create output mask
        result = zone_prediction.copy()
        
        # Set blood (4) and melanin (5) locations
        result[spot_prediction == 1] = 4  # Blood
        result[spot_prediction == 2] = 5  # Melanin

        # Unique values in result
        print("Unique values in result:", np.unique(result))

        print(f"Result tensor type: {type(result)}, shape: {result.shape}, dtype: {result.dtype}")

        return result
    
    def displayNumpy(self, image1: np.ndarray, image2_path: str):
            """
            Displays a predicted segmentation mask numpy array and an original image side-by-side.

            Args:
                image1: A numpy array representing the segmentation map (values 0-5).
                image2_path: The file path to the original image (e.g., WebP).
            Notes:
                - The segmentation map is displayed with a color map for better visualization.
                - The original image is designed to only be displayed in the main pipeline script "run_pipeline.ipynb", otherwise a placeholder image is provided.
                - The function handles different image modes (RGB, RGBA, P, L) using Pillow.
            """
            # --- Load and Prepare image2 (Original Image) ---
            try:
                # Open the image using Pillow
                img2_pil = Image.open(image2_path)
                # Convert to RGB if it's in a different mode (like P or L)
                # Ensure alpha channel is handled if present, often convert to RGB or RGBA
                if img2_pil.mode in ('P', 'L'):
                    img2_pil = img2_pil.convert('RGB')
                elif img2_pil.mode == 'RGBA':
                    # Matplotlib handles RGBA, no need to convert
                    pass # Or convert to RGB if you don't want alpha: img2_pil = img2_pil.convert('RGB')
                elif img2_pil.mode != 'RGB':
                    img2_pil = img2_pil.convert('RGB') # Convert other modes to RGB

                # Convert the PIL image to a numpy array for matplotlib
                image2_np = np.array(img2_pil)

            except FileNotFoundError:
                print(f"Error: Image file not found at {image2_path}")
                return
            except Exception as e:
                print(f"Error loading or processing image file {image2_path}: {e}")
                return

            # --- Prepare image1 (Segmentation) ---
            # Define colors for each label (0-5)
            # You can customize these colors
            colors = [
                'black',    # 0: background
                'blue',     # 1: Back
                'green',    # 2: Belly rear
                'yellow',   # 3: Belly forward
                'red',      # 4: blood
                'purple'    # 5: melanin
            ]

            # Create a colormap from the defined colors
            cmap = mcolors.ListedColormap(colors)

            # Define the boundaries for the colormap (between the integer values)
            # This ensures each integer gets a distinct color block
            bounds = [i - 0.5 for i in range(len(colors) + 1)] # e.g., [-0.5, 0.5, 1.5, ..., 5.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Define the labels for the colorbar
            labels = [
                '0: background',
                '1: Back',
                '2: Belly forward',
                '3: Belly rear',
                '4: blood',
                '5: melanin'
            ]

            # --- Create the Plot ---
            # Create a figure with two subplots side by side (1 row, 2 columns)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Adjust figsize as needed

            # Plot image1 (Segmentation) on the left subplot
            im1 = axes[0].imshow(image1, cmap=cmap, norm=norm, interpolation='nearest')
            axes[0].set_title('Segmentation Prediction')
            axes[0].axis('off') # Turn off axes ticks and labels

            # Plot image2 (Original Image) on the right subplot
            axes[1].imshow(image2_np)
            axes[1].set_title('Original Image')
            axes[1].axis('off') # Turn off axes ticks and labels

            # --- Add Colorbar (Legend) for image1 ---
            # Create a colorbar associated with the image1 plot
            # We need to set the ticks and labels manually to match our discrete categories
            cbar = fig.colorbar(im1, ax=axes[0], ticks=np.arange(len(colors))) # Set ticks at integer positions
            cbar.ax.set_yticklabels(labels) # Set the labels for the ticks

            # Adjust layout to prevent titles/labels from overlapping
            plt.tight_layout()

            # Display the plot
            plt.show()

    def getPredictedZone(self, image):
        """
        Analyzes an image to detect zones and predict blood/melanin spots within those zones.
        This method processes an input image through the saved trained segmentation model to identify different body zones 
        (Rygg, Buk foran, Buk bak) and detect blood and melanin spots within these zones. It returns 
        dictionaries containing information about which zones contain blood and melanin spots.
        Args:
            image (numpy.ndarray): Input image array. Can be either 3D (H,W,C) or 4D (B,H,W,C) where:
                - B: Batch size (optional)
                - H: Height
                - W: Width
                - C: Channels
        Returns:
            tuple: Two dictionaries containing zone information:
                - blood_overlap_dict: Dictionary with 'zone' key containing list of zones with blood spots
                - melanin_overlap_dict: Dictionary with 'zone' key containing list of zones with melanin spots
        Note:
            - The method automatically handles both batched and unbatched input images
            - Prints debug information including shape and pixel counts for each class
            - Zone labels are mapped as: 1=Rygg, 2=Buk foran, 3=Buk bak
        """


        image = np.asarray(image)
        
        # Print shape for debugging
        print(f"Input shape: {image.shape}")
        
        # Handle batch dimension if present
        if image.ndim == 4:  # [B,H,W,C]
            image = image[0]  # Remove batch dimension
        
        print(f"Preprocessed shape: {image.shape}")
        
        # Get predictions
        evaluator = ModelEvaluator(self.model_path)
        zone_prediction, spot_prediction = evaluator.predict(image)
        
        # Get combined result
        result = self.combineResults(image)

        # Create separate dictionaries for blood and melanin overlaps
        blood_overlap_dict = {}
        melanin_overlap_dict = {}

        zone_labels = ["", "Rygg", "Buk foran", "Buk bak"]
        
        blood_overlap_zones = []
        melanin_overlap_zones = []

        for zone_value in [1, 2, 3]: 
            blood_overlap = np.logical_and(zone_prediction == zone_value, spot_prediction == 1)
            if np.any(blood_overlap):
                blood_overlap_zones.append(zone_labels[zone_value]) # Append to the list

        blood_overlap_dict["zone"] = blood_overlap_zones
        
        # Find overlaps for melanin (2 in spot_prediction)
        for zone_value in [1, 2, 3]:
            melanin_overlap = np.logical_and(zone_prediction == zone_value, spot_prediction == 2)
            if np.any(melanin_overlap):
                melanin_overlap_zones.append(zone_labels[zone_value])

        melanin_overlap_dict["zone"] = melanin_overlap_zones

        print("Overlap zones blood", blood_overlap_zones)
        print("Overlap zones melanin", melanin_overlap_zones)

        # Count and print the total number of pixels for each class
        for i in range(6):  # 0 to 5
            pixel_count = np.sum(result == i)
            print(f"Class {i} count: {pixel_count} pixels")

        return blood_overlap_dict, melanin_overlap_dict
    

    def pipeline_prediction(self, image_path):
        """
        Main function for running the segmentation model in the pipeline on a given image.
        It processes
        an input image through the saved segmentation model and returns detected features and their locations.
        The function utilizes other helper methods of this class for processing and combining results.
        Parameters
        ----------
        image_path : str
            Path to the input image file that needs to be processed
        Returns
        -------
        dict
            A dictionary containing the segmentation results with the following keys:
            - 'melanin': bool, indicates if melanin is detected
            - 'melanin_location': list, locations where melanin is detected
            - 'blood': bool, indicates if blood is detected
            - 'blood_location': list, locations where blood is detected
        Notes
        -----
        The function performs the following steps:
        1. Resizes the input image to (512, 256)
        2. Converts the image to PyTorch tensor format
        3. Processes the image through segmentation models
        4. Combines and displays results
        5. Analyzes predicted zones for blood and melanin
        The function depends on other class methods:
        - combineResults()
        - displayNumpy()
        - getPredictedZone()
        """


        target_size = (512, 256) # height, width (Note: torchvision.transforms.Resize expects (h, w))

        transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR), # Resize the PIL image
            T.ToTensor() # Converts PIL Image (HWC, uint8) to Tensor (CHW, float32, [0.0, 1.0])
        ])

        # --- Loading and processing the image ---
        img = Image.open(image_path).convert("RGB") # Load and ensure it's in RGB format

        # Apply the transformations
        img_tensor = transform(img) # Result is a PyTorch Tensor, shape (C, H, W), dtype float32

        # This changes shape from (C, H, W) to (1, C, H, W)
        img_tensor_batch = img_tensor.unsqueeze(0)

        # The shape will be (1, C, H, W) and dtype likely float32
        img_numpy_batch = img_tensor_batch.cpu().numpy()

        result = self.combineResults(img_numpy_batch) # Pass the NumPy array batch

        self.displayNumpy(result, image_path) 
        blood_dict, melanin_dict = self.getPredictedZone(img_numpy_batch)

        result_dict = {
            "melanin": True if melanin_dict["zone"].__len__() > 0 else False,
            "melanin_location": melanin_dict["zone"],
            "blood": True if blood_dict["zone"].__len__() > 0 else False,
            "blood_location": blood_dict["zone"]
        }

        return result_dict



