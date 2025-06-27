import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class BinaryClassifierEvaluator:
    """
    Utility class to predict on a single image using the saved binary model. Class is initialized with the image path,
    then the evaluate method is called to make a prediction.
    Attributes:
        img_path (str): Path to the local image.
    """
    def __init__(self, img_path):
        if not img_path:
            raise ValueError("Binary classifier is missing image path. Please provide an image.")
        self.img_path = img_path
        

    def evaluate(self):

        """
        Predict on a single image using the saved binary model. Prints the prediction and confidence level to the console.

        Returns:
            int: 1 if the image is classified as "Good", 0 if "Bad"
        """

        # Define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check if the image path exists
        img_path = self.img_path
        if not os.path.exists(img_path):
            print(f"Error: Image file not found at {img_path}")
            exit()

        # Use the same transform as during training
        transform = transforms.Compose([
            transforms.Resize((256, 512)),  # Resize images to 256x512
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet values
        ])

        # Initialize the model structure exactly as during training
        model = models.resnet18()  # Don't move to device yet
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 1),  # Change to a single output
            nn.Sigmoid()             # sigmoid for binary classification
        )

        # Load the saved model weights
        model_path = 'binary_classifier/saved_models/binary_classifier_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        # Move model to device AFTER loading weights
        model = model.to(device)
        model.eval()

        # Load and preprocess the image
        image = Image.open(img_path)
        input_tensor = transform(image).unsqueeze(0).to(device)  # Ensure tensor is on the right device

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            raw_confidence = output.item()
            prediction = raw_confidence > 0.5

        if prediction:
                confidence_percent = raw_confidence * 100
                print(f"Prediction: Good with {confidence_percent:.1f}% confidence")
        else:
                confidence_percent = (1 - raw_confidence) * 100
                print(f"Prediction: Bad with {confidence_percent:.1f}% confidence")

        if prediction == True:
            return 1
        else:
            return 0