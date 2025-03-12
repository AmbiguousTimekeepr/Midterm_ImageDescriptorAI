import CNN
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load the real image
image_path = "download.jfif"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Ensure it's RGB

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match CNN input
    transforms.ToTensor(),  # Convert image to tensor (C, H, W)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Apply transformations
image = transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 256, 256)

# Ensure it's on the correct device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Print shape to verify
print("Image tensor shape:", image.shape)  # Should be (1, 3, 256, 256)

# Initialize the CNN model
cnn_model = CNN.CNN(output_dim=256).to(device)

# Extract features from the image
cnn_model.eval()  # Set model to evaluation mode (important!)
with torch.no_grad():  # No gradients needed
    features = cnn_model(image)

# Print the extracted feature vector
print("Extracted feature vector shape:", features.shape)  # Expected: (1, 256)

