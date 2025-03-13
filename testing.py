import CNN
import cv2
import numpy as np
import torch


# Load the real image
image_path = "download.jfif"  # Replace with your image path

# Initialize the CNN model
cnn_model = CNN.CNN(output_dim=256)

# Extract features from the image
features = cnn_model.extract_features(image_path)

# Print the extracted feature vector
print("Extracted feature vector shape:", features)  # Expected: (1, 256)

