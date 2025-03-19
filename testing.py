import CNN
import cv2
import numpy as np
import torch


# Load the real image
image_path = "basenji.webp"  # Replace with your image path

# Initialize the CNN model
cnn_model = CNN.CNN(output_dim=224)

# # Load the trained model
# model_path = "trained_models\model_2025-03-18_19-22-11.pth"  # Replace with actual filename
# checkpoint = torch.load(model_path)

# cnn_model.load_state_dict(checkpoint['model_state_dict'])

# Extract features from the image
# features = cnn_model.extract_features(image_path)

# Print the extracted feature vector
# print("Extracted feature vector shape:", features)  # Expected: (1, 256)


dataset = CNN.DogDataset("Datasets/CNN/dogs.csv", "Datasets/CNN")
print(dataset.label_map)  # Print the mapping from class names to integers
print(dataset[0])  # Check if image and label are loaded correctly

train_loader, valid_loader, test_loader = CNN.get_dataloaders("Datasets\CNN\dogs.csv","Datasets\CNN", batch_size=32)
CNN.train_model(cnn_model, train_loader, valid_loader, epochs=15, lr=0.001, device="cpu")
print("Training complete")
print("Evaluation on test set:", CNN.evaluate_model(cnn_model, test_loader, device="cpu"))

print(cnn_model.classify_image(image_path))  # Expected: "Dalmatian"