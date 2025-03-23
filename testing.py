import CNN
import CNN_LSTM
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn

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


# dataset = CNN.DogDataset("Datasets/CNN/dogs.csv", "Datasets/CNN")
# print(dataset.label_map)  # Print the mapping from class names to integers
# print(dataset[0])  # Check if image and label are loaded correctly

# train_loader, valid_loader, test_loader = CNN.get_dataloaders("Datasets\CNN\dogs.csv","Datasets\CNN", batch_size=32)
# CNN.train_model(cnn_model, train_loader, valid_loader, epochs=15, lr=0.001, device="cuda")
# print("Training complete")
# print("Evaluation on test set:", CNN.evaluate_model(cnn_model, test_loader, device="cuda"))

# print(cnn_model.classify_image(image_path))  # Expected: "Dalmatian"


# Create properly initialized vocabulary with special tokens

vocab = CNN_LSTM.prerequisite()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# train_pairs = CNN_LSTM.load_qa_data(r'Datasets\VQA\vqa_dataset.csv')
# val_pairs = CNN_LSTM.load_qa_data(r'Datasets\VQA\valid_vqa_dataset.csv')
# CNN_LSTM.train_model(train_pairs, val_pairs, vocab=vocab, num_epochs=50)



vocab = CNN_LSTM.prerequisite()
train_pairs = CNN_LSTM.load_qa_data_subset(r'Datasets\external\vqa_train_split.csv')
val_pairs = CNN_LSTM.load_qa_data_subset(r'Datasets\external\vqa_valid_split.csv')
test_pairs = CNN_LSTM.load_qa_data_subset(r'Datasets\external\vqa_test_split.csv')

pretrained_model, house_trained_model = CNN_LSTM.train_model(train_pairs, val_pairs, vocab=vocab, num_epochs=30)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = CNN_LSTM.VQAModel(vocab_size=len(vocab))  # Adjust parameters as needed
pretrained_model.load_state_dict(torch.load("pretrained_vqa_model.pth"))
pretrained_model.to(device)
pretrained_model.eval()

house_trained_model = CNN_LSTM.VQAModel(vocab_size=len(vocab))
house_trained_model.load_state_dict(torch.load("house-trained_vqa_model.pth"))
pretrained_model.to(device)
house_trained_model.eval()


test_dataset = CNN_LSTM.DogBreedVQADataset(test_pairs, vocab, transform)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=CNN_LSTM.collate_fn)

# Now you can call compare_models
# CNN_LSTM.compare_models(pretrained_model, house_trained_model, test_loader, device)
criterion = nn.CrossEntropyLoss(ignore_index=pretrained_model.pad_idx)
criterion = nn.CrossEntropyLoss(ignore_index=house_trained_model.pad_idx)

print("Pretrained model evaluation:")
CNN_LSTM.evaluate(pretrained_model, test_loader, criterion, device= device, vocab= vocab, print_samples=3)
print("House-trained model evaluation:")
CNN_LSTM.evaluate(house_trained_model, test_loader, criterion, device= device, vocab= vocab, print_samples=3)
# (model, dataloader, criterion, device, vocab=None, print_samples=5)

