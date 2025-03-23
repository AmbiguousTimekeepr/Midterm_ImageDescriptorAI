import CNN
import CNN_LSTM
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn


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
train_pairs = CNN_LSTM.load_qa_data_subset(r'Datasets\VQA\vqa_dataset.csv')
val_pairs = CNN_LSTM.load_qa_data_subset(r'Datasets\VQA\valid_vqa_dataset.csv')
test_pairs = CNN_LSTM.load_qa_data_subset(r'Datasets\VQA\test_vqa_dataset.csv')

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

