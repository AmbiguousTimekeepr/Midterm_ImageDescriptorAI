import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os

import datetime

class CNN(nn.Module):
    def __init__(self, output_dim=256):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # first prtototype: 3conv - 2fc
            # # Convolutional Layer 1: Input = (3, 256, 256) → Output = (32, 128, 128)
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  

            # # Convolutional Layer 2: Input = (32, 128, 128) → Output = (64, 64, 64)
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  

            # # Convolutional Layer 3: Input = (64, 64, 64) → Output = (128, 32, 32)
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)  
            
            # Prototype No.2: 7 conv - 3 fc
            # # Convolutional Layer 1: (3, 256, 256) → (32, 256, 256)
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 128, 128)

            # # Convolutional Layer 2: (32, 128, 128) → (64, 128, 128)
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Fixed kernel size
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 64, 64)

            # # Convolutional Layer 3: (64, 64, 64) → (128, 64, 64)
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Fixed kernel size
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 32, 32)

            # # Convolutional Layer 4: (128, 32, 32) → (256, 32, 32)
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Fixed kernel size
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 16, 16)

            # # Convolutional Layer 5: (256, 16, 16) → (512, 16, 16)
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Fixed kernel size
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (512, 8, 8)

            # # Convolutional Layer 6: (512, 8, 8) → (512, 8, 8)
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            # # Convolutional Layer 7: (512, 8, 8) → (512, 4, 4)
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)  # (512, 4, 4)
            
            # Prototype No.3: 4 conv - 2 fc
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # (64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (64, 56, 56)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (128, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 28, 28)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (256, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 14, 14)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (512, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (512, 7, 7)
            
            
        )
        
        # # Flattened feature vector (128 * 32 * 32 → output_dim)
        # self.fc = nn.Linear(512 * 4 * 4, 1024) 
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1024, output_dim) 
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        # print("Shape after conv layers:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # print("Shape after flattening:", x.shape)
        x = self.fc_layers(x)  # Fully connected layers
        return x

    # def forward(self, x):
    #     x = self.conv_layers(x)  # Pass through conv layers
    #     x = x.view(x.size(0), -1)  # Flatten
    #     x = self.relu(self.fc(x))  # Fully connected layer to second fc layer
    #     # x = torch.sigmoid(x)  # Sigmoid activation
    #     x = self.dropout(x)  # Dropout
    #     x = self.fc2(x)
    #     # x = torch.softmax(x, dim=1)
    #     return x
    
    def preprocess(self, image_path, image=None):
        if image is None:
            image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        return image

    def extract_features(self, image_path, device="cpu"):
        image = self.preprocess(image_path)
        image = image.to(device)
        self.eval()
        with torch.no_grad():
            features = self(image)
        return features
    
    def classify_image(self, image_path, device="cpu"):
        image = self.preprocess(image_path)
        image = image.to(device)

        self.eval()
        with torch.no_grad():
            output = self(image)
            probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
            top_probs, top_classes = torch.topk(probabilities, 10, dim=1)  # Get top 10

        # Convert to Python list
        top_classes = top_classes.squeeze(0).tolist()  # Class indices
        top_probs = top_probs.squeeze(0).tolist()  # Confidence scores

        return list(zip(top_classes, top_probs))  # Returns list of (class_index, confidence)



class DogDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.label_map = {label: idx for idx, label in enumerate(self.data.iloc[:, 1].unique())}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])  # Filepath
        label_str = self.data.iloc[idx, 1]  # This is the class name (string)

        # Convert class name to integer index
        label = self.label_map[label_str]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    
# def train_model(model, train_loader, valid_loader, epochs=100, lr=0.003, device="cpu"):
#     print("Training Model...")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     model.to(device)
    
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
#     print("Training Complete!")
    
#     save_dir = "trained_models"
#     os.makedirs(save_dir, exist_ok=True)
    
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     model_filename = os.path.join(save_dir, f"model_{timestamp}.pth")
    
#     torch.save({
#         'epoch': epochs,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': running_loss / len(train_loader)
#     }, model_filename)

#     print(f"Model saved as {model_filename}")
    
    
def train_model(model, train_loader, valid_loader, epochs=100, lr=0.003, device="cpu"):
    print("Training Model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    print("Training Complete!")
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = os.path.join(save_dir, f"model_{timestamp}.pth")
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss / len(train_loader)
    }, model_filename)
    print(f"Model saved as {model_filename}")

def evaluate_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc


def get_dataloaders(csv_path, root_path, batch_size=32):
    transform = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        "valid": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    }
    
    train_dataset = DogDataset(csv_path, root_path, transform=transform['train'])
    valid_dataset = DogDataset(csv_path, root_path, transform=transform['valid'])
    test_dataset = DogDataset(csv_path, root_path, transform=transform['valid'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader