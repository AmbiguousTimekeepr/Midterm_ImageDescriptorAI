import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self, output_dim=256):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Convolutional Layer 1: Input = (3, 256, 256) → Output = (32, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            # Convolutional Layer 2: Input = (32, 128, 128) → Output = (64, 64, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            # Convolutional Layer 3: Input = (64, 64, 64) → Output = (128, 32, 32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        # Flattened feature vector (128 * 32 * 32 → output_dim)
        self.fc = nn.Linear(128 * 32 * 32, output_dim)  

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through conv layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Fully connected layer to output_dim
        # x = torch.sigmoid(x)  # Sigmoid activation
        x= torch.relu(x)
        return x
    
    def preprocess(self, image_path, image=None):
        if image is None:
            image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
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
    
    # def train(self, train_loader, val_loader, optimizer, criterion, device="cpu", n_epochs=1000):
    #     self.to(device)
    #     for epoch in range(n_epochs):
    #         self.train()
    #         train_loss = 0.0
    #         for i, (images, _) in enumerate(train_loader):
    #             images = self.preprocess(None, images)
    #             images = images.to(device)
    #             optimizer.zero_grad()
    #             outputs = self(images)
    #             loss = criterion(outputs, outputs)
            
    #     print("Training Loss: {:.4f}".format(train_loss))