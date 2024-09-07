# train/train.py

import sys
sys.path.append('..')

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from data.dataset import RotatedMNISTDataset
from model.model import CNN
import torch.nn as nn

def train_model():
    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='C:/Users/uddip/Downloads/MNIST', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='C:/Users/uddip/Downloads/MNIST', train=False, download=True, transform=transform)

    rotated_train_dataset = RotatedMNISTDataset(train_dataset)
    rotated_test_dataset = RotatedMNISTDataset(test_dataset)

    train_loader = DataLoader(rotated_train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Pretext Task Training (Rotation Prediction)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Save the pre-trained model
    torch.save(model.state_dict(), 'pretrained_model.pth')
