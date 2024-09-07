# finetune/finetune.py

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

def fine_tune_and_evaluate():
    # Load the pre-trained model
    model = CNN()
    model.load_state_dict(torch.load('pretrained_model.pth'))

    # Modify the last layer for digit classification
    model.fc2 = nn.Linear(128, 10)  # Adjust the last layer for 10 classes (digits 0-9)

    # Freeze the earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer (fc2) so it can be trained
    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True

    # Define the optimizer for the last layer
    optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Fine-tuning on the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='C:/Users/uddip/Downloads/MNIST', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='C:/Users/uddip/Downloads/MNIST', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

    # Evaluate the Model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10,000 test images: {100 * correct / total:.2f}%')
