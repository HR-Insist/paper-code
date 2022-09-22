from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ViT
import os


def train():
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    mnist_train = datasets.MNIST("download", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_model = ViT.ViT(
        image_size=28,
        patch_size=7,
        num_classes=10,
        channels=1,
        dim=512,
        depth=1,
        heads=2,
        mlp_dim=1024,
        dropout=0,
        emb_dropout=0)
    loss_fn = nn.CrossEntropyLoss()
    mnist_model = mnist_model.to(device)
    opitimizer = optim.Adam(mnist_model.parameters(), lr=0.00001)
    mnist_model.train()
    for epoch in range(epochs):
        total_loss = 0
        corrects = 0
        num = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            opitimizer.zero_grad()
            outputs = mnist_model(batch_X)
            _, pred = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, batch_Y)
            loss.backward()
            opitimizer.step()
            total_loss += loss.item()
            corrects = torch.sum(pred==batch_Y.data)
            num += batch_size
            print("epoch:", epoch+1, ", loss:", total_loss / float(num), ", acc:",corrects.item() / float(batch_size))
            # print(epoch, total_loss / float(num), corrects.item() / float(batch_size))
    torch.save(mnist_model, './mnist_model.pth')


if __name__ == '__main__':
    train()
