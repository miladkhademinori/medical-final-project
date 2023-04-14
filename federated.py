import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader


# Define a basic block for the ResNet
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# Define the three-layer ResNet for contrastive modeling
class ContrastiveResNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes=128):
        super(ContrastiveResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, num_classes, 2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


# Define a basic block for the ResNet
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# Define the three-layer ResNet for a Variational Autoencoder (VAE)
class VAE_ResNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=256, latent_dim=64):
        super(VAE_ResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)

        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)

        self.layer4 = self._make_layer(256, 128, 2, stride=1)
        self.layer5 = self._make_layer(128, 64, 2, stride=1)

        self.conv_transpose = torch.nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = torch.nn.Tanh()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return torch.nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std


# Define the mse loss function
def mse_loss(x_hat, x):
    return torch.nn.functional.mse_loss(x_hat, x, reduction='sum')


# Define the kld loss function
def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Define the three-layer ResNet for a classifier
class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Three ResNet blocks
        self.layer2 = self._make_resnet_block(64, 64, 2)
        self.layer3 = self._make_resnet_block(64, 128, 2)
        self.layer4 = self._make_resnet_block(128, 256, 2)

        # Final fully connected layer
        self.fc = nn.Linear(256, num_classes)

        # Softmax activation
        self.softmax = nn.Softmax(dim=1)

    def _make_resnet_block(self, in_channels, out_channels, num_blocks):
        layers = []

        # First convolutional layer of the block
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Middle ResNet blocks
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    # Define the forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.avg_pool2d(x, kernel_size=4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x


# Define the decoder
def decode(self, z):
    z = z.view(-1, self.hidden_dim, 1, 1)

    x = self.layer4(z)
    x = self.layer5(x)

    x = self.conv_transpose(x)
    x = self.tanh(x)

    return x


# Define the forward pass
def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    x_hat = self.decode(z)

    return x_hat, mu, logvar


# Binary Cross Entropy Loss function
class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
        return loss


# Define the training function
def train(model, train_loader, optimizer, criterion, epochs=10):
    # Assume y_true and y_pred are tensors of shape (batch_size,)
    # where 0 represents negative and 1 represents positive examples

    # Set the model to training mode
    model.train()

    # Loop over the epochs
    for epoch in range(epochs):
        data = tqdm(train_loader, desc="Epoch {}".format(epoch + 1), leave=False)
        for x, y_true in data:
            y_pred = model(x)
            loss = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data.set_postfix(loss=loss.item())

        print("Epoch {} Loss: {}".format(epoch + 1, loss.item()))

    # Calculate true positives, false positives, true negatives, and false negatives
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    # Calculate precision and accuracy
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    print("Precision:", precision)
    print("Accuracy:", accuracy)

    return precision, accuracy


def visulize_latent_space(model, test_loader, device):
    model.eval()
    z = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            z.append(mu.cpu().numpy())
    z = np.concatenate(z)
    return z


# Define the main function
def main():
    # Define the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # If you don't have a GPU, you cannot use a CPU instead
        raise Exception("No GPU found, please use a GPU to train your model")

    # Define the data loaders
    # Load the malaria dataset from the following directory
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root="data/cell_images",
            transform=transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        ),
        batch_size=32,
        shuffle=True
    )

    # Define the model
    model = ResNetVAE(in_channels=3, latent_dim=32)

    # Define the loss function
    criterion = BCELoss()

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    train(model, train_loader, optimizer, criterion, epochs=10)

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Load the model
    model.load_state_dict(torch.load("model.pth"))

    # Set the model to evaluation mode
    model.eval()

    # Generate a random latent vector
    z = torch.randn(1, 32)

    # Generate an image from the latent vector
    x = model.decode(z)
