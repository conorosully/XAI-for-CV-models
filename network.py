import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

#================================================================================================
# CNN
#================================================================================================

class CNN(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes=4, input_channels=3, conv_filters=[16, 32, 64], kernel_size=3, dropout_rate=0,input_dim=256):
        super(CNN, self).__init__() 

        # Dynamic Convolutional layers
        layers = []
        in_channels = input_channels

        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Compute the size after convolutional layers for the fully connected layers
        self.flattened_size = conv_filters[-1] * input_dim//(2**len(conv_filters)) * input_dim//(2**len(conv_filters))  

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.flattened_size, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.fc_layers(x)
        return x

class CNNWithGAP(nn.Module):
    """Convolutional Neural Network with Global Average Pooling for image classification."""
    
    def __init__(self, num_classes=4, input_channels=3, conv_filters=[16, 32, 64], kernel_size=3, dropout_rate=0):
        super(CNNWithGAP, self).__init__() 

        # Convolutional layers without max pooling
        layers = []
        in_channels = input_channels

        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP to a 1x1 spatial dimension

        # Fully connected layer (output layer only)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters[-1], num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)  # Apply Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layer
        x = self.fc(x)
        return x

