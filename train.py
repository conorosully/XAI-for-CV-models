import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import glob
import os
import json
import argparse
import csv
from network import CNN, CNNWithGAP  # Importing the model from network.py
from datasets import ImageDataset  # Importing the dataset from datasets.py


def train_model(args):
    # Load data
    train_paths = glob.glob(args.data_path + "/train/*")
    val_paths = glob.glob(args.data_path + "/val/*")
    
    train_data = ImageDataset(train_paths, num_classes=args.classes)
    val_data = ImageDataset(val_paths, num_classes=args.classes)

    print(f"Training samples: {train_data.__len__()}") 
    print(f"Validation samples: {val_data.__len__()}")

    # Prepare data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=val_data.__len__())

    if args.network == 'cnn':
        model = CNN(num_classes=args.classes,input_dim=args.input_dim)
    elif args.network == 'cnn_gap':
        model = CNNWithGAP(num_classes=args.classes)
    
    # Initialize model
    if args.classes == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    min_loss = np.inf
    history = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            images, targets = next(iter(valid_loader))
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            valid_loss = criterion(outputs, targets)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {valid_loss.item():.4f}")
        
        # Save model if validation loss improves
        if valid_loss < min_loss:

            # Send model to cpu before saving
            model.to('cpu')
            
            # Save model state dict
            torch.save(model.state_dict(), args.model_path)

            # Send model back to device
            model.to(device)
            min_loss = valid_loss
            print(f"Model saved at {args.model_path}")

        # Append epoch and validation loss to history
        history.append([epoch + 1, valid_loss.item()])

    # Save training history to CSV
    csv_path = args.model_path.replace(".pth", ".csv")
    with open(csv_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Validation Loss"])
        writer.writerows(history)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a CNN on images.")
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes (1 for regression)')
    parser.add_argument('--input_dim', type=int, default=256, help='Input dimension of the image')
    parser.add_argument('--network', type=str, default='cnn', help='Model type (cnn or cnn_gap)')

    args = parser.parse_args()

    # Create model folder if it does not exist
    save_path = os.path.join("../models",args.save_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.model_path = os.path.join(save_path, "model.pth")

    # Save model config
    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "classes": args.classes,
        "input_dim": args.input_dim,
        "network": args.network
        }
    
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

    print(f"Training model with config: {config}")
    
    # Train the model
    train_model(args)


if __name__ == "__main__":
    main()
