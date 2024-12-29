import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Constants
DATA_DIR = "./chest_xray"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.00001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CLASSES = ["NORMAL", "PNEUMONIA"]

# Define transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform

# Load datasets
def load_datasets(data_dir):
    train_transform, test_transform = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
    return train_dataset, val_dataset, test_dataset

# Data loaders
def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Fine-tune Inception v3
def create_model(num_classes):
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.aux_logits = True
    return model

# Train function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, aux_outputs = model(images) if model.training else (model(images), None)
            loss = criterion(outputs, labels)
            if aux_outputs is not None:
                aux_loss = criterion(aux_outputs, labels)
                loss += 0.4 * aux_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

        train_accuracy = correct / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {train_loss/len(train_loader.dataset):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return history

# Evaluate function
def evaluate_model(model, loader, criterion, device):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
    accuracy = correct / len(loader.dataset)
    return loss / len(loader.dataset), accuracy

# Plot training curves
def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Test function with visualization
def test_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    correct_images, wrong_images = [], []
    correct_labels, wrong_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for img, pred, label in zip(images, preds, labels):
                if pred == label:
                    correct_images.append(img.cpu())
                    correct_labels.append(label.cpu())
                else:
                    wrong_images.append(img.cpu())
                    wrong_labels.append((pred.cpu(), label.cpu()))

                y_true.append(label.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Visualize correct and wrong predictions
    visualize_predictions(correct_images, correct_labels, wrong_images, wrong_labels, len(y_true), sum(np.array(y_true) == np.array(y_pred)) / len(y_true))

# Visualization function
def visualize_predictions(correct_images, correct_labels, wrong_images, wrong_labels, total_samples, accuracy):
    plt.figure(figsize=(16, 8))

    # Correct predictions
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(correct_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.title(f"Correct: {CLASSES[correct_labels[i]]}")
        plt.axis("off")

    # Wrong predictions
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(wrong_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        pred, label = wrong_labels[i]
        plt.title(f"Pred: {CLASSES[pred]} \n True: {CLASSES[label]}")
        plt.axis("off")

    plt.suptitle(f"Test Accuracy: {accuracy:.4f}")
    plt.tight_layout()
    plt.show()

# Main function
def main():
    train_dataset, val_dataset, test_dataset = load_datasets(DATA_DIR)
    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

    model = create_model(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)

    print("Plotting training curves...")
    plot_training_curves(history)

    print("Evaluating on test data...")
    test_model(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()
