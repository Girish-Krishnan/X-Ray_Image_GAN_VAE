import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "./chest_xray"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CLASSES = ["NORMAL", "PNEUMONIA"]

# Define transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG19 expects 224x224 input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

# Fine-tune VGG19 with custom layers
def create_model(num_classes):
    # Load pretrained VGG19 model
    model = models.vgg19(pretrained=True)
    
    # Freeze all VGG19 layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with custom layers
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model

# Train function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
        
        train_accuracy = correct / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        # Record metrics
        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

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
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

# Visualize predictions
def visualize_predictions(model, test_loader, device):
    model.eval()
    correct_images, correct_labels, correct_preds = [], [], []
    wrong_images, wrong_labels, wrong_preds = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for img, label, pred in zip(images, labels, preds):
                if label == pred:
                    correct_images.append(img)
                    correct_labels.append(label)
                    correct_preds.append(pred)
                else:
                    wrong_images.append(img)
                    wrong_labels.append(label)
                    wrong_preds.append(pred)

    def unnormalize(img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        return img * std[:, None, None] + mean[:, None, None]

    # Display correct predictions
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 6, i + 1)
        img = unnormalize(correct_images[i].cpu()).permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Label: {CLASSES[correct_labels[i]]}\nPred: {CLASSES[correct_preds[i]]}")
        plt.axis("off")

    # Display wrong predictions
    for i in range(6):
        plt.subplot(2, 6, i + 7)
        img = unnormalize(wrong_images[i].cpu()).permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Label: {CLASSES[wrong_labels[i]]}\nPred: {CLASSES[wrong_preds[i]]}")
        plt.axis("off")

    plt.suptitle(f"Correct and Wrong Predictions")
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
    
    print("Training complete. Plotting curves...")
    plot_training_curves(history)

    print("Evaluating on test data...")
    visualize_predictions(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()
