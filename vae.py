import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import inception_v3
from torch.nn.functional import softmax

# Constants
DATA_DIR = "./chest_xray"
BATCH_SIZE = 32
LATENT_DIM = 64  # Dimensionality of the latent space
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define transformations
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images for the autoencoder
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
    ])
    return transform

# Load dataset
def load_dataset(data_dir):
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output range [-1, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 128, 16, 16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld, recon_loss, kld

def calculate_inception_score(generated_images, model, num_splits=10):
    """
    Calculate the Inception Score (IS) for generated images.
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        preds = softmax(model(generated_images), dim=-1).detach().cpu().numpy()
    
    # Split into smaller batches
    split_scores = []
    for i in range(num_splits):
        part = preds[i * len(preds) // num_splits : (i + 1) * len(preds) // num_splits, :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        split_scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))
    
    return np.mean(split_scores), np.std(split_scores)

def train_vae(model, train_loader, optimizer, device, num_epochs):
    model.to(device)
    
    # Load Inception v3 model for metrics
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = nn.Identity()  # Remove final layer for feature extraction
    
    history = {"loss": [], "inception_score": []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_recon_loss, total_kld = 0, 0, 0
        
        for images, _ in tqdm(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = model(images)
            loss, recon_loss, kld = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld += kld.item()
        
        # Evaluate IS after every epoch
        model.eval()
        real_images = next(iter(train_loader))[0].to(device)[:32]
        with torch.no_grad():
            z = torch.randn(32, LATENT_DIM).to(device)
            generated_images = model.decode(z)

        is_mean, is_std = calculate_inception_score(generated_images, inception)

        history["loss"].append(total_loss / len(train_loader))
        history["inception_score"].append(is_mean)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Recon Loss: {total_recon_loss:.4f}, "
              f"KLD: {total_kld:.4f}, IS: {is_mean:.4f} ± {is_std:.4f}")
    
    return history

# Visualize reconstruction
def visualize_reconstruction(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            recon_images, _, _ = model(images)
            break  # Take one batch

    # Denormalize images for visualization
    images = images.cpu() * 0.5 + 0.5
    recon_images = recon_images.cpu() * 0.5 + 0.5

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_images[i].permute(1, 2, 0))
        axes[1, i].axis("off")
    plt.show()

def calculate_class_means(model, loader, device, class_idx):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            mu, _ = model.encode(images)
            for i in range(len(labels)):
                if labels[i] == class_idx:
                    latent_vectors.append(mu[i].cpu().numpy())
    latent_vectors = np.array(latent_vectors)
    return np.mean(latent_vectors, axis=0)

def generate_class_images(model, class_mean, latent_dim, num_images, device):
    model.eval()
    with torch.no_grad():
        # Sample latent vectors around the class mean
        sampled_latents = class_mean + np.random.randn(num_images, latent_dim) * 0.1
        sampled_latents = torch.tensor(sampled_latents).float().to(device)
        generated_images = model.decode(sampled_latents)
        generated_images = generated_images.cpu() * 0.5 + 0.5  # Denormalize to [0, 1]

    # Visualize the generated images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(generated_images[i].permute(1, 2, 0))
        axes[i].axis("off")
    plt.show()

def plot_metrics(history):
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["inception_score"])
    plt.title("Inception Score Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()

    plt.show()

# Main function
def main():
    train_dataset, val_dataset, test_dataset = load_dataset(DATA_DIR)
    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)
    
    model = VAE(latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training Variational Autoencoder...")
    history = train_vae(model, train_loader, optimizer, DEVICE, NUM_EPOCHS)
    plot_metrics(history)
    
    print("Visualizing Reconstructions...")
    visualize_reconstruction(model, test_loader, DEVICE)

    print("Calculating Class Means...")
    normal_mean = calculate_class_means(model, train_loader, DEVICE, class_idx=0)  # Class 0: NORMAL
    pneumonia_mean = calculate_class_means(model, train_loader, DEVICE, class_idx=1)  # Class 1: PNEUMONIA

    print("Generating NORMAL Images...")
    generate_class_images(model, normal_mean, LATENT_DIM, num_images=10, device=DEVICE)

    print("Generating PNEUMONIA Images...")
    generate_class_images(model, pneumonia_mean, LATENT_DIM, num_images=10, device=DEVICE)


if __name__ == "__main__":
    main()
