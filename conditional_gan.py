import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "./chest_xray"
BATCH_SIZE = 128
LATENT_DIM = 100  # Dimensionality of the latent space
NUM_CLASSES = 2  # NORMAL and PNEUMONIA
IMAGE_SIZE = 64  # Resize images to this size
LEARNING_RATE_D = 0.0001
LEARNING_RATE_G = 0.0002
BETA1 = 0.5  # Beta1 hyperparameter for Adam optimizer
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define transformations
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform

# Load dataset
def load_dataset(data_dir):
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    return train_dataset

def get_data_loader(train_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Conditional Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, kernel_size=4, stride=2, padding=1), # (num_channels, 64, 64)
            nn.Tanh(),  # Output range [-1, 1]
        )

    def forward(self, z, labels):
        label_embeddings = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, num_classes, 1, 1)
        z = torch.cat([z, label_embeddings], dim=1)  # Concatenate noise and label embeddings
        return self.model(z)

# Conditional Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(num_channels + num_classes, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output: (1, 5, 5)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce spatial dimensions to (1, 1)
        self.sigmoid = nn.Sigmoid()  # Output range [0, 1]

    def forward(self, img, labels):
        label_embeddings = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, num_classes, 1, 1)
        label_embeddings = label_embeddings.expand(-1, -1, img.size(2), img.size(3))  # Match image dimensions
        x = torch.cat([img, label_embeddings], dim=1)  # Concatenate image and label embeddings
        x = self.model(x)  # Pass through convolutional layers
        x = self.global_avg_pool(x)  # Reduce to (batch_size, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 1)
        x = x.squeeze(1)  # Return (batch_size)
        x = self.sigmoid(x)  # Output range [0, 1]
        return x

def train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, num_epochs, latent_dim):
    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for real_images, labels in tqdm(train_loader):
            batch_size = real_images.size(0)
            real_images, labels = real_images.to(device), labels.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.full((batch_size,), 0.9, device=device)  # Smooth real labels
            fake_labels = torch.zeros(batch_size, device=device)

            outputs = discriminator(real_images, labels)
            loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(z, labels)
            outputs = discriminator(fake_images.detach(), labels)
            loss_fake = criterion(outputs, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images, labels)
            loss_g = -torch.mean(torch.log(outputs + 1e-8))  # Modified generator loss

            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

        # Save generated samples
        save_generated_images(generator, latent_dim, device, epoch, label=0, label_name="NORMAL")
        save_generated_images(generator, latent_dim, device, epoch, label=1, label_name="PNEUMONIA")

    # Save final models
    torch.save(generator.state_dict(), "model_files/conditional_generator.pth")
    torch.save(discriminator.state_dict(), "model_files/conditional_discriminator.pth")

# Save generated images
def save_generated_images(generator, latent_dim, device, epoch, label, label_name, num_images=64):
    z = torch.randn(num_images, latent_dim, 1, 1, device=device)
    labels = torch.full((num_images,), label, device=device, dtype=torch.long)
    fake_images = generator(z, labels)
    fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]

    grid = utils.make_grid(fake_images, nrow=8)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(f"Generated {label_name} Images - Epoch {epoch+1}")
    plt.savefig(f"./conditional_generated_images/{label_name}_epoch_{epoch+1}.png")
    plt.close()

def main():
    train_dataset = load_dataset(DATA_DIR)
    train_loader = get_data_loader(train_dataset, BATCH_SIZE)

    num_channels = 3  # RGB images
    generator = Generator(latent_dim=LATENT_DIM, num_channels=num_channels, num_classes=NUM_CLASSES)
    discriminator = Discriminator(num_channels=num_channels, num_classes=NUM_CLASSES)

    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))

    criterion = nn.BCELoss()

    print("Training Conditional GAN...")
    train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, DEVICE, NUM_EPOCHS, LATENT_DIM)

if __name__ == "__main__":
    main()
