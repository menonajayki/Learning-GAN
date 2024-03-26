import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a basic transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset and select a subset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_indices = list(range(1000))  # Select 1000 images
train_subset = Subset(train_dataset, subset_indices)
dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)


# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Output range: -1 to 1 for images
        )

    def forward(self, z):
        return self.model(z)


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output: probability of being real
        )

    def forward(self, x):
        return self.model(x)


# Define hyperparameters
latent_dim = 100  # Size of the noise vector
image_dim = 3 * 32 * 32  # CIFAR-10 images are 3x32x32

# Create the Generator and Discriminator
generator = Generator(latent_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, real_images in enumerate(dataloader):
        # Train Discriminator
        real_images = real_images[0].view(-1, image_dim).to(device)
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)

        # Generate fake images
        z = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(z)

        # Train Discriminator with real images
        d_optimizer.zero_grad()
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_real.backward()

        # Train Discriminator with fake images
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

    # Save generated images
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        generated = generator(z).view(-1, 3, 32, 32).cpu()
        grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis("off")
        plt.show()

# Save Generator and Discriminator weights
torch.save(generator.state_dict(), 'generator_weights.pth')
torch.save(discriminator.state_dict(), 'discriminator_weights.pth')

# Load saved weights back into models
loaded_generator = Generator(latent_dim, image_dim).to(device)
loaded_discriminator = Discriminator(image_dim).to(device)

loaded_generator.load_state_dict(torch.load('generator_weights.pth'))
loaded_discriminator.load_state_dict(torch.load('discriminator_weights.pth'))

# Set models to evaluation mode
loaded_generator.eval()
loaded_discriminator.eval()
