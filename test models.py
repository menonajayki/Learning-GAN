import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Define your Generator class
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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to your saved Generator model
generator_path = 'generator_weights.pth'

# Initialize the Generator with the same dimensions as your trained model
latent_dim = 100  # Assuming this is the latent dimension
image_dim = 3 * 32 * 32  # Assuming this is the output image dimension

# Initialize the Generator model
generator = Generator(latent_dim, image_dim).to(device)

# Load the saved Generator model
generator.load_state_dict(torch.load(generator_path, map_location=device))

# Set the Generator to evaluation mode
generator.eval()

# Generate images with the loaded Generator
with torch.no_grad():
    num_images = 16  # Number of images to generate
    z = torch.randn(num_images, latent_dim).to(device)
    generated_images = generator(z).cpu()

# Plot the generated images
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(4, 4, i+1)
    img = generated_images[i].numpy().reshape(3, 32, 32).transpose((1, 2, 0))  # Convert to numpy and transpose
    img = 0.5 * img + 0.5  # Unnormalize
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.show()
