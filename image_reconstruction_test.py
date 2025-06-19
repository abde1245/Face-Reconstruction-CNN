import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the Autoencoder512_BN Model Architecture
class Autoencoder512_BN(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder512_BN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, latent_dim, kernel_size=4, stride=2, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Load Trained Model
latent_dim = 512
model = Autoencoder512_BN(latent_dim).cuda()
checkpoint_path = r"D:\python programmes\CNN Models\Reconstruction CNN Model\autoencoder_ffhq_512.pth" # Path to your saved model checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set to evaluation mode

# Image Transformation
image_size = 512
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Input Image Path - CHANGE THIS TO YOUR TEST IMAGE
input_image_path = r"D:\python programmes\CNN Models\faces_dataset_small\00713.png"
input_image = Image.open(input_image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).cuda()

# Model Inference
with torch.no_grad():
    reconstructed_tensor, _ = model(input_tensor)

# Post-process Reconstructed Image
reconstructed_image_tensor = reconstructed_tensor.squeeze(0).cpu().permute(1, 2, 0)
reconstructed_image_tensor = (reconstructed_image_tensor * 0.5 + 0.5).clamp(0, 1)
reconstructed_image_np = reconstructed_image_tensor.numpy()

# Prepare Original Image for Difference Calculation and Visualization
original_image_tensor = transform(input_image).permute(1, 2, 0)
original_image_tensor = (original_image_tensor * 0.5 + 0.5).clamp(0, 1)
original_image_np = original_image_tensor.numpy()

# Calculate Absolute Difference Image
difference_image_np = np.abs(original_image_np - reconstructed_image_np)

# Visualization of Results
plt.figure(figsize=(15, 5))

# Plot Original Image
plt.subplot(1, 3, 1)
plt.imshow(input_image)
plt.title("Original")
plt.axis('off')

# Plot Reconstructed Image
plt.subplot(1, 3, 2)
plt.imshow(reconstructed_image_np)
plt.title("Reconstructed")
plt.axis('off')

# Plot Difference Image
plt.subplot(1, 3, 3)
plt.imshow(difference_image_np)
plt.title("Difference (Absolute)")
plt.axis('off')

plt.tight_layout()
plt.show()
