import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image

# --- 1. Dataset Preparation ---
# Configuration for image size and transformations
image_size = 512
transform = transforms.Compose([
    transforms.Resize(image_size), # Resize images to ensure consistent size
    transforms.CenterCrop(image_size), # Center crop to get square images of desired size
    transforms.RandomHorizontalFlip(), # Apply random horizontal flip for data augmentation
    transforms.ToTensor(), # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize pixel values to the range [-1, 1]
])

# Define a custom dataset class for FFHQ dataset
class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # Directory containing the images
        self.transform = transform # Transformations to be applied to images
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))] # List of image file names

    def __len__(self):
        return len(self.image_files) # Return the total number of images in the dataset

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx]) # Construct the full image file path
        image = Image.open(img_name).convert('RGB') # Open image using PIL and convert to RGB format

        if self.transform:
            image = self.transform(image) # Apply transformations if specified
        return image # Return the transformed image

# Specify the path to the FFHQ dataset
dataset_path = r"D:\python programmes\CNN Models\faces_dataset_small" # <--- **CHANGE THIS TO YOUR FFHQ DATASET PATH**
if not os.path.isdir(dataset_path):
    print(f"Error: FFHQ dataset not found at '{dataset_path}'. Please download and place it there.")
    exit()

# Create FFHQ dataset and data loader
ffhq_dataset = FFHQDataset(root_dir=dataset_path, transform=transform)
batch_size = 8 # Batch size for training, adjust based on GPU memory
dataloader = DataLoader(ffhq_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # DataLoader for efficient batching and shuffling, num_workers=0 for Windows


# --- 2. Autoencoder Model Definition ---
# Define the Autoencoder512_BN model with Batch Normalization
class Autoencoder512_BN(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder512_BN, self).__init__()

        # Encoder Definition
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # Input: 512x512x3, Output: 256x256x64
            nn.BatchNorm2d(64), # Batch Normalization for stable training
            nn.ReLU(), # ReLU activation function
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Input: 256x256x64, Output: 128x128x128
            nn.BatchNorm2d(128), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Input: 128x128x128, Output: 64x64x256
            nn.BatchNorm2d(256), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Input: 64x64x256, Output: 32x32x512
            nn.BatchNorm2d(512), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # Input: 32x32x512, Output: 16x16x1024
            nn.BatchNorm2d(1024), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.Conv2d(1024, latent_dim, kernel_size=4, stride=2, padding=1) # Input: 16x16x1024, Output: 8x8xlatent_dim (Latent Space)
        )

        # Decoder Definition
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=2, padding=1), # Input: 8x8xlatent_dim, Output: 16x16x1024
            nn.BatchNorm2d(1024), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # Input: 16x16x1024, Output: 32x32x512
            nn.BatchNorm2d(512), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Input: 32x32x512, Output: 64x64x256
            nn.BatchNorm2d(256), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Input: 64x64x256, Output: 128x128x128
            nn.BatchNorm2d(128), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Input: 128x128x128, Output: 256x256x64
            nn.BatchNorm2d(64), # Batch Normalization
            nn.ReLU(), # ReLU activation
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Input: 256x256x64, Output: 512x512x3
            nn.Tanh() # Tanh activation to output pixel values in range [-1, 1]
        )

    def forward(self, x):
        latent = self.encoder(x) # Encode the input image to latent space
        reconstructed = self.decoder(latent) # Decode the latent space to reconstruct the image
        return reconstructed, latent # Return reconstructed image and latent representation


# --- 3. Loss Function and Optimizer Definition ---
# Define Perceptual Loss using pre-trained VGG16 features
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True).features[:16].eval() # Load pre-trained VGG16 features (first few layers)
        for param in vgg16.parameters():
            param.requires_grad = False # Freeze VGG weights to use as feature extractor
        self.vgg = vgg16.cuda() # Move VGG to GPU

    def forward(self, reconstructed_images, original_images):
        features_original = self.vgg(original_images) # Extract features from original images using VGG16
        features_reconstructed = self.vgg(reconstructed_images) # Extract features from reconstructed images using VGG16
        return nn.MSELoss()(features_reconstructed, features_original) # Calculate MSE loss in the feature space


latent_dim = 512 # Define the dimensionality of the latent space
model = Autoencoder512_BN(latent_dim).cuda() # Instantiate the Autoencoder model and move to GPU
mse_criterion = nn.MSELoss() # Define Mean Squared Error loss for pixel-wise comparison
perceptual_criterion = PerceptualLoss() # Instantiate Perceptual Loss
optimizer = optim.Adam(model.parameters(), lr=0.001) # Define Adam optimizer for training


# --- 4. Training Loop ---
num_epochs = 50 # Number of training epochs
for epoch in range(num_epochs):
    for batch_idx, images in enumerate(dataloader):
        images = images.cuda(non_blocking=True) # Move batch of images to GPU

        # Forward pass
        reconstructed_images, _ = model(images) # Get reconstructed images from the model

        # Calculate Total Loss - Combined Perceptual and MSE Loss
        mse_loss = mse_criterion(reconstructed_images, images) # Calculate MSE loss
        perc_loss = perceptual_criterion(reconstructed_images, images) # Calculate Perceptual loss
        loss = mse_loss + 0.02 * perc_loss # Combine MSE and Perceptual loss with a weight for perceptual loss

        # Backward pass and optimization
        optimizer.zero_grad() # Clear gradients from the previous iteration
        loss.backward() # Backpropagate the loss
        optimizer.step() # Update model weights

        if (batch_idx+1) % 100 == 0: # Print training progress every 100 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Perceptual Loss: {perc_loss.item():.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Perceptual Loss: {perc_loss.item():.4f}')
    torch.save(model.state_dict(), 'autoencoder_ffhq_512_v1.3.pth') # Save model weights after each epoch

    # --- 5. Visualization of Reconstruction (at the end of each epoch) ---
    with torch.no_grad(): # Disable gradient calculation for visualization
        example_images = next(iter(dataloader)).cuda() # Get a batch of example images
        reconstructed_example, _ = model(example_images) # Reconstruct example images
        n_samples = 4 # Number of samples to visualize

        plt.figure(figsize=(16, 8)) # Set figure size for visualization
        for i in range(n_samples):
            # Original images subplot
            plt.subplot(2, n_samples, i + 1)
            original_img = example_images[i].cpu().permute(1, 2, 0) # Move to CPU and change tensor dimensions for plotting
            original_img = (original_img * 0.5 + 0.5).clamp(0, 1) # Denormalize and clamp pixel values to [0, 1]
            plt.imshow(original_img) # Display original image
            plt.title("Original") # Set title
            plt.axis('off') # Turn off axis

            # Reconstructed images subplot
            plt.subplot(2, n_samples, i + n_samples + 1)
            recon_img = reconstructed_example[i].cpu().permute(1, 2, 0) # Move to CPU and change tensor dimensions for plotting
            recon_img = (recon_img * 0.5 + 0.5).clamp(0, 1) # Denormalize and clamp pixel values to [0, 1]
            plt.imshow(recon_img) # Display reconstructed image
            plt.title("Reconstructed") # Set title
            plt.axis('off') # Turn off axis
        plt.tight_layout() # Adjust subplot parameters for a tight layout
        plt.show() # Show the plot

print("Training finished!")
torch.save(model.state_dict(), 'autoencoder_ffhq_512_v1.3.pth') # Save model weights after training completion
print("Model saved to autoencoder_ffhq_512_v1.3.pth") # Print confirmation message for model saving
