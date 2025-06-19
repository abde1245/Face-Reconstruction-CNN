### **Model Description**

The implemented model is a **Convolutional Autoencoder**, designed to reconstruct face images from the FFHQ dataset. The autoencoder consists of:

-   **Encoder**: Compresses the input image into a lower-dimensional latent space.
-   **Decoder**: Reconstructs the image from the latent representation.

### **Training Details**

**Dataset**: A subset of the **FFHQ (Flickr-Faces-HQ)** dataset with high-resolution face images.

**Model Architecture**:
*   7-layer Convolutional Encoder and a mirrored 7-layer Deconvolutional Decoder.
*   Batch Normalization for stabilization.
*   Perceptual Loss (VGG16-based) + MSE Loss to enhance reconstruction quality.

**Key Hyperparameters**:
*   **Batch Size**: `4`
*   **Epochs**: `1000`
*   **Learning Rate**: `0.005`
*   **Latent Dimension**: `512`
*   **Image Size**: `512×512`
*   **Optimizer**: Adam
*   **Loss Function**: MSE + Perceptual Loss
*   **Optimization**: Mixed precision training (`torch.cuda.amp`) was used to speed up training and reduce memory usage.

**Training Loss**: The combined loss (MSE + Perceptual Loss) consistently decreased over 1000 epochs, showing effective learning and improved reconstruction quality.

### **Trained Model Download**

Due to its large size, the trained model file (`autoencoder_ffhq_512.pth`) is available for download via Google Drive:

🔗 **[Download Trained Model](https://drive.google.com/file/d/1vQSVuSdOxly8Pep-IQ0HInoFVS6z6e_y/view?usp=sharing)**

To test the model, please run `image_reconstruction_test.py`, ensuring that the downloaded weights file is placed in the correct directory or updating the file path accordingly.
