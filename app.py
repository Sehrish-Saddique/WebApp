# STREAMLIT APP - app.py

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define Generator (Same as training)
class Generator(nn.Module):
    def __init__(self, z_dim=18, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Linear(z_dim + num_classes, 128)

        self.deconv = nn.Sequential(
            nn.Linear(128, 7 * 7 * 64),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 28x28
            nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = self.label_embedding(labels)
        x = torch.cat([z, one_hot], dim=1)
        x = self.fc(x)
        img = self.deconv(x)
        return img

# Load model
device = torch.device("cpu")
model = Generator()
try:
    # Use map_location to load the model trained on potentially different devices
    model.load_state_dict(torch.load("digit_generator.pth", map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    st.error("Model file 'digit_generator.pth' not found. Please make sure it's in the same directory as the app.")
    model = None # Set model to None if loading fails
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None # Set model to None if loading fails


st.title("🖊️ Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate (0–9)", list(range(10)))

# Check if the model was loaded successfully before proceeding
if model is not None:
    if st.button("Generate"):
        # Generate an initial batch of 5 random latent vectors
        z_base = torch.randn(5, 20)

        # Create labels for the batch - all the same digit
        labels = torch.full((5,), digit, dtype=torch.long)

        # Generate a small, unique perturbation for each of the 5 latent vectors
        # We use arange to create a sequence [0, 1, 2, 3, 4] and reshape it.
        # This provides a deterministic way to make each perturbation unique.
        # The scale factor (e.g., 0.1) controls the strength of the variation.
        # Adjust this factor to control the amount of diversity.
        perturbations = torch.arange(5).float().view(-1, 1) * 0.2 # shape (5, 1) * scale
        # Expand perturbations to match the shape of z_base (5, 20)
        # This adds [0*scale, 0*scale, ..., 0*scale] to the first z vector,
        # [1*scale, 1*scale, ..., 1*scale] to the second, and so on.
        # You could also add random noise here instead, but a deterministic offset guarantees difference.
        z_perturbed = z_base + perturbations # Broadcasting handles the addition

        with torch.no_grad():
            # Pass the batch of *perturbed* latent vectors and labels to the generator
            # The differences in z_perturbed should force the generator to output
            # slightly different images, even if z_base alone wasn't enough.
            gen_imgs = model(z_perturbed, labels).squeeze().numpy()

        # Display the generated images
        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            # Ensure image data is correctly formatted for imshow (height, width)
            axs[i].imshow(gen_imgs[i], cmap="gray")
            axs[i].axis("off")
        st.pyplot(fig)
else:
    st.warning("Model could not be loaded. Please check the logs for details.")
