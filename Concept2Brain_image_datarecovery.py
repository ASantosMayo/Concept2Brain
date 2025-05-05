# -*- coding: utf-8 -*-
"""
Concept2Brain text test

Code for assesing model data recovery on EEG generation from same pictures used in training phase.
"""
# =============================================================================


import clip
import torch
import numpy as np
import time
import torch.nn as nn
import pandas as pd
from PIL import Image
import scipy.io as sio
import os

# =============================================================================
# Model Definitions (Copied from training script)
# =============================================================================

# Define the Conditional VAE structure for EEG
# Note: Comments mention GPU allocation (0, 1, 2) but the code explicitly sets devices to "cpu".
# This might be legacy comments from the training phase or intended design. The script runs on CPU as written.
class EEG_CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for EEG data.
    Original design potentially targeted multi-GPU (comments mention GPU 0, 1, 2),
    but devices are set to CPU in this inference script instance.
    """
    def __init__(self, input_channels, condition_dim, latent_dim):
        super(EEG_CVAE, self).__init__()
        self.condition_dim = condition_dim

        # --- Encoder ---
        # Originally intended for GPU 0, set to CPU here.
        self.encoder_device = torch.device("cpu")
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1).to(self.encoder_device) # Output: (N, 32, 65, 375)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1).to(self.encoder_device) # Output: (N, 64, 33, 188)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1).to(self.encoder_device) # Output: (N, 128, 17, 94)
        self.flattened_size = 128 * 17 * 94 # 204224

        # --- Bottleneck Layers ---
        # Originally intended for GPU 1, set to CPU here.
        self.bottleneck_device = torch.device("cpu")
        self.fc_mu = nn.Linear(self.flattened_size + condition_dim, latent_dim).to(self.bottleneck_device)
        self.fc_logvar = nn.Linear(self.flattened_size + condition_dim, latent_dim).to(self.bottleneck_device)

        # --- Decoder ---
        # Originally intended for GPU 2, set to CPU here.
        self.decoder_device = torch.device("cpu")
        self.fc_decode = nn.Linear(latent_dim + condition_dim, self.flattened_size).to(self.decoder_device)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1).to(self.decoder_device)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1).to(self.decoder_device)
        self.deconv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1).to(self.decoder_device)

    def encode(self, x, c):
        """Encodes input x conditioned by c into mu and logvar."""
        x = x.to(self.encoder_device)
        c = c.to(self.encoder_device) # Condition also needed for encoder if architecture uses it early

        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2)) # Use relu here

        h3_flat = h3.view(h3.size(0), -1) # Flatten
        h3_flat = h3_flat.to(self.bottleneck_device) # Move to bottleneck device
        c_bottleneck = c.to(self.bottleneck_device) # Move condition to bottleneck device
        # Concatenate flattened data and condition on bottleneck device
        combined_bottleneck = torch.cat([h3_flat, c_bottleneck], dim=1)

        # Ensure fc layers are on the correct device before use (redundant if done in init)
        # self.fc_mu.to(self.bottleneck_device)
        # self.fc_logvar.to(self.bottleneck_device)
        mu = self.fc_mu(combined_bottleneck)
        logvar = self.fc_logvar(combined_bottleneck)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Performs the reparameterization trick."""
        # Ensure tensors are on the correct device
        mu = mu.to(self.bottleneck_device)
        logvar = logvar.to(self.bottleneck_device)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(std.device) # Create epsilon on the same device
        # Clamping the standard deviation to avoid explosion/implosion - Check range appropriateness
        # Original had [-5, 2.0], maybe meant for std itself? Clamping std usually involves positive min.
        # std = torch.clamp(std, min=1e-6, max=10.0) # Example of typical std clamping
        # Keeping original clamp logic for now:
        std = torch.clamp(std, min=-5, max=2.0) # Warning: Clamping std to negative values is unusual. Check original intent.
        return mu + eps * std

    def decode(self, z, c):
        """Decodes latent vector z conditioned by c into reconstruction."""
        # Move z and condition to the bottleneck device for concatenation & fc_decode
        z = z.to(self.bottleneck_device)
        c = c.to(self.bottleneck_device)
        combined_input = torch.cat([z, c], dim=1)

        # Ensure fc_decode is on the correct device
        # self.fc_decode.to(self.bottleneck_device) # Redundant if done in init
        h6_bottleneck = torch.relu(self.fc_decode(combined_input))

        # Move to decoder device and reshape
        h6 = h6_bottleneck.to(self.decoder_device)
        h6 = h6.view(h6.size(0), 128, 17, 94) # Reshape to (N, C, H, W) for deconv

        # Ensure deconv layers are on the correct device (redundant if done in init)
        # self.deconv1.to(self.decoder_device)
        # self.deconv2.to(self.decoder_device)
        # self.deconv3.to(self.decoder_device)
        h9 = torch.relu(self.deconv1(h6))
        h10 = torch.relu(self.deconv2(h9))
        x_recon_raw = self.deconv3(h10) # Final deconv

        # Crop to original EEG dimensions (assuming 129 electrodes, 750 time points)
        x_recon = x_recon_raw[:, :, :129, :750]
        return x_recon

    def forward(self, x, c):
        """Complete forward pass."""
        # Move initial inputs
        x = x.to(self.encoder_device)
        c = c.to(self.encoder_device) # Condition needed on encoder device

        # Encode
        mu, logvar = self.encode(x, c)

        # Reparameterize (on bottleneck device)
        z = self.reparameterize(mu, logvar)

        # Decode (requires z and c on decoder device)
        c_bottleneck_dev = c.to(self.bottleneck_device)
        x_recon = self.decode(z, c_bottleneck_dev)

        # Return reconstruction and latent parameters (on bottleneck device)
        return x_recon, mu.to(self.bottleneck_device), logvar.to(self.bottleneck_device)


# Definition of the Latent Network (CLIP -> CVAE Latent Space)
# Note: Comments mention BatchNorm/Dropout layers which are commented out in the forward pass.
class LATENT_XD(nn.Module):
    """
    Neural Network mapping (CLIP features + Condition) to CVAE latent space (z).
    Includes commented-out options for BatchNorm and Dropout.
    Operates on CPU in this script instance.
    """
    def __init__(self, input_features, condition_dim, hidden_units_1, hidden_units_2, output_features):
        super(LATENT_XD, self).__init__()
        self.device_latent = torch.device("cpu")

        # Layer definitions
        self.fc1 = nn.Linear(input_features + condition_dim, hidden_units_1).to(self.device_latent)

        self.fc2 = nn.Linear(hidden_units_1, hidden_units_2).to(self.device_latent)

        self.fc3 = nn.Linear(hidden_units_2, output_features).to(self.device_latent)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x, c):
        """Forward pass."""
        x = x.to(self.device_latent)
        c = c.to(self.device_latent)
        xc = torch.cat([x, c], dim=1) # Concatenate features and condition

        # Ensure layers are on the correct device (redundant if done in init)
        # self.fc1.to(self.device_latent)
        # self.fc2.to(self.device_latent)
        # self.fc3.to(self.device_latent)

        # Apply layers according to the active (uncommented) path in the original code
        x = self.elu(self.fc1(xc)) # Apply first layer and activation

        x = self.elu(self.fc2(x)) # Apply second layer and activation

        x = self.fc3(x) # Final output layer (linear)
        return x

# =============================================================================
# Configuration and Model Loading
# =============================================================================
print("--- Setting up configuration and loading models ---")

# --- Device Setup ---
# Using CPU for this inference script based on model definitions
device = torch.device("cpu")
print(f"Using device: {device}")

# --- Model Parameters ---
# EEG_CVAE Parameters
input_channels = 1      # Number of input channels (EEG)
latent_dim = 256        # Dimension of the latent space
condition_dim = 88      # Dimension of the condition vector (number of subjects)

# LATENT_XD Parameters (ensure these match the trained model)
input_features = 512    # CLIP feature dimension
# condition_dim is the same (88)
hidden_units_1 = 256    # Hidden layer sizes from original LATENT_XD definition
hidden_units_2 = 512
output_features = latent_dim # Output matches CVAE latent dim

# --- Instantiate Models ---
print("Instantiating models...")
model_EEG_CVAE = EEG_CVAE(input_channels, condition_dim, latent_dim).to(device)
model_LATENT_XD = LATENT_XD(input_features, condition_dim, hidden_units_1, hidden_units_2, output_features).to(device)
model_CLIP, preprocess = clip.load("ViT-B/32", device=device)
print("Models instantiated.")

# --- Load Pre-trained Weights ---
# Define paths to the saved model weights
cvae_weights_path = "MODEL_CCVAE_AnnelingBeta00001_01-100sigmoid_lr1e-6_1300epc_dict.pth"
latent_xd_weights_path = "MODEL_Latent_10epc_512batch_dict.pth"

print("Loading pre-trained weights...")
# Load CVAE weights
if os.path.exists(cvae_weights_path):
    state_dict_cvae = torch.load(cvae_weights_path, map_location=device)
    model_EEG_CVAE.load_state_dict(state_dict_cvae)
    model_EEG_CVAE.eval() # Set CVAE to evaluation mode
    print("EEG_CVAE weights loaded successfully.")
else:
    print(f"Error: EEG_CVAE weights file not found at {cvae_weights_path}")
    # exit() # Optional: stop script if weights are essential

# Load LATENT_XD weights
if os.path.exists(latent_xd_weights_path):
    state_dict_latent = torch.load(latent_xd_weights_path, map_location=device)
    model_LATENT_XD.load_state_dict(state_dict_latent)
    model_LATENT_XD.eval() # Set Latent model to evaluation mode
    print("LATENT_XD weights loaded successfully.")
else:
    print(f"Error: LATENT_XD weights file not found at {latent_xd_weights_path}")
    # exit() # Optional: stop script if weights are essential

# Set CLIP model to evaluation mode as well
model_CLIP.eval()

# --- Subject IDs ---
# Assuming SUBJECTS_IDS is needed and defined elsewhere or loaded
# For standalone execution, let's define it based on condition_dim
# This should ideally be loaded consistently with the training data.
# Example: SUBJECTS_IDS = np.arange(1, condition_dim + 1)
# Placeholder if SUBJECTS_IDS is not loaded:
if 'SUBJECTS_IDS' not in locals():
    print("Warning: SUBJECTS_IDS not found. Creating placeholder based on condition_dim.")
    SUBJECTS_IDS = np.arange(1, condition_dim + 1)


# =============================================================================
# Single Text Input Inference Example
# =============================================================================
print("\n--- Running inference for a single text input ---")

text = 'A picture of a cat' # Example text input
subject_id = 34 # Example subject ID (1-based index)
print(f"Input Text: '{text}'")
print(f"Target Subject ID: {subject_id}")

# --- Prepare Inputs ---
# Tokenize text for CLIP
text_input = clip.tokenize([text]).to(device)

# Create one-hot condition vector for the subject
# Ensure subject_id is within the valid range [1, condition_dim]
if 1 <= subject_id <= condition_dim:
    conditionTest = np.zeros((1, condition_dim))
    conditionTest[0, subject_id - 1] = 1 # 0-based index
    conditionTest_tensor = torch.tensor(conditionTest, dtype=torch.float32).to(device)
else:
    print(f"Error: Subject ID {subject_id} is out of valid range [1, {condition_dim}]. Using subject 1.")
    subject_id = 1
    conditionTest = np.zeros((1, condition_dim))
    conditionTest[0, 0] = 1
    conditionTest_tensor = torch.tensor(conditionTest, dtype=torch.float32).to(device)


# --- Run Inference Pipeline ---
start_time_total = time.time()
with torch.no_grad(): # Disable gradient calculations for inference
    # 1. Get text features from CLIP
    start_time_clip = time.time()
    text_features = model_CLIP.encode_text(text_input)
    clip_time = time.time()
    print(f"  CLIP model time: {clip_time - start_time_clip:.4f} seconds")

    # 2. Map text features to CVAE latent space using LATENT_XD
    start_time_latent = time.time()
    # Ensure feature tensor is float
    text_vae_latent_z = model_LATENT_XD(text_features.float(), conditionTest_tensor)
    latent_xd_time = time.time()
    print(f"  LATENT_XD model time: {latent_xd_time - start_time_latent:.4f} seconds")

    # 3. Decode latent vector z to generate EEG using CVAE decoder
    start_time_cvae = time.time()
    # Pass the predicted z and the condition tensor to the decoder
    text_EEG_recon = model_EEG_CVAE.decode(text_vae_latent_z, conditionTest_tensor)
    eeg_cvae_time = time.time()
    print(f"  EEG_CVAE decode time: {eeg_cvae_time - start_time_cvae:.4f} seconds")

# --- Process Output ---
# Move generated EEG to CPU and remove batch/channel dimensions
text_EEG_output = text_EEG_recon.detach().cpu().numpy().squeeze() # Shape: (E, T) e.g., (129, 750)

end_time_total = time.time()
print(f"Total time for single text inference: {end_time_total - start_time_total:.4f} seconds")
print(f"Shape of generated EEG for text: {text_EEG_output.shape}")


# =============================================================================
# Data Recovery Test (Generating EEG from Images)
# =============================================================================
print("\n--- Running data recovery test (Image -> EEG Generation) ---")

# --- Setup Paths and Load Metadata ---
# Base path (adjust as needed)
path = './'
# Path to image metadata file (ensure it exists)
excel_file_path = os.path.join(path, 'PictureLabels.xlsx')
# Path to images directory (ensure it exists)
image_dir_path = os.path.join(path, 'Training_pictures')
# Output path for results
output_dir_path = os.path.join(path, 'LatentCLIP')
output_filename = os.path.join(output_dir_path, 'RESULTS_DataRecoveryPic.mat')

# Create output directory if it doesn't exist
os.makedirs(output_dir_path, exist_ok=True)

print(f"Loading image metadata from: {excel_file_path}")
if os.path.exists(excel_file_path):
    excelPic = pd.read_excel(excel_file_path)
    num_images = len(excelPic) # Should be 360 based on original code
    print(f"Found metadata for {num_images} images.")
else:
    print(f"Error: Image metadata file not found at {excel_file_path}")
    exit()

# Extract affective labels (P=1, N=2, U=3)
labelAffect = np.zeros((num_images, 1))
affect_map = {'P': 1, 'N': 2, 'U': 3}
# Assuming the affective label is in the 5th column (index 4)
for pic_idx in range(num_images):
    affect_label = excelPic.iloc[pic_idx, 4]
    labelAffect[pic_idx] = affect_map.get(affect_label, 0) # Use 0 for unknown labels

print("Affective labels extracted (1:Pleasant, 2:Neutral, 3:Unpleasant).")

# --- Initialize Result Arrays ---
num_electrodes = 129 # Based on CVAE output cropping
num_timepoints = 750
num_subjects = condition_dim # 88

RESULTS_EEG_P = np.zeros((num_electrodes, num_timepoints, num_subjects))
RESULTS_EEG_N = np.zeros((num_electrodes, num_timepoints, num_subjects))
RESULTS_EEG_U = np.zeros((num_electrodes, num_timepoints, num_subjects))
print(f"Initialized result arrays for {num_subjects} subjects.")

# --- Loop Through Subjects and Images ---
total_start_time = time.time()
print("Starting generation loop...")

for subject_idx in range(num_subjects): # Loop from 0 to 87
    subject_id = subject_idx + 1 # 1-based subject ID
    print(f"Processing Subject ID: {subject_id} ({subject_idx + 1}/{num_subjects})")

    # Array to store generated EEG for this subject for all pictures
    img_subj_EEG = np.zeros((num_electrodes, num_timepoints, num_images))

    # Create one-hot condition vector for the current subject
    conditionTest = np.zeros((1, num_subjects))
    conditionTest[0, subject_idx] = 1
    conditionTest_tensor = torch.tensor(conditionTest, dtype=torch.float32).to(device)

    subj_start_time = time.time()
    for pic_idx in range(num_images):
        # Construct image path
        img_filename = f"{excelPic.iloc[pic_idx, 1]}.jpg" # Assuming column 1 has the base filename
        pic2analyze = os.path.join(image_dir_path, img_filename)

        try:
            # Load and preprocess image
            image = preprocess(Image.open(pic2analyze)).unsqueeze(0).to(device)

            with torch.no_grad():
                # 1. Get image features from CLIP
                image_features = model_CLIP.encode_image(image) # Features are on 'device'

                # 2. Map image features to CVAE latent space
                img_vae_latent_z = model_LATENT_XD(image_features.float(), conditionTest_tensor)

                # 3. Decode latent vector z to generate EEG
                img_EEG_recon = model_EEG_CVAE.decode(img_vae_latent_z, conditionTest_tensor)

            # Store the generated EEG (move to CPU, remove dims)
            img_subj_EEG[:, :, pic_idx] = img_EEG_recon.detach().cpu().numpy().squeeze()

        except FileNotFoundError:
            print(f"  Warning: Image file not found {pic2analyze}. Skipping pic {pic_idx}.")
        except Exception as e:
            print(f"  Error processing image {pic_idx} ({img_filename}): {e}")

    # --- Aggregate Results for this Subject ---
    # Average generated EEG across pictures for each affective category
    # Use boolean indexing based on labelAffect
    RESULTS_EEG_P[:, :, subject_idx] = img_subj_EEG[:, :, labelAffect[:, 0] == 1].mean(axis=2)
    RESULTS_EEG_N[:, :, subject_idx] = img_subj_EEG[:, :, labelAffect[:, 0] == 2].mean(axis=2)
    RESULTS_EEG_U[:, :, subject_idx] = img_subj_EEG[:, :, labelAffect[:, 0] == 3].mean(axis=2)

    subj_end_time = time.time()
    print(f"  Subject {subject_id} finished in {subj_end_time - subj_start_time:.2f} seconds.")

# --- Save Results ---
print(f"\nGeneration loop finished. Saving results to: {output_filename}")

# Create dictionary for saving
results_dict = {
    'RESULTS_EEG_P': RESULTS_EEG_P,
    'RESULTS_EEG_N': RESULTS_EEG_N,
    'RESULTS_EEG_U': RESULTS_EEG_U,
    'labelAffect': labelAffect # Include the label mapping used
}

# Save as .mat file
sio.savemat(output_filename, results_dict)

total_end_time = time.time()
print(f"Results saved successfully.")
print(f"Total execution time for data recovery: {total_end_time - total_start_time:.2f} seconds.")
