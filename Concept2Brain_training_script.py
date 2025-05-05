# -*- coding: utf-8 -*-
"""
Concept2Brain script

Code for processing EEG data, extracting image features with CLIP,
training a Conditional Variational Autoencoder (CVAE) on EEG data,
and training a neural network to map CLIP features to the CVAE's latent space.
"""

# =============================================================================
# Section 1: Imports and Helper Functions
# =============================================================================
# Description: Imports necessary libraries and defines utility functions.

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import clip
from PIL import Image
import numpy as np
import pandas as pd
import random
import scipy.io as sio
from scipy.stats import shapiro
import gc # Garbage Collector to explicitly free memory if needed
import os # To check if files exist (e.g., for loading best model)

# Function to set a seed for reproducibility
def set_seed(seed):
    """Sets the seed for PyTorch (CPU and CUDA), NumPy, and random."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    # These settings help reproducibility on CUDA
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# =============================================================================
# Section 2: Initial Configuration and Environment Setup
# =============================================================================
# Description: Sets the reproducibility seed and configures the device (CPU/GPU).

# Set seed for reproducibility
set_seed(42)

# Configure device to use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# Section 3: Image Feature Extraction with CLIP
# =============================================================================
# Description: Loads the CLIP model, processes images from a specific directory,
# extracts their features (embeddings), and saves them to a .npy file.

print("\n--- Starting CLIP feature extraction ---")
# Load Excel file with image information
# Ensure this path is correct for your system
file_path = '../PictureLabels.xlsx'
excel = pd.read_excel(file_path)

# Load the CLIP model and image preprocessing function
# Using ViT-B/32 model and moving it to the configured device (GPU/CPU)
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Path to the folder containing the images
# Ensure this path is correct for your system
folder_path = '../Pictures_Version12'
# Initialize array to store features (360 images, 512 features per image)
image_features = np.zeros((360, 512))

# Iterate over the images specified in the Excel file
for i in range(360):
    pic_filename = f"{excel.iloc[i, 1]}.jpg"
    pic2analyze = f"{folder_path}/{pic_filename}"
    # print(f"Processing: {pic2analyze}") # Uncomment if debugging needed
    try:
        # Open, preprocess the image, and move it to the device
        image = preprocess_clip(Image.open(pic2analyze)).unsqueeze(0).to(device)
        # Extract image features using CLIP model without gradient calculation
        with torch.no_grad():
            features = model_clip.encode_image(image)
            image_features[i, :] = features.cpu().numpy() # Move features to CPU and store
    except FileNotFoundError:
        print(f"Warning: File not found {pic2analyze}. Skipping.")
    except Exception as e:
        print(f"Error processing {pic2analyze}: {e}")

# Save the extracted features to a NumPy file
np.save('image_features.npy', image_features)
print("Image features saved to image_features.npy")
# Free CLIP model memory if possible (optional)
del model_clip
del preprocess_clip
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# Section 4: EEG Data Loading and Preprocessing
# =============================================================================
# Description: Loads EEG data, labels, and subject information from .mat files.
# Performs normalization on the EEG data.

print("\n--- Loading and preprocessing EEG data ---")
# Base path where scripts and data are located
# Ensure this path is correct for your system
path = './' # Example path, adjust as needed

# Load Clean voltage EEG data (raw) from .mat files and concatenate them
# All trials (> 10k) were divided for python loading reasons into 3 parts
# Ensure these file paths are correct
mat_contents = sio.loadmat(path + 'DATA_EEG_PARTS.ma.mat')
DATA_EEG = np.concatenate((
    np.asarray(mat_contents['DATA_RAW_ALL_PART1']),
    np.asarray(mat_contents['DATA_RAW_ALL_PART2']),
    np.asarray(mat_contents['DATA_RAW_ALL_PART3'])
), axis=2)

print(f"Shape of loaded EEG data: {DATA_EEG.shape}") # E.g., (Electrodes, Timepoints, Trials)

# Load image labels (numbers corresponding to rows in Excel/CLIP features)
# Ensure this file path is correct
tmp = sio.loadmat(path + 'LABEL_ALL_MyAPS12.mat')
LABELS = tmp['PICTURE_XCEL_NUM'].T
print(f"Shape of loaded labels: {LABELS.shape}") # E.g., (Trials, 1)

# Load subject identifiers
# Ensure this file path is correct
tmp = sio.loadmat(path + 'SUBJECTS_ALL_MyAPS12.mat')
SUBJECTS = tmp['SUBJECT_NUM'].T
SUBJECTS_IDS = np.unique(SUBJECTS)
print(f"Shape of loaded subject IDs: {SUBJECTS.shape}") # E.g., (Trials, 1)
print(f"Number of unique subjects: {len(SUBJECTS_IDS)}")

# Create condition vector (one-hot encoding for subjects)
ConditionedVector = np.zeros((len(SUBJECTS), len(SUBJECTS_IDS)))
for i in range(len(SUBJECTS)):
    ConditionedVector[i, SUBJECTS_IDS == SUBJECTS[i]] = 1
print(f"Shape of condition vector (subjects one-hot): {ConditionedVector.shape}") # E.g., (Trials, NumSubjects)

# Z-score normalization of EEG data based on the baseline (first 50 timepoints)
print("Normalizing EEG data...")
# Calculate mean and standard deviation of the baseline period.
# It corresponds with first 50 time points, i.e 100 ms, for each channel and trial
meanData = DATA_EEG[:, :50, :].mean(axis=1, keepdims=True)
stdData = DATA_EEG[:, :50, :].std(axis=1, keepdims=True)
# Avoid division by zero if standard deviation is very small
stdData[stdData < 1e-6] = 1e-6
# Apply Z-score normalization
DATA_EEG_NORM = (DATA_EEG - meanData) / stdData
print("Normalization completed.")

# Free up memory
del mat_contents, tmp, meanData, stdData, DATA_EEG
gc.collect()

# =============================================================================
# Section 5: Data Splitting into Training and Testing Sets
# =============================================================================
# Description: Randomly permutes the data and splits it into training (80%)
# and testing (20%) sets.

print("\n--- Splitting data into training and testing sets ---")
# Generate randomly permuted indices
num_trials = DATA_EEG_NORM.shape[2]
newInds = np.random.permutation(num_trials)

# Calculate the split index for 80/20 division
th_traintest = int(num_trials * 0.8)

# Split the normalized EEG data
# Shape: (Electrodes, Timepoints, Trials) -> (E, T, N_train/N_test)
X_train = DATA_EEG_NORM[:, :, newInds[:th_traintest]]
X_test = DATA_EEG_NORM[:, :, newInds[th_traintest:]]

# Split labels, condition vectors, and subject IDs
# Shape: (Trials, Features) -> (N_train/N_test, Features)
y_train = LABELS[newInds[:th_traintest]]
y_test = LABELS[newInds[th_traintest:]]
c_train = ConditionedVector[newInds[:th_traintest]]
c_test = ConditionedVector[newInds[th_traintest:]]
s_train = SUBJECTS[newInds[:th_traintest]]
s_test = SUBJECTS[newInds[th_traintest:]]

print(f"Training set size: {X_train.shape[2]} trials")
print(f"Testing set size: {X_test.shape[2]} trials")

# Free up memory
del DATA_EEG_NORM, LABELS, ConditionedVector, SUBJECTS, newInds
gc.collect()

# =============================================================================
# Section 6: Conditional Variational Autoencoder (CVAE) Definition
# =============================================================================
# Description: Defines the CVAE architecture using PyTorch, including the
# encoder, decoder, and loss function. The model is designed for multi-GPU use.

print("\n--- Defining the CVAE model ---")

# Function to free GPU memory (if needed)
def free_gpu_memory():
    """Clears the GPU memory cache."""
    if torch.cuda.is_available():
        # print("Clearing GPU cache...") # Can be verbose, uncomment if needed
        torch.cuda.empty_cache()
    gc.collect()

# Definition of the conditional CVAE structure
class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with convolutional layers.
    Designed to operate across multiple GPUs:
    - Encoder on GPU 0
    - Bottleneck layers (mu, logvar) on GPU 1
    - Decoder on GPU 2
    Adjusts to the main device if fewer than 3 GPUs are available.
    """
    def __init__(self, input_channels, condition_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # Check availability of sufficient GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus < 3:
             print(f"Warning: Found {num_gpus} GPUs, but the model is designed for 3. Adjusting to the main device: {device}")
             self.encoder_device = torch.device(device)
             self.bottleneck_device = torch.device(device)
             self.decoder_device = torch.device(device)
        else:
             self.encoder_device = torch.device("cuda:0")
             self.bottleneck_device = torch.device("cuda:1")
             self.decoder_device = torch.device("cuda:2")

        # --- Encoder (GPU 0) ---
        # Input shape: (N, input_channels, 129, 750) [N, C, H, W]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1).to(self.encoder_device) # Output: (N, 32, 65, 375)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1).to(self.encoder_device) # Output: (N, 64, 33, 188)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1).to(self.encoder_device) # Output: (N, 128, 17, 94)
        # Flattened size after conv3: 128 * 17 * 94 = 204224

        # --- Bottleneck Layers (GPU 1) ---
        self.flattened_size = 128 * 17 * 94
        self.fc_mu = nn.Linear(self.flattened_size + condition_dim, latent_dim).to(self.bottleneck_device)
        self.fc_logvar = nn.Linear(self.flattened_size + condition_dim, latent_dim).to(self.bottleneck_device)

        # --- Decoder (GPU 2) ---
        self.fc_decode = nn.Linear(latent_dim + condition_dim, self.flattened_size).to(self.decoder_device)
        # Input to deconv1: (N, 128, 17, 94)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1).to(self.decoder_device) # Output: (N, 64, 33, 188) approx
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1).to(self.decoder_device) # Output: (N, 32, 65, 375) approx
        self.deconv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1).to(self.decoder_device) # Output: (N, C, 129, 750) approx
        # The output size might slightly differ due to padding/stride, will be cropped.

    def encode(self, x, c):
        """Encodes input x conditioned by c into mu and logvar."""
        x = x.to(self.encoder_device)
        # No need to move c here if it's already on the correct device before calling encode
        # c = c.to(self.encoder_device) # Will be moved in the main forward pass if needed

        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2)) # Using relu here is also common

        # Flatten and move to the bottleneck GPU
        h3_flat = h3.view(h3.size(0), -1).to(self.bottleneck_device)
        # Move condition c to the bottleneck GPU as well
        c_bottleneck = c.to(self.bottleneck_device)

        # Concatenate the flattened representation with the condition
        combined_bottleneck = torch.cat([h3_flat, c_bottleneck], dim=1)

        # Calculate mu and logvar
        mu = self.fc_mu(combined_bottleneck)
        logvar = self.fc_logvar(combined_bottleneck)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Performs the reparameterization trick."""
        # Ensure tensors are on the correct device (bottleneck)
        mu = mu.to(self.bottleneck_device)
        logvar = logvar.to(self.bottleneck_device)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Creates tensor on the same device as std
        # Optional clamping for numerical stability
        # std = torch.clamp(std, min=1e-6, max=10) # Example
        return mu + eps * std

    def decode(self, z, c):
        """Decodes latent vector z conditioned by c into reconstruction."""
        # Move z and c to the decoder GPU
        z = z.to(self.decoder_device)
        c_decoder = c.to(self.decoder_device)

        # Concatenate latent vector and condition
        combined_decoder_input = torch.cat([z, c_decoder], dim=1)

        # Pass through the dense layer and reshape
        h6 = torch.relu(self.fc_decode(combined_decoder_input))
        # Reshape to the expected shape for the first deconvolutional layer (N, C, H, W)
        h6_reshaped = h6.view(h6.size(0), 128, 17, 94) # Use post-conv3 dimensions

        # Pass through the deconvolutional layers
        h9 = torch.relu(self.deconv1(h6_reshaped))
        h10 = torch.relu(self.deconv2(h9))
        # Final layer might not have activation (depends on data domain, e.g., sigmoid for [0,1])
        x_recon_raw = self.deconv3(h10)

        # Crop the output to the original input size (129 electrodes, 750 time points)
        # Original input dimensions were (N, 1, 129, 750)
        original_height, original_width = 129, 750
        x_recon = x_recon_raw[:, :, :original_height, :original_width]
        return x_recon

    def forward(self, x, c):
        """Complete forward pass of the CVAE."""
        # Move initial condition c to the encoder device
        c_enc = c.to(self.encoder_device)
        x_enc = x.to(self.encoder_device)

        # Encode
        mu, logvar = self.encode(x_enc, c_enc)

        # Reparameterize (occurs on bottleneck_device)
        z = self.reparameterize(mu, logvar)

        # Decode (requires z and c on decoder_device)
        c_dec = c.to(self.decoder_device)
        x_recon = self.decode(z, c_dec)

        # Ensure mu and logvar are on the correct device for loss calculation (bottleneck device)
        mu_final = mu.to(self.bottleneck_device)
        logvar_final = logvar.to(self.bottleneck_device)

        return x_recon, mu_final, logvar_final

# VAE Loss Function
def vae_loss(x_recon, x, mu, logvar, epoch, beta_factor=0.00001, annealing_midpoint=100, annealing_steepness=0.1):
    """
    Calculates the VAE loss (Reconstruction + KL Divergence).
    Includes a beta factor with sigmoidal annealing for the KL divergence.
    Assumes inputs (mu, logvar) determine the device for calculation.
    """
    # Move all tensors to the same device for loss calculation
    # Assuming loss is calculated on the bottleneck GPU (where mu/logvar reside)
    loss_device = mu.device
    x_recon = x_recon.to(loss_device)
    x = x.to(loss_device)
    # mu and logvar should already be on loss_device

    # 1. Reconstruction Loss (Mean Squared Error)
    # Sum over pixels/timepoints, then average over the batch
    recon_loss = nn.MSELoss(reduction='sum')(x_recon, x) / x.size(0)

    # 2. Kullback-Leibler (KL) Divergence
    # KL divergence between the latent distribution q(z|x,c) and the prior p(z)=N(0,I)
    # Summed over latent dimensions, averaged over the batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss) # Average over the batch

    # 3. Beta Factor with Sigmoidal Annealing
    # Beta starts near 0 and grows towards beta_factor
    beta = beta_factor / (1 + np.exp(-annealing_steepness * (epoch - annealing_midpoint)))

    # 4. Total Loss
    total_loss = recon_loss + beta * kl_loss

    # Return total loss and KL component (for monitoring)
    return total_loss, kl_loss


# =============================================================================
# Section 7: CVAE Training Preparation
# =============================================================================
# Description: Converts data to PyTorch tensors, creates DataLoaders for batch
# handling, instantiates the CVAE model, and defines the optimizer.

print("\n--- Preparing CVAE training ---")

# Convert NumPy data to PyTorch tensors
# Permute dimensions to be (N, C, H, W) where N=trials, C=channels(1), H=electrodes, W=time
# Add channel dimension (unsqueeze(1))
X_train_torch = torch.tensor(X_train, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)
c_train_torch = torch.tensor(c_train, dtype=torch.float32)
c_test_torch = torch.tensor(c_test, dtype=torch.float32)

print(f"Shape of X_train_torch: {X_train_torch.shape}") # E.g., (N_train, 1, E, T)
print(f"Shape of c_train_torch: {c_train_torch.shape}") # E.g., (N_train, NumSubjects)

# Create Datasets and DataLoaders
batch_size_cvae = 64
train_dataset = TensorDataset(X_train_torch, c_train_torch)
# Use multiple workers and pin memory for potentially faster data loading
train_loader = DataLoader(train_dataset, batch_size=batch_size_cvae, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = TensorDataset(X_test_torch, c_test_torch)
test_loader = DataLoader(test_dataset, batch_size=batch_size_cvae, shuffle=False, num_workers=4, pin_memory=True)

print(f"CVAE Batch size: {batch_size_cvae}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# Instantiate the CVAE model
input_channels = 1  # EEG is single-channel (in this context)
latent_dim = 256     # Latent space dimension
condition_dim = c_train.shape[1]  # Number of subjects (dimension of one-hot vector)

# Instantiate the model. Internal modules will be moved to designated GPUs.
model_cvae = ConditionalVAE(input_channels, condition_dim, latent_dim)
# The model object itself resides on the CPU, but its parameters are registered correctly across GPUs.

# Define the optimizer
# Adam is a common choice
learning_rate_cvae = 1e-6 # Very low learning rate for precise tunin
optimizer_cvae = optim.Adam(model_cvae.parameters(), lr=learning_rate_cvae)
print(f"CVAE Optimizer: Adam, Learning Rate: {learning_rate_cvae}")

# Lists to store loss history
train_loss_all_cvae = []
val_loss_all_cvae = []
kl_loss_all_cvae = [] # To monitor KL divergence

# =============================================================================
# Section 8: CVAE Training Loop
# =============================================================================
# Description: Performs the training and validation cycle for the CVAE over a
# defined number of epochs. Logs training progress to a file.

print("\n--- Starting CVAE training ---")
epochs_cvae = 1300 # Number of epochs for training the cVAE

# Annealing parameters for VAE loss (consistent with original settings)
beta_factor_cvae = 0.00001 # Final beta value for KL term weight
annealing_midpoint_cvae = 100 # Epoch where beta reaches half its final value
annealing_steepness_cvae = 0.1 # Controls how quickly beta increases

# Load state if exists (e.g., if training was interrupted)
# This would require saving/loading model state_dict, optimizer state_dict, and loss lists.
# For simplicity, starting from scratch here as in the original script.
start_epoch = 0 # Start from epoch 0

# Open log file to append results
log_filename = 'training_log_cvae.txt'
with open(log_filename, 'a') as f:
    f.write("\n--- New CVAE Training Session ---\n")
    f.write(f"Epochs: {epochs_cvae}, Batch Size: {batch_size_cvae}, LR: {learning_rate_cvae}, Latent Dim: {latent_dim}\n")
    f.write(f"Beta Factor: {beta_factor_cvae}, Annealing Midpoint: {annealing_midpoint_cvae}, Steepness: {annealing_steepness_cvae}\n")

for epoch in range(start_epoch, epochs_cvae):
    # --- Training Phase ---
    model_cvae.train() # Set model to training mode
    train_loss_epoch = 0
    kl_loss_epoch_train = 0

    for batch_idx, (data, condition) in enumerate(train_loader):
        # Data and conditions will be moved to the correct devices inside the model
        # Move initial input to the encoder device
        data = data.to(model_cvae.encoder_device, non_blocking=True)
        condition = condition.to(model_cvae.encoder_device, non_blocking=True)

        # Reset optimizer gradients
        optimizer_cvae.zero_grad()

        # Forward pass: get reconstruction, mu, and logvar
        x_recon, mu, logvar = model_cvae(data, condition)

        # Calculate loss (occurs on bottleneck_device)
        # Move original 'data' to that device for comparison
        data_loss_device = data.to(mu.device, non_blocking=True)
        loss, kl_component = vae_loss(x_recon, data_loss_device, mu, logvar, epoch,
                                      beta_factor=beta_factor_cvae,
                                      annealing_midpoint=annealing_midpoint_cvae,
                                      annealing_steepness=annealing_steepness_cvae)

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer_cvae.step()

        # Accumulate epoch losses
        train_loss_epoch += loss.item()
        kl_loss_epoch_train += kl_component.item()

    # Calculate average training loss for the epoch (per sample)
    train_loss_avg = train_loss_epoch / len(train_loader.dataset)
    kl_loss_avg_train = kl_loss_epoch_train / len(train_loader) # KL average per batch
    train_loss_all_cvae.append(train_loss_avg)


    # --- Validation Phase ---
    model_cvae.eval() # Set model to evaluation mode
    val_loss_epoch = 0
    kl_loss_epoch_val = 0
    mu_all_val = [] # To collect mu values for Shapiro test

    with torch.no_grad(): # Disable gradient calculations during validation
        for batch_idx, (data, condition) in enumerate(test_loader):
            data = data.to(model_cvae.encoder_device, non_blocking=True)
            condition = condition.to(model_cvae.encoder_device, non_blocking=True)

            x_recon, mu, logvar = model_cvae(data, condition)

            data_loss_device = data.to(mu.device, non_blocking=True)
            loss, kl_component = vae_loss(x_recon, data_loss_device, mu, logvar, epoch,
                                          beta_factor=beta_factor_cvae,
                                          annealing_midpoint=annealing_midpoint_cvae,
                                          annealing_steepness=annealing_steepness_cvae)

            val_loss_epoch += loss.item()
            kl_loss_epoch_val += kl_component.item()
            mu_all_val.append(mu.cpu().numpy()) # Store mu for Shapiro test (move to CPU)

    # Calculate average validation loss for the epoch (per sample)
    val_loss_avg = val_loss_epoch / len(test_loader.dataset)
    kl_loss_avg_val = kl_loss_epoch_val / len(test_loader) # KL average per batch
    val_loss_all_cvae.append(val_loss_avg)
    kl_loss_all_cvae.append(kl_loss_avg_val) # Store average validation KL

    # Shapiro-Wilk normality test on 'mu' values from the validation set
    # Concatenate mus from all batches and flatten for Shapiro test
    mu_val_np = np.concatenate(mu_all_val, axis=0).flatten()
    shapiro_stat, shapiro_p = -1, -1 # Default values
    if len(mu_val_np) >= 3: # Shapiro requires at least 3 samples
       try:
           shapiro_stat, shapiro_p = shapiro(mu_val_np)
       except ValueError:
           print(f"Warning: Could not compute Shapiro test for epoch {epoch+1} (possibly zero variance).")


    # Print progress and save to log
    log_message = (
        f"Epoch [{epoch + 1}/{epochs_cvae}], "
        f"Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
        f"KL Val: {kl_loss_avg_val:.3f}, Shapiro p: {shapiro_p:.3g}, "
        # Use last batch mu/logvar for min/max/std stats (representative)
        f"Mu Val (min/max/std): {mu.min():.3f}/{mu.max():.3f}/{mu.std():.3f}, "
        f"LogVar Val (min/max/std): {logvar.min():.3f}/{logvar.max():.3f}/{logvar.std():.3f}"
    )
    print(log_message)
    with open(log_filename, 'a') as f:
        f.write(log_message + "\n")

    # Free GPU memory at the end of each epoch (optional, but can help)
    # free_gpu_memory() # Can slow down training if called too often

print("CVAE training completed.")

# Save the trained model (optional but recommended)
# Save state_dict to CPU for easier loading in different environments
model_cvae_save_path = "MODEL_CCVAE_AnnelingBeta00001_01-100sigmoid_lr1e-6_1300epc_dict.pth"
# Ensure the model's parameters are moved to CPU before saving state_dict
# Or save directly, but specify map_location='cpu' when loading if needed
# torch.save(model_cvae.cpu().state_dict(), model_cvae_save_path)
# print(f"CVAE model state dictionary saved to {model_cvae_save_path}")
# Note: Multi-GPU models might require careful handling for saving/loading. Saving state_dict is generally safer.


# =============================================================================
# Section 9: CVAE Post-Training Visualization and Analysis
# =============================================================================
# Description: Plots loss curves, generates reconstruction examples, and performs
# visual analysis to evaluate the trained CVAE's performance.

print("\n--- CVAE Post-Training Analysis ---")

# 1. Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss_all_cvae, label='Train Loss (Total)')
plt.plot(val_loss_all_cvae, label='Validation Loss (Total)')
# Could also plot KL loss if desired
# plt.plot(kl_loss_all_cvae, label='Validation KL Loss (Avg per Batch)')
plt.xlabel('Epoch')
plt.ylabel('Loss (Per Sample)')
plt.title('CVAE Loss Curves during Training')
plt.legend()
plt.grid(True)
plt.savefig('cvae_loss_curve.png')
# plt.show() # Uncomment if running interactively

# 2. Generate and visualize a reconstruction example from the test set
model_cvae.eval() # Ensure the model is in evaluation mode
with torch.no_grad():
    # Take a sample from the test set
    idx_sample = 0 # Use the first sample
    x_sample = X_test_torch[idx_sample].unsqueeze(0) # Add batch dimension
    c_sample = c_test_torch[idx_sample].unsqueeze(0)

    # Move sample to the appropriate initial device (encoder)
    x_sample_dev = x_sample.to(model_cvae.encoder_device)
    c_sample_dev = c_sample.to(model_cvae.encoder_device)

    # Generate reconstruction
    x_recon_sample, mu_sample, logvar_sample = model_cvae(x_sample_dev, c_sample_dev)

    # Move results to CPU for visualization
    x_original_np = x_sample.squeeze().cpu().numpy() # Remove batch/channel, move to CPU
    x_reconstructed_np = x_recon_sample.squeeze().cpu().numpy() # Remove batch/channel, move to CPU
    mu_sample_np = mu_sample.cpu().numpy()
    logvar_sample_np = logvar_sample.cpu().numpy()

# Visualize original vs. reconstructed (as image and time series)
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(x_original_np, aspect='auto', cmap='viridis')
plt.title(f'Original EEG (Test Sample {idx_sample})')
plt.xlabel('Time (points)')
plt.ylabel('Electrode')
plt.colorbar()

# Reconstructed image
plt.subplot(2, 3, 2)
plt.imshow(x_reconstructed_np, aspect='auto', cmap='viridis')
plt.title('CVAE Reconstruction')
plt.xlabel('Time (points)')
plt.ylabel('Electrode')
plt.colorbar()

# Difference image
plt.subplot(2, 3, 3)
diff = x_original_np - x_reconstructed_np
vmax = np.max(np.abs(diff))
plt.imshow(diff, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
plt.title('Difference (Original - Reconstructed)')
plt.xlabel('Time (points)')
plt.ylabel('Electrode')
plt.colorbar()


# Original time series (all electrodes overlaid)
plt.subplot(2, 3, 4)
plt.plot(x_original_np.T) # Transpose so time is the x-axis
plt.title('Original EEG (Time Series)')
plt.xlabel('Time (points)')
plt.ylabel('Normalized Amplitude')

# Reconstructed time series
plt.subplot(2, 3, 5)
plt.plot(x_reconstructed_np.T)
plt.title('CVAE Reconstruction (Time Series)')
plt.xlabel('Time (points)')
plt.ylabel('Normalized Amplitude')

# Latent space (mu and logvar) for this sample
plt.subplot(2, 3, 6)
plt.plot(mu_sample_np.flatten(), label=f'Mu (std={mu_sample_np.std():.2f})')
plt.plot(logvar_sample_np.flatten(), label=f'LogVar (std={logvar_sample_np.std():.2f})')
plt.title('Latent Space (mu, logvar)')
plt.xlabel('Latent Dimension')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.savefig('cvae_reconstruction_sample.png')
# plt.show()

# 3. Analysis of Subject Effects (requires generating multiple reconstructions)
# Original code generated reconstructions for the first 3 subject conditions (c_test[0], c_test[1], c_test[2])
# applied to *all* X_test inputs. This checks if the model can impose a subject's "style".

print("Generating reconstructions conditioned by different subjects...")
model_cvae.eval()
num_test_samples = X_test_torch.shape[0]
num_subjects_to_test = min(3, c_test.shape[1]) # Test with the first 3 subjects
reconstructions_by_subject = {} # Dictionary: {subject_idx: [reconstructions]}

with torch.no_grad():
    for subj_idx in range(num_subjects_to_test):
        # Create condition vector for this subject (one-hot)
        c_subject = torch.zeros(1, condition_dim)
        c_subject[0, subj_idx] = 1
        # Move the *target* condition to the decoder device
        c_subject_target = c_subject.to(model_cvae.decoder_device)

        subject_recons = []
        # Iterate over a subset of test samples for efficiency
        samples_to_process = min(num_test_samples, 100) # Limit to 100 samples
        print(f"  Processing {samples_to_process} samples for subject {subj_idx+1} condition...")
        for i in range(samples_to_process):
            x_input = X_test_torch[i].unsqueeze(0).to(model_cvae.encoder_device)
             # Use the *original* condition associated with x_input for encoding
            c_input_original = c_test_torch[i].unsqueeze(0).to(model_cvae.encoder_device)

            # Encode to get mu, logvar using the original input and condition
            mu, logvar = model_cvae.encode(x_input, c_input_original)
            # Reparameterize to get z (on bottleneck device)
            z = model_cvae.reparameterize(mu, logvar)
            # Decode using the obtained z but with the *new target* subject condition
            x_recon_subj = model_cvae.decode(z, c_subject_target)
            subject_recons.append(x_recon_subj.squeeze().cpu().numpy()) # Collect reconstructions

        # Store all reconstructions for this target subject condition
        reconstructions_by_subject[subj_idx] = np.stack(subject_recons, axis=-1) # Shape: (E, T, N_samples)

# Visualize the average ERP for each simulated subject condition
plt.figure(figsize=(15, 5))
plt.suptitle('Average Simulated ERP Conditioned by Subject (Using Test Set Inputs)')
for subj_idx in range(num_subjects_to_test):
    # Average across the samples processed for this condition
    erp_mean = reconstructions_by_subject[subj_idx].mean(axis=2)
    plt.subplot(1, num_subjects_to_test, subj_idx + 1)
    plt.plot(erp_mean.T) # Plot time series
    plt.title(f'Simulated Subject {subj_idx + 1}')
    plt.xlabel('Time (points)')
    if subj_idx == 0:
        plt.ylabel('Normalized Amplitude')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.savefig('cvae_subject_conditioning_erp.png')
# plt.show()

# 4. Analysis of Emotional Condition Effects (requires mapping labels to conditions)
print("Analyzing average reconstructions by emotional condition (Pleasant, Neutral, Unpleasant)...")
# Define label ranges for each condition (based on original code comments)
pics_pleasant = list(range(322, 362))
pics_neutral = list(range(282, 322))
pics_unpleasant = list(range(242, 282))

# Need the *actual* reconstructions from the test set (not simulated ones)
all_test_recons = []
all_test_mu = [] # Also collect mu if needed later
print("  Generating reconstructions for the entire test set...")
with torch.no_grad():
    for data, condition in test_loader:
        data_dev = data.to(model_cvae.encoder_device)
        cond_dev = condition.to(model_cvae.encoder_device)
        x_recon, mu, _ = model_cvae(data_dev, cond_dev)
        all_test_recons.append(x_recon.cpu().numpy())
        all_test_mu.append(mu.cpu().numpy())

all_test_recons_np = np.concatenate(all_test_recons, axis=0).squeeze() # Shape: (N_test, E, T)
# y_test has shape (N_test, 1), need (N_test,) for isin
y_test_flat = y_test.squeeze()

# Find indices in the test set corresponding to each emotional condition
indices_p = np.where(np.isin(y_test_flat, pics_pleasant))[0]
indices_n = np.where(np.isin(y_test_flat, pics_neutral))[0]
indices_u = np.where(np.isin(y_test_flat, pics_unpleasant))[0]

# Calculate average ERP for each condition based on the reconstructions
erp_p_recon = all_test_recons_np[indices_p].mean(axis=0) # Shape: (E, T)
erp_n_recon = all_test_recons_np[indices_n].mean(axis=0)
erp_u_recon = all_test_recons_np[indices_u].mean(axis=0)

# Visualize average reconstructed ERPs by condition
plt.figure(figsize=(18, 6))
plt.suptitle('Average Reconstructed ERP by Emotional Condition')

# Time series plot (all electrodes)
plt.subplot(1, 2, 1)
# Use different linestyles or colors if plotting all electrodes is too crowded
plt.plot(erp_p_recon.T, label='Pleasant Recon', alpha=0.1, color='green')
plt.plot(erp_n_recon.T, label='Neutral Recon', alpha=0.1, color='blue')
plt.plot(erp_u_recon.T, label='Unpleasant Recon', alpha=0.1, color='red')
# Create a custom legend for clarity
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='green', lw=2),
                Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2)]
plt.legend(custom_lines, ['Pleasant', 'Neutral', 'Unpleasant'])
plt.title('All Reconstructed Signals')
plt.xlabel('Time (points)')
plt.ylabel('Normalized Amplitude')


# Plot for a specific electrode (e.g., electrode 61 corresponds to Pz)
plt.subplot(1, 2, 2)
electrode_idx = 61 # Check if this index is valid (0 to E-1)
if 0 <= electrode_idx < erp_p_recon.shape[0]:
     plt.plot(erp_p_recon[electrode_idx, :], label='Pleasant Recon', color='green')
     plt.plot(erp_n_recon[electrode_idx, :], label='Neutral Recon', color='blue')
     plt.plot(erp_u_recon[electrode_idx, :], label='Unpleasant Recon', color='red')
     plt.title(f'Reconstructed ERP - Electrode {electrode_idx+1}')
     plt.xlabel('Time (points)')
     plt.ylabel('Normalized Amplitude')
     plt.legend()
     plt.grid(True)
else:
     plt.text(0.5, 0.5, f'Electrode index {electrode_idx} out of range [0, {erp_p_recon.shape[0]-1}]',
              horizontalalignment='center', verticalalignment='center')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cvae_condition_analysis_reconstructed.png')
# plt.show()

# Free up memory
del X_train_torch, X_test_torch, c_train_torch, c_test_torch, train_loader, test_loader
del all_test_recons, all_test_recons_np, all_test_mu
if 'reconstructions_by_subject' in locals(): del reconstructions_by_subject
free_gpu_memory()


# =============================================================================
# Section 10: Data Preparation for Latent Neural Network (CLIP -> CVAE)
# =============================================================================
# Description: Loads the saved CLIP features and creates corresponding datasets
# to train a network mapping from CLIP's latent space to the CVAE's latent space.

print("\n--- Preparing data for the Latent Neural Network (CLIP -> CVAE) ---")

# Load previously saved CLIP image features
try:
    # Ensure the path matches where features were saved
    image_features_loaded = np.load('image_features.npy')
    print(f"CLIP features loaded from image_features.npy, shape: {image_features_loaded.shape}")
except FileNotFoundError:
    print("Error: File image_features.npy not found. Run Section 3 first.")
    exit() # Exit if features cannot be loaded

# Create input datasets (CLIP features) for training and testing
# Use y_train, y_test labels to find the correct CLIP features
# Labels (y_train, y_test) are 1-based image numbers, subtract 1 for 0-based indexing
CLIP_latent_train = image_features_loaded[y_train.squeeze() - 1]
CLIP_latent_test = image_features_loaded[y_test.squeeze() - 1]

print(f"Shape of CLIP_latent_train: {CLIP_latent_train.shape}") # (N_train, 512)
print(f"Shape of CLIP_latent_test: {CLIP_latent_test.shape}") # (N_test, 512)

# The 'targets' for this network are not directly the CVAE's mu/logvar.
# Instead, the loss is calculated by comparing the *reconstructed EEG* generated
# from the 'z' predicted by this network, with the original EEG.
# Therefore, we need the original EEG data (X) and conditions (c) again.

# Convert data to PyTorch tensors for the latent network
input_clip_train = torch.tensor(CLIP_latent_train, dtype=torch.float32)
input_clip_test = torch.tensor(CLIP_latent_test, dtype=torch.float32)

# Need original EEG data (X) and conditions (c) again
# Reload or reuse tensors if still in memory (reloading for clarity is safer)
X_train_torch_latent = torch.tensor(X_train, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)
X_test_torch_latent = torch.tensor(X_test, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)
c_train_torch_latent = torch.tensor(c_train, dtype=torch.float32)
c_test_torch_latent = torch.tensor(c_test, dtype=torch.float32)

# Create DataLoaders for the latent network
batch_size_latent = 512 # Based on original code
# Dataset contains (CLIP_input, Original_EEG_output, Condition_input)
latent_train_dataset = TensorDataset(input_clip_train, X_train_torch_latent, c_train_torch_latent)
latent_train_loader = DataLoader(latent_train_dataset, batch_size=batch_size_latent, shuffle=True, num_workers=4, pin_memory=True)

latent_test_dataset = TensorDataset(input_clip_test, X_test_torch_latent, c_test_torch_latent)
latent_test_loader = DataLoader(latent_test_dataset, batch_size=batch_size_latent, shuffle=False, num_workers=4, pin_memory=True)

print(f"DataLoader for LatentNN created. Batch size: {batch_size_latent}")
print(f"Number of LatentNN training batches: {len(latent_train_loader)}")
print(f"Number of LatentNN testing batches: {len(latent_test_loader)}")

# Free memory (original X_train, X_test arrays no longer needed if tensors created)
del X_train, X_test, y_train, c_train, s_train, y_test, c_test, s_test
gc.collect()


# =============================================================================
# Section 11: Latent Neural Network (LatentNN) Definition
# =============================================================================
# Description: Defines the architecture of the neural network mapping CLIP
# features and subject condition to the CVAE's latent space.

print("\n--- Defining the Latent Neural Network (LatentNN) ---")

# Network parameters (based on original code)
input_features_clip = CLIP_latent_train.shape[1] # 512
condition_features_subj = c_train_torch_latent.shape[1] # NumSubjects (e.g., 88)
output_features_latent = latent_dim # 256 (CVAE's latent dimension)
hidden_units_1 = 256
hidden_units_2 = 512
# Original code had commented-out layers (BatchNorm, Dropout), omitted here to match active version.

class LatentNN(nn.Module):
    """
    Neural Network to map (CLIP features + Subject Condition) -> CVAE Latent Space (z).
    Designed to operate on a specific GPU (GPU 1 according to CVAE setup).
    """
    def __init__(self):
        super(LatentNN, self).__init__()
        # Device where this network will operate (GPU 1 if available, matching CVAE bottleneck)
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
             # Assumes GPU 1 corresponds to the CVAE bottleneck device
             self.device_latent = torch.device("cuda:1")
        else:
             print("Warning: Not enough GPUs for original assignment (LatentNN on GPU 1). Using main device.")
             self.device_latent = torch.device(device)

        print(f"LatentNN will operate on: {self.device_latent}")

        # Define layers and move them to the designated device
        self.fc1 = nn.Linear(input_features_clip + condition_features_subj, hidden_units_1).to(self.device_latent)
        self.fc2 = nn.Linear(hidden_units_1, hidden_units_2).to(self.device_latent)
        self.fc3 = nn.Linear(hidden_units_2, output_features_latent).to(self.device_latent)
        self.elu = nn.ELU(alpha=1.0) # ELU activation used in original code

    def forward(self, x_clip, c_subj):
        """Forward pass."""
        # Move inputs to this network's device
        x_clip = x_clip.to(self.device_latent)
        c_subj = c_subj.to(self.device_latent)

        # Concatenate CLIP features and subject condition
        xc_combined = torch.cat([x_clip, c_subj], dim=1)

        # Pass through layers
        hidden1 = self.elu(self.fc1(xc_combined))
        hidden2 = self.elu(self.fc2(hidden1))
        z_predicted = self.fc3(hidden2) # Linear output (predicting the latent vector z)

        return z_predicted

# Instantiate the LatentNN model
model_latent = LatentNN()
# The model already moves its layers to the correct device in __init__

# =============================================================================
# Section 12: LatentNN Training Preparation
# =============================================================================
# Description: Defines the loss function (MSE on reconstructed EEG), the
# optimizer, and sets the CVAE to evaluation mode (freezing its weights).

print("\n--- Preparing LatentNN training ---")

# Loss function: Mean Squared Error (MSE) between original EEG and EEG reconstructed
# from the z predicted by LatentNN.
criterion_latent = nn.MSELoss()

# Optimizer for LatentNN
learning_rate_latent = 1e-3 # Based on original code
optimizer_latent = optim.Adam(model_latent.parameters(), lr=learning_rate_latent)
print(f"LatentNN Optimizer: Adam, LR: {learning_rate_latent}")

# Freeze CVAE weights: we don't want to train it further
print("Freezing CVAE weights...")
for param in model_cvae.parameters():
    param.requires_grad = False
model_cvae.eval() # Set CVAE to evaluation mode (important if it uses Dropout/BatchNorm)

# Lists to store LatentNN loss history
train_loss_all_latent = []
val_loss_all_latent = []

# Parameters for Early Stopping (based on original code)
patience = 30 # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0
best_val_loss = float('inf')
best_model_latent_path = 'MODEL_Latent_10epc_512batch_dict.pth' # File to save the best model
# In original script a model fit in 10 epochs was the best found option.

# =============================================================================
# Section 13: LatentNN Training Loop
# =============================================================================
# Description: Performs the training and validation cycle for LatentNN, using
# the frozen CVAE to decode the predicted 'z' and calculate loss on the EEG
# reconstruction. Implements Early Stopping.

print("\n--- Starting LatentNN training ---")
epochs_latent = 1000 # Maximum number of epochs (may stop earlier due to Early Stopping)
#Original script stopped at 10 epochs when evaluation data was not better.

for epoch in range(epochs_latent):
    # --- Training Phase ---
    model_latent.train() # Set LatentNN to training mode
    epoch_loss_train = 0.0

    for batch_clip, batch_eeg_original, batch_condition in latent_train_loader:
        # batch_clip & batch_condition go to LatentNN device
        # batch_eeg_original goes to CVAE Decoder/Loss device
        # batch_condition also needed on Decoder device

        # Move condition to LatentNN device (and later to decoder device)
        batch_condition_latent = batch_condition.to(model_latent.device_latent, non_blocking=True)
        # Move clip features to LatentNN device
        batch_clip_latent = batch_clip.to(model_latent.device_latent, non_blocking=True)

        # Reset LatentNN gradients
        optimizer_latent.zero_grad()

        # 1. Predict z using LatentNN
        z_hat = model_latent(batch_clip_latent, batch_condition_latent)

        # 2. Decode z_hat using the frozen CVAE to get reconstructed EEG
        # Move predicted z_hat and condition to the CVAE decoder device
        z_hat_decoder = z_hat.to(model_cvae.decoder_device, non_blocking=True)
        batch_condition_decoder = batch_condition.to(model_cvae.decoder_device, non_blocking=True)
        X_hat_reconstructed = model_cvae.decode(z_hat_decoder, batch_condition_decoder)

        # 3. Calculate MSE loss between reconstruction and original EEG
        # Move original EEG to the device where loss is calculated (decoder device)
        batch_eeg_original_loss = batch_eeg_original.to(model_cvae.decoder_device, non_blocking=True)
        loss = criterion_latent(X_hat_reconstructed, batch_eeg_original_loss)

        # Backward pass (computes gradients only for LatentNN)
        loss.backward()

        # Update LatentNN weights
        optimizer_latent.step()

        epoch_loss_train += loss.item()

    # Calculate average training loss for the epoch (per sample)
    avg_epoch_loss_train = epoch_loss_train / len(latent_train_loader.dataset)
    train_loss_all_latent.append(avg_epoch_loss_train)


    # --- Validation Phase ---
    model_latent.eval() # Set LatentNN to evaluation mode
    epoch_loss_val = 0.0
    z_hat_last_batch = None # To print min/max/std stats

    with torch.no_grad():
        for batch_clip, batch_eeg_original, batch_condition in latent_test_loader:
            batch_condition_latent = batch_condition.to(model_latent.device_latent, non_blocking=True)
            batch_clip_latent = batch_clip.to(model_latent.device_latent, non_blocking=True)

            # 1. Predict z
            z_hat = model_latent(batch_clip_latent, batch_condition_latent)
            z_hat_last_batch = z_hat # Store for stats

            # 2. Decode z_hat
            z_hat_decoder = z_hat.to(model_cvae.decoder_device, non_blocking=True)
            batch_condition_decoder = batch_condition.to(model_cvae.decoder_device, non_blocking=True)
            X_hat_reconstructed = model_cvae.decode(z_hat_decoder, batch_condition_decoder)

            # 3. Calculate loss
            batch_eeg_original_loss = batch_eeg_original.to(model_cvae.decoder_device, non_blocking=True)
            loss = criterion_latent(X_hat_reconstructed, batch_eeg_original_loss)

            epoch_loss_val += loss.item()

    # Calculate average validation loss for the epoch (per sample)
    avg_epoch_loss_val = epoch_loss_val / len(latent_test_loader.dataset)
    val_loss_all_latent.append(avg_epoch_loss_val)

    # Print progress
    z_min, z_max, z_std = -1, -1, -1 # Default values
    if z_hat_last_batch is not None:
        z_min = z_hat_last_batch.min().item()
        z_max = z_hat_last_batch.max().item()
        z_std = z_hat_last_batch.std().item()

    print(f"Epoch {epoch+1}/{epochs_latent}, Train Loss: {avg_epoch_loss_train:.4f}, Val Loss: {avg_epoch_loss_val:.4f}, "
          f"z_hat Val (min/max/std): {z_min:.3f}/{z_max:.3f}/{z_std:.3f}")

    # Early Stopping logic
    if avg_epoch_loss_val < best_val_loss:
        best_val_loss = avg_epoch_loss_val
        epochs_no_improve = 0
        # Save the best model found so far
        torch.save(model_latent.state_dict(), best_model_latent_path)
        print(f"  Validation Loss improved. Saving model to {best_model_latent_path}")
    else:
        epochs_no_improve += 1
        print(f"  Validation Loss did not improve for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after epoch {epoch + 1}.")
        break

    # Optional memory clearing
    # free_gpu_memory()


print("LatentNN training completed.")
# Load the best model saved by Early Stopping
if os.path.exists(best_model_latent_path):
    print(f"Loading best model state from {best_model_latent_path}")
    # Ensure map_location matches the device LatentNN is intended for
    model_latent.load_state_dict(torch.load(best_model_latent_path, map_location=model_latent.device_latent))
else:
    print(f"Warning: Best model file {best_model_latent_path} not found. Using the model's last state.")

# =============================================================================
# Section 14: LatentNN Post-Training Visualization and Analysis
# =============================================================================
# Description: Plots LatentNN loss curves, generates EEG from test CLIP features,
# and analyzes the results by emotional condition.

print("\n--- LatentNN Post-Training Analysis ---")

# 1. Plot LatentNN loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss_all_latent, label='Train Loss (LatentNN)')
plt.plot(val_loss_all_latent, label='Validation Loss (LatentNN)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (on EEG reconstruction, per sample)')
plt.title('LatentNN Loss Curves during Training')
plt.legend()
plt.grid(True)
plt.yscale('log') # Loss might decrease significantly, log scale can be useful
plt.savefig('latentnn_loss_curve.png')
# plt.show()

# 2. Generate EEG from test CLIP features using LatentNN + CVAE(decoder)
print("Generating EEG from test CLIP features...")
model_latent.eval()
model_cvae.eval()

# Process the test set (or a large subset as in original code)
num_samples_to_generate = min(2000, len(latent_test_dataset)) # Limit to 2000
generated_eegs = []
generated_latents = [] # Store the predicted 'z' vectors

# Use DataLoader or iterate manually. Manual iteration for clarity:
# Need the CLIP inputs and conditions for the subset
input_clip_test_subset = input_clip_test[:num_samples_to_generate]
c_test_torch_subset = c_test_torch_latent[:num_samples_to_generate]
# Also need the corresponding labels for analysis
y_test_subset = y_test[:num_samples_to_generate] # Assuming y_test numpy array is still available

print(f"  Processing {num_samples_to_generate} test samples...")
with torch.no_grad():
    # Process in batches if the subset is too large for GPU memory
    batch_size_inference = 256 # Adjust based on GPU memory
    for i in range(0, num_samples_to_generate, batch_size_inference):
        # Prepare batch inputs
        batch_clip = input_clip_test_subset[i:i+batch_size_inference].to(model_latent.device_latent)
        batch_cond = c_test_torch_subset[i:i+batch_size_inference].to(model_latent.device_latent)

        # 1. Predict z using LatentNN
        z_hat_batch = model_latent(batch_clip, batch_cond)
        generated_latents.append(z_hat_batch.cpu().numpy()) # Store predicted z

        # 2. Decode z using CVAE
        z_hat_dec = z_hat_batch.to(model_cvae.decoder_device)
        batch_cond_dec = batch_cond.to(model_cvae.decoder_device) # Condition also needed for decoder
        X_hat_batch = model_cvae.decode(z_hat_dec, batch_cond_dec)
        generated_eegs.append(X_hat_batch.cpu().numpy()) # Store generated EEG

# Concatenate results from all batches
generated_eegs_np = np.concatenate(generated_eegs, axis=0).squeeze() # Shape: (N_gen, E, T)
generated_latents_np = np.concatenate(generated_latents, axis=0) # Shape: (N_gen, latent_dim)

print(f"Shape of generated EEGs: {generated_eegs_np.shape}")
print(f"Shape of generated latents (z_hat): {generated_latents_np.shape}")


# 3. Visualize the generated latent space (z_hat) by emotional condition
# Use the labels corresponding to the generated subset
y_test_subset_flat = y_test_subset.squeeze()
# Find indices within the subset for each condition
indices_p_gen = np.where(np.isin(y_test_subset_flat, pics_pleasant))[0]
indices_n_gen = np.where(np.isin(y_test_subset_flat, pics_neutral))[0]
indices_u_gen = np.where(np.isin(y_test_subset_flat, pics_unpleasant))[0]

# Group predicted latents by condition
latents_p = generated_latents_np[indices_p_gen]
latents_n = generated_latents_np[indices_n_gen]
latents_u = generated_latents_np[indices_u_gen]
# Concatenate for visualization (ensure order P, N, U for consistency)
concatenated_latents = np.concatenate((latents_p, latents_n, latents_u), axis=0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(concatenated_latents, aspect='auto', cmap='viridis')
plt.title('Generated Latent Space Z_hat (P, N, U ordered)')
plt.xlabel('Latent Dimension')
plt.ylabel('Samples (Ordered Pleasant-Neutral-Unpleasant)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.plot(latents_p.mean(axis=0), label='Pleasant (Mean Z_hat)')
plt.plot(latents_n.mean(axis=0), label='Neutral (Mean Z_hat)')
plt.plot(latents_u.mean(axis=0), label='Unpleasant (Mean Z_hat)')
plt.title('Average Predicted Latent Vector Z_hat by Condition')
plt.xlabel('Latent Dimension')
plt.ylabel('Average Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('latentnn_latent_space_analysis.png')
# plt.show()


# 4. Analysis of the generated EEGs by emotional condition
# Calculate average generated ERP for each condition
erp_p_gen = generated_eegs_np[indices_p_gen].mean(axis=0) # Shape: (E, T)
erp_n_gen = generated_eegs_np[indices_n_gen].mean(axis=0)
erp_u_gen = generated_eegs_np[indices_u_gen].mean(axis=0)

plt.figure(figsize=(18, 12))
plt.suptitle('Average Generated ERP (CLIP -> LatentNN -> CVAE) by Emotional Condition')

# Time series plot (all electrodes overlaid - might be messy)
plt.subplot(2, 2, 1)
plt.plot(erp_p_gen.T, alpha=0.1, color='green') # Pleasant generated
plt.plot(erp_n_gen.T, alpha=0.1, color='blue')  # Neutral generated
plt.plot(erp_u_gen.T, alpha=0.1, color='red')   # Unpleasant generated
# Custom legend for clarity
custom_lines_gen = [Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
plt.legend(custom_lines_gen, ['Pleasant (Gen)', 'Neutral (Gen)', 'Unpleasant (Gen)'])
plt.title('Generated Signals (All Electrodes)')
plt.xlabel('Time (points)')
plt.ylabel('Normalized Amplitude')


# Image plots (topographical maps over time)
# Determine symmetric color limits based on max absolute amplitude
clim_max_gen = np.max(np.abs(np.stack([erp_p_gen, erp_n_gen, erp_u_gen])))
clim_gen = (-clim_max_gen, clim_max_gen)

plt.subplot(2, 2, 2)
plt.imshow(erp_p_gen, aspect='auto', cmap='viridis', vmin=clim_gen[0], vmax=clim_gen[1])
plt.title('Pleasant (Generated)')
plt.xlabel('Time (points)'), plt.ylabel('Electrode'), plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(erp_n_gen, aspect='auto', cmap='viridis', vmin=clim_gen[0], vmax=clim_gen[1])
plt.title('Neutral (Generated)')
plt.xlabel('Time (points)'), plt.ylabel('Electrode'), plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(erp_u_gen, aspect='auto', cmap='viridis', vmin=clim_gen[0], vmax=clim_gen[1])
plt.title('Unpleasant (Generated)')
plt.xlabel('Time (points)'), plt.ylabel('Electrode'), plt.colorbar()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.savefig('latentnn_condition_analysis_generated_eeg_maps.png')
# plt.show()

# Plot for the specific electrode (61) as in the original code
plt.figure(figsize=(8, 5))
electrode_idx = 61 # Ensure this is a valid index (0 to E-1)
if 0 <= electrode_idx < erp_p_gen.shape[0]:
     plt.plot(erp_p_gen[electrode_idx, :], label='Pleasant (Generated)', color='green')
     plt.plot(erp_n_gen[electrode_idx, :], label='Neutral (Generated)', color='blue')
     plt.plot(erp_u_gen[electrode_idx, :], label='Unpleasant (Generated)', color='red')
     plt.title(f'Generated EEG - Electrode {electrode_idx+1}')
     plt.xlabel('Time (points)')
     plt.ylabel('Normalized Amplitude')
     plt.legend()
     plt.grid(True)
else:
     plt.text(0.5, 0.5, f'Electrode index {electrode_idx} out of range [0, {erp_p_gen.shape[0]-1}]',
              horizontalalignment='center', verticalalignment='center')

plt.savefig('latentnn_condition_analysis_generated_eeg_electrode_61.png')
# plt.show()


print("\n--- Analysis completed ---")

# Optional: Explicitly clear memory at the end
del model_cvae, model_latent
del latent_train_dataset, latent_test_dataset, latent_train_loader, latent_test_loader
del generated_eegs_np, generated_latents_np
free_gpu_memory()





