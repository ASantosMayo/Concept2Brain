# -*- coding: utf-8 -*-
"""
Concept2Brain text test

Code for assesing model performance on EEG generation from text descriptions.
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt # Added for plotting

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
        # Clamping the standard deviation to avoid explosion/implosion
        # Keeping original clamp logic:
        std = torch.clamp(std, min=-5, max=2.0) # Warning: Clamping std to negative values is unusual. Check original intent.
        return mu + eps * std

    def decode(self, z, c):
        """Decodes latent vector z conditioned by c into reconstruction."""
        # Move z and condition to the bottleneck device for concatenation & fc_decode
        z = z.to(self.bottleneck_device)
        c = c.to(self.bottleneck_device)
        combined_input = torch.cat([z, c], dim=1)

        h6_bottleneck = torch.relu(self.fc_decode(combined_input))

        # Move to decoder device and reshape
        h6 = h6_bottleneck.to(self.decoder_device)
        h6 = h6.view(h6.size(0), 128, 17, 94) # Reshape to (N, C, H, W) for deconv

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

        # Decode
        c_bottleneck_dev = c.to(self.bottleneck_device)
        x_recon = self.decode(z, c_bottleneck_dev)

        # Return reconstruction and latent parameters (on bottleneck device)
        return x_recon, mu.to(self.bottleneck_device), logvar.to(self.bottleneck_device)


# Definition of the Latent Network (CLIP -> CVAE Latent Space)
# Note: Comments mention BatchNorm/Dropout layers which are commented out in the forward pass.
class LATENT_XD(nn.Module):
    """
    Neural Network mapping (CLIP features + Condition) to CVAE latent space (z).
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
# Note: Device is redefined later before the processing loop. Ensure consistency.
device_load = torch.device("cpu")
print(f"Device used for initial model loading: {device_load}")

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
model_EEG_CVAE = EEG_CVAE(input_channels, condition_dim, latent_dim).to(device_load)
model_LATENT_XD = LATENT_XD(input_features, condition_dim, hidden_units_1, hidden_units_2, output_features).to(device_load)
# CLIP model will be loaded later with the potentially different processing device

print("Models instantiated on CPU.")

# --- Load Pre-trained Weights ---
# Define paths to the saved model weights
# IMPORTANT: Ensure these paths are correct for your system!
cvae_weights_path = "MODEL_CCVAE_AnnelingBeta00001_01-100sigmoid_lr1e-6_1300epc_dict.pth"
latent_xd_weights_path = "MODEL_Latent_10epc_512batch_dict.pth"

print("Loading pre-trained weights...")
# Load CVAE weights
if os.path.exists(cvae_weights_path):
    state_dict_cvae = torch.load(cvae_weights_path, map_location=device_load)
    model_EEG_CVAE.load_state_dict(state_dict_cvae)
    model_EEG_CVAE.eval() # Set CVAE to evaluation mode
    print("EEG_CVAE weights loaded successfully.")
else:
    print(f"Error: EEG_CVAE weights file not found at {cvae_weights_path}")
    # exit() # Optional: stop script if weights are essential

# Load LATENT_XD weights
if os.path.exists(latent_xd_weights_path):
    state_dict_latent = torch.load(latent_xd_weights_path, map_location=device_load)
    model_LATENT_XD.load_state_dict(state_dict_latent)
    model_LATENT_XD.eval() # Set Latent model to evaluation mode
    print("LATENT_XD weights loaded successfully.")
else:
    print(f"Error: LATENT_XD weights file not found at {latent_xd_weights_path}")
    # exit() # Optional: stop script if weights are essential

# --- Subject IDs ---
# Placeholder if SUBJECTS_IDS is not loaded from elsewhere
if 'SUBJECTS_IDS' not in locals():
    print("Warning: SUBJECTS_IDS not found. Creating placeholder based on condition_dim.")
    SUBJECTS_IDS = np.arange(1, condition_dim + 1) # Array [1, 2, ..., 88]

# =============================================================================
# Experimental Conditions (Text Descriptions)
# =============================================================================
print("\n--- Defining experimental text conditions ---")

# List of text descriptions intended to evoke pleasant emotions
experimental_conditions_pleasant = [
    "A picture of a serene sunrise over a calm lake",
    "A picture of a group of friends laughing in a sunlit park",
    "A picture of a family enjoying a joyful picnic on a warm day",
    "A picture of a beautiful baby giggling in a colorful nursery",
    "A picture of a vibrant garden filled with blooming flowers and happy visitors",
    "A picture of a couple holding hands during a romantic evening walk",
    "A picture of children playing with bubbles in a bright backyard",
    "A picture of a peaceful beach with gentle waves and a clear blue sky",
    "A picture of a birthday party with smiling children and festive decorations",
    "A picture of a joyful dog running freely in an open field",
    "A picture of an artist painting a vivid landscape with passion",
    "A picture of a group of coworkers celebrating a well-earned success",
    "A picture of a chef happily preparing a delicious meal in a cozy kitchen",
    "A picture of a runner crossing the finish line with pride and joy",
    "A picture of a person meditating in a tranquil garden at dawn", # Note: Original had "meditando"
    "A picture of a couple dancing gracefully under twinkling lights",
    "A picture of a community festival filled with music, color, and smiles",
    "A picture of a toddler hugging a soft toy with pure delight",
    "A picture of a musician performing live to an enthusiastic crowd",
    "A picture of a family gathered around a table for a festive dinner",
    "A picture of a couple sharing a tender kiss under a starlit sky",
    "A picture of friends enjoying a cheerful barbecue on a summer afternoon",
    "A picture of a scenic mountain view inspiring awe and serenity",
    "A picture of a child lost in wonder while reading a magical book",
    "A picture of a group of travelers exploring a charming old town",
    "A picture of a playful cat basking in a warm patch of sunlight",
    "A picture of a serene river flowing through a lush green valley",
    "A picture of a couple enjoying a quiet moment on a lakeside bench",
    "A picture of a family building a sandcastle on a sunny beach",
    "A picture of a person savoring a peaceful moment with a cup of tea",
    "A picture of a lively parade with colorful floats and joyful spectators",
    "A picture of children laughing while running through a sprinkler on a hot day",
    "A picture of a couple sharing intimate conversation in a cozy café",
    "A picture of an outdoor art fair with people admiring creative works",
    "A picture of a friendly market bustling with vibrant colors and aromas",
    "A picture of a scenic bike ride along a country road in full bloom",
    "A picture of a person enjoying a calm moment of reflection in nature",
    "A picture of a group of dancers performing with energy and grace",
    "A picture of a peaceful snowy day with friends building a cheerful snowman",
    "A picture of a joyful family reunion filled with laughter and love",
]

# List of text descriptions intended to evoke neutral emotions (Using the refined list)
experimental_conditions_neutral = [
    "A picture of a pencil resting on a desk in a quiet room",
    "A picture of a window showing a view of a busy street outside",
    "A picture of a door leading into a simple, well-lit hallway",
    "A picture of a single lamp glowing softly in a dim room",
    "A picture of a plain road stretching under an overcast sky",
    "A picture of a parked car waiting in a supermarket parking lot",
    "A picture of a clock with a black frame on a plain wall",
    "A picture of a stack of papers neatly arranged on a desk",
    "A picture of a clean sidewalk running through an urban neighborhood",
    "A picture of a concrete building standing tall against the sky",
    "A picture of a metal chair placed against a blank wall",
    "A picture of a wooden table resting under a soft overhead light",
    "A picture of a simple bookshelf filled with semi organized books",
    "A picture of a group of plastic chairs gathered around a small table",
    "A picture of a quiet park with paths winding through fir trees",
    "A picture of a public bench resting in a shaded open space",
    "A picture of a standard street sign marking a city intersection",
    "A picture of a parked bicycle secured to a bike rack on the sidewalk",
    "A picture of a single tree standing tall in an open field",
    "A picture of a city skyline with tall buildings stretching skyward",
    "A picture of a small bridge spanning a calm river under clear skies",
    "A picture of a rural house surrounded by open fields and green space",
    "A picture of a calm, quiet street lined with houses on both sides",
    "A picture of a minimalistic office with cubicles and simple furnishings",
    "A picture of a modern building with sleek, reflective windows",
    "A picture of a plain computer screen glowing in a dark room",
    "A picture of a generic smartphone sitting in a store display case",
    "A picture of a basic mobile phone placed on a kitchen counter",
    "A picture of a sealed envelope resting on a desk",
    "A picture of a stack of books arranged neatly on a wooden shelf",
    "A picture of a collection of coins in a mason jar",
    "A picture of a group of notebooks stacked together on a desk",
    "A picture of a metal spoon resting on a table beside a bowl",
    "A picture of a wooden bench sitting under a large, leafy tree",
    "A picture of a standard staircase leading upward with white railings",
    "A picture of a simple lamp casting soft light next to an armchair",
    "A picture of a vacant lot with scattered weeds and tall grasses",
    "A picture of a minimalistic bedroom with simple furniture and soft decor",
    "A picture of a plain wall painted in a neutral, calming color",
    "A picture of a quiet library filled with rows of books and soft lighting",
]

# List of text descriptions intended to evoke unpleasant emotions
experimental_conditions_unpleasant = [
    "A picture of a swarm of bloodthirsty bats descending from the sky",
    "A picture of a decaying graveyard with tilted tombstones",
    "A picture of a mutilated doll with cracked porcelain skin",
    "A picture of a haunted mansion with flickering candlelight",
    "A picture of a deranged clown grinning in a dark alley",
    "A picture of a toxic waste spill spreading through a barren field",
    "A picture of a rotting corpse half-submerged in murky water",
    "A picture of a venomous snake slithering across your path",
    "A picture of a dilapidated asylum with broken windows",
    "A picture of a monstrous creature lurking in a shadowy forest",
    "A picture of a gruesome crime scene with scattered evidence",
    "A picture of a sinister figure shrouded in a tattered cloak",
    "A picture of a putrid carcass lying in a puddle of slime",
    "A picture of a crumbling bridge engulfed in fog",
    "A picture of a storm of acid rain corroding abandoned ruins",
    "A picture of a shadowy ghost drifting through a ruined hall",
    "A picture of a bloodstained weapon discarded on the floor",
    "A picture of a creepy attic filled with cobwebs and decay",
    "A picture of a wild animal with feral eyes in an urban wasteland",
    "A picture of a derelict amusement park with twisted metal",
    "A picture of a malignant tumor spreading across a dead landscape",
    "A picture of a nightmarish creature with glowing red eyes",
    "A picture of a sinister ritual conducted under a full moon",
    "A picture of a broken mirror reflecting a distorted face",
    "A picture of a contaminated river filled with toxic sludge",
    "A picture of a raging fire consuming an abandoned building",
    "A picture of a savage animal growling near a shattered fence",
    "A picture of a cursed relic exuding an aura of malevolence",
    "A picture of a deteriorated hospital ward with eerie silence",
    "A picture of a violent storm shattering a glass façade",
    "A picture of a ghostly apparition hovering above the ground",
    "A picture of a deranged surgeon wielding bloodstained tools",
    "A picture of a festering wound oozing dark pus",
    "A picture of a corrupted forest with twisted, barren trees",
    "A picture of a polluted cityscape under a sickly sky",
    "A picture of a mutilated face emerging from the darkness",
    "A picture of a sinister labyrinth filled with endless corridors",
    "A picture of a decaying industrial site with leaking chemicals",
    "A picture of a deranged criminal leaving a trail of blood",
    "A picture of a dangerous dog chasing you through a desolate street",
]

# =============================================================================
# Process Experimental Conditions (Text -> EEG Generation)
# =============================================================================
print("\n--- Generating EEG data from text descriptions ---")

# --- Setup for Processing Loop ---
# Define the device for CLIP processing (can be different from loading device)
# Using cuda:1 if available, otherwise cpu, as per the original script's intention here.
device_process = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Device used for CLIP encoding and inference loop: {device_process}")

# Load CLIP model onto the processing device
# Note: Previous script loaded CLIP earlier. Reloading here onto potentially different device.
model_CLIP, preprocess = clip.load("ViT-B/32", device=device_process)
model_CLIP.eval() # Ensure CLIP is in eval mode

# Move other models to the processing device if needed.
# Currently, EEG_CVAE and LATENT_XD are on CPU ('device_load').
# The processing loop needs them on 'device_process' if it's a GPU.
# Decision: Keep CVAE/LATENT_XD on CPU as loaded, move intermediate tensors only.
# This avoids moving large models if device_process=GPU and device_load=CPU.
# If device_process is also CPU, no change needed.
model_EEG_CVAE.to(device_load) # Ensure they remain on the load device (CPU)
model_LATENT_XD.to(device_load)

# Initialize result arrays
num_pleasant = len(experimental_conditions_pleasant)
num_neutral = len(experimental_conditions_neutral)
num_unpleasant = len(experimental_conditions_unpleasant)
num_subjects = len(SUBJECTS_IDS)
num_electrodes = 129 # From CVAE output cropping
num_timepoints = 750

condP = np.zeros((num_pleasant, num_electrodes, num_timepoints, num_subjects))
condP_latent = np.zeros((num_pleasant, latent_dim, num_subjects))

condN = np.zeros((num_neutral, num_electrodes, num_timepoints, num_subjects))
condN_latent = np.zeros((num_neutral, latent_dim, num_subjects)) # Corrected size

condU = np.zeros((num_unpleasant, num_electrodes, num_timepoints, num_subjects))
condU_latent = np.zeros((num_unpleasant, latent_dim, num_subjects)) # Corrected size

# Define output paths
output_dir = './'
os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists


# --- Processing Loop for Pleasant Texts ---
print("\nProcessing Pleasant conditions...")
start_time_p = time.time()
for sub_idx in range(num_subjects):
    subject_id = SUBJECTS_IDS[sub_idx]
    print(f"  Pleasant - Subject {subject_id} ({sub_idx + 1}/{num_subjects})")
    # Create condition tensor for the subject (on CPU initially)
    conditionTest = np.zeros((1, num_subjects))
    conditionTest[0, sub_idx] = 1
    conditionTest_tensor = torch.tensor(conditionTest, dtype=torch.float32).to(device_load) # Keep on CPU for LATENT_XD/CVAE

    for txt_idx, text in enumerate(experimental_conditions_pleasant):
        # Tokenize text and move to processing device
        text_input = clip.tokenize([text]).to(device_process)

        with torch.no_grad(): # Disable gradients for inference
            # 1. Encode text with CLIP (on device_process)
            text_features = model_CLIP.encode_text(text_input)

            # 2. Map CLIP features -> latent space using LATENT_XD (on CPU)
            # Move text_features to CPU for LATENT_XD
            text_features_cpu = text_features.float().to(device_load)
            text_vae_latent_z = model_LATENT_XD(text_features_cpu, conditionTest_tensor) # Output is on CPU

            # 3. Decode latent z -> EEG using CVAE (on CPU)
            text_EEG_recon = model_EEG_CVAE.decode(text_vae_latent_z, conditionTest_tensor) # Output is on CPU

        # Store results (already on CPU)
        text_EEG_np = text_EEG_recon.detach().numpy().squeeze()
        condP[txt_idx, :, :, sub_idx] = text_EEG_np
        condP_latent[txt_idx, :, sub_idx] = text_vae_latent_z.detach().numpy()

# Save Pleasant results
save_path_p = os.path.join(output_dir, 'RESULTS_condP_ann_raw.mat')
print(f"Saving Pleasant results to {save_path_p}")
sio.savemat(save_path_p, {'condP': condP, 'condP_latent': condP_latent})
end_time_p = time.time()
print(f"Pleasant conditions processed in {end_time_p - start_time_p:.2f} seconds.")


# --- Processing Loop for Neutral Texts ---
print("\nProcessing Neutral conditions...")
start_time_n = time.time()
for sub_idx in range(num_subjects):
    subject_id = SUBJECTS_IDS[sub_idx]
    print(f"  Neutral - Subject {subject_id} ({sub_idx + 1}/{num_subjects})")
    conditionTest = np.zeros((1, num_subjects))
    conditionTest[0, sub_idx] = 1
    conditionTest_tensor = torch.tensor(conditionTest, dtype=torch.float32).to(device_load)

    for txt_idx, text in enumerate(experimental_conditions_neutral):
        text_input = clip.tokenize([text]).to(device_process)
        with torch.no_grad():
            text_features = model_CLIP.encode_text(text_input)
            text_features_cpu = text_features.float().to(device_load)
            # CORRECTION: Use the correct model variable names as loaded earlier
            text_vae_latent_z = model_LATENT_XD(text_features_cpu, conditionTest_tensor)
            text_EEG_recon = model_EEG_CVAE.decode(text_vae_latent_z, conditionTest_tensor)

        text_EEG_np = text_EEG_recon.detach().numpy().squeeze()
        condN[txt_idx, :, :, sub_idx] = text_EEG_np
        condN_latent[txt_idx, :, sub_idx] = text_vae_latent_z.detach().numpy()

# Save Neutral results
save_path_n = os.path.join(output_dir, 'RESULTS_condN_ann_raw.mat')
print(f"Saving Neutral results to {save_path_n}")
sio.savemat(save_path_n, {'condN': condN, 'condN_latent': condN_latent})
end_time_n = time.time()
print(f"Neutral conditions processed in {end_time_n - start_time_n:.2f} seconds.")


# --- Processing Loop for Unpleasant Texts ---
print("\nProcessing Unpleasant conditions...")
start_time_u = time.time()
for sub_idx in range(num_subjects):
    subject_id = SUBJECTS_IDS[sub_idx]
    print(f"  Unpleasant - Subject {subject_id} ({sub_idx + 1}/{num_subjects})")
    conditionTest = np.zeros((1, num_subjects))
    conditionTest[0, sub_idx] = 1
    conditionTest_tensor = torch.tensor(conditionTest, dtype=torch.float32).to(device_load)

    for txt_idx, text in enumerate(experimental_conditions_unpleasant):
        text_input = clip.tokenize([text]).to(device_process)
        with torch.no_grad():
            text_features = model_CLIP.encode_text(text_input)
            text_features_cpu = text_features.float().to(device_load)
            # CORRECTION: Use the correct model variable names as loaded earlier
            text_vae_latent_z = model_LATENT_XD(text_features_cpu, conditionTest_tensor)
            text_EEG_recon = model_EEG_CVAE.decode(text_vae_latent_z, conditionTest_tensor)

        text_EEG_np = text_EEG_recon.detach().numpy().squeeze()
        condU[txt_idx, :, :, sub_idx] = text_EEG_np
        condU_latent[txt_idx, :, sub_idx] = text_vae_latent_z.detach().numpy()

# Save Unpleasant results
save_path_u = os.path.join(output_dir, 'RESULTS_condU_ann_raw.mat')
print(f"Saving Unpleasant results to {save_path_u}")
sio.savemat(save_path_u, {'condU': condU, 'condU_latent': condU_latent})
end_time_u = time.time()
print(f"Unpleasant conditions processed in {end_time_u - start_time_u:.2f} seconds.")


# =============================================================================
# Visualization and Analysis
# =============================================================================
print("\n--- Plotting results ---")

# --- Plot average ERP for electrode 61 across all subjects ---
# Note: Electrode indices are 0-based, so 61 refers to the 62nd electrode.
electrode_idx_plot = 61
if 0 <= electrode_idx_plot < num_electrodes:
    plt.figure(figsize=(10, 6))
    # Average across texts (axis 0) and subjects (axis 3)
    plt.plot(condP[:, electrode_idx_plot, :, :].mean(axis=(0, 2)), label='Pleasant', color='blue')
    plt.plot(condN[:, electrode_idx_plot, :, :].mean(axis=(0, 2)), label='Neutral', color='black')
    plt.plot(condU[:, electrode_idx_plot, :, :].mean(axis=(0, 2)), label='Unpleasant', color='red')
    plt.title(f'Average Generated ERP at Electrode {electrode_idx_plot + 1} (All Subjects)')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'average_erp_electrode_61.png'))
    # plt.show()
else:
    print(f"Warning: Electrode index {electrode_idx_plot} is out of range for plotting.")

# --- Plot average ERP for electrode 62 across all subjects ---
# Note: Electrode indices are 0-based, so 62 refers to the 63rd electrode.
electrode_idx_plot_alt = 62
if 0 <= electrode_idx_plot_alt < num_electrodes:
    plt.figure(figsize=(10, 6))
    plt.plot(condP[:, electrode_idx_plot_alt, :, :].mean(axis=(0, 2)), label='Pleasant', color='blue')
    plt.plot(condN[:, electrode_idx_plot_alt, :, :].mean(axis=(0, 2)), label='Neutral', color='black')
    plt.plot(condU[:, electrode_idx_plot_alt, :, :].mean(axis=(0, 2)), label='Unpleasant', color='red')
    plt.title(f'Average Generated ERP at Electrode {electrode_idx_plot_alt + 1} (All Subjects)')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'average_erp_electrode_62.png'))
    # plt.show()
else:
    print(f"Warning: Electrode index {electrode_idx_plot_alt} is out of range for plotting.")


# --- Plot average topographical maps (EEG images) across all subjects and texts ---
plt.figure(figsize=(18, 5))
# Average across texts (axis 0) and subjects (axis 3)
mean_condP_img = condP.mean(axis=(0, 3))
mean_condN_img = condN.mean(axis=(0, 3))
mean_condU_img = condU.mean(axis=(0, 3))

# Determine shared color limits
clim_val = np.max(np.abs([mean_condP_img, mean_condN_img, mean_condU_img])) * 0.8 # Use 80% of max abs value for viz
clim = (-clim_val, clim_val) if clim_val > 0 else (-1, 1) # Avoid clim=(0,0)

plt.subplot(1, 3, 1)
plt.imshow(mean_condP_img, aspect='auto', cmap='viridis', vmin=clim[0], vmax=clim[1])
plt.colorbar()
plt.title('Pleasant (Avg All)')
plt.xlabel('Time Points')
plt.ylabel('Electrodes')

plt.subplot(1, 3, 2)
plt.imshow(mean_condN_img, aspect='auto', cmap='viridis', vmin=clim[0], vmax=clim[1])
plt.colorbar()
plt.title('Neutral (Avg All)')
plt.xlabel('Time Points')
plt.ylabel('Electrodes')

plt.subplot(1, 3, 3)
plt.imshow(mean_condU_img, aspect='auto', cmap='viridis', vmin=clim[0], vmax=clim[1])
plt.colorbar()
plt.title('Unpleasant (Avg All)')
plt.xlabel('Time Points')
plt.ylabel('Electrodes')

plt.suptitle('Average Generated EEG Across All Texts and Subjects')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, 'average_eeg_images.png'))
# plt.show()


# --- Plot average ERP for electrode 61 with Standard Error (All Subjects) ---
# Average across texts (axis 0), calculate mean and SEM across subjects (axis 2)
mean_condP_e61 = condP[:, electrode_idx_plot, :, :].mean(axis=0).mean(axis=1)
std_err_condP_e61 = condP[:, electrode_idx_plot, :, :].mean(axis=0).std(axis=1) / np.sqrt(num_subjects)

mean_condN_e61 = condN[:, electrode_idx_plot, :, :].mean(axis=0).mean(axis=1)
std_err_condN_e61 = condN[:, electrode_idx_plot, :, :].mean(axis=0).std(axis=1) / np.sqrt(num_subjects)

mean_condU_e61 = condU[:, electrode_idx_plot, :, :].mean(axis=0).mean(axis=1)
std_err_condU_e61 = condU[:, electrode_idx_plot, :, :].mean(axis=0).std(axis=1) / np.sqrt(num_subjects)

# Assuming time axis from -0.1 to 1.4 seconds for 750 points
time_axis = np.linspace(-0.1, 1.4, num_timepoints)

plt.figure(figsize=(12, 7))
# Plot Pleasant
plt.fill_between(time_axis, mean_condP_e61 - std_err_condP_e61, mean_condP_e61 + std_err_condP_e61, color='blue', alpha=0.3)
plt.plot(time_axis, mean_condP_e61, color='blue', label='Pleasant')
# Plot Neutral
plt.fill_between(time_axis, mean_condN_e61 - std_err_condN_e61, mean_condN_e61 + std_err_condN_e61, color='black', alpha=0.3)
plt.plot(time_axis, mean_condN_e61, color='black', label='Neutral')
# Plot Unpleasant
plt.fill_between(time_axis, mean_condU_e61 - std_err_condU_e61, mean_condU_e61 + std_err_condU_e61, color='red', alpha=0.3)
plt.plot(time_axis, mean_condU_e61, color='red', label='Unpleasant')

plt.title(f'Mean ERP +/- SEM at Electrode {electrode_idx_plot + 1} (Averaged Across Subjects)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'average_erp_electrode_61_sem.png'))
# plt.show()


# --- Plot average ERP for electrode 61 with Standard Error (Single Subject) ---
subject_plot_idx = 50 # Index (0-based) of the subject to plot
if 0 <= subject_plot_idx < num_subjects:
    subject_plot_id = SUBJECTS_IDS[subject_plot_idx]
    # Average across texts (axis 0), calculate mean and SEM for the selected subject
    mean_condP_s = condP[:, electrode_idx_plot, :, subject_plot_idx].mean(axis=0)
    std_err_condP_s = condP[:, electrode_idx_plot, :, subject_plot_idx].std(axis=0) / np.sqrt(num_pleasant)

    mean_condN_s = condN[:, electrode_idx_plot, :, subject_plot_idx].mean(axis=0)
    std_err_condN_s = condN[:, electrode_idx_plot, :, subject_plot_idx].std(axis=0) / np.sqrt(num_neutral)

    mean_condU_s = condU[:, electrode_idx_plot, :, subject_plot_idx].mean(axis=0)
    std_err_condU_s = condU[:, electrode_idx_plot, :, subject_plot_idx].std(axis=0) / np.sqrt(num_unpleasant)

    plt.figure(figsize=(12, 7))
    plt.fill_between(time_axis, mean_condP_s - std_err_condP_s, mean_condP_s + std_err_condP_s, color='blue', alpha=0.3)
    plt.plot(time_axis, mean_condP_s, color='blue', label='Pleasant')
    plt.fill_between(time_axis, mean_condN_s - std_err_condN_s, mean_condN_s + std_err_condN_s, color='black', alpha=0.3)
    plt.plot(time_axis, mean_condN_s, color='black', label='Neutral')
    plt.fill_between(time_axis, mean_condU_s - std_err_condU_s, mean_condU_s + std_err_condU_s, color='red', alpha=0.3)
    plt.plot(time_axis, mean_condU_s, color='red', label='Unpleasant')

    plt.title(f'Mean ERP +/- SEM at Electrode {electrode_idx_plot + 1} (Subject {subject_plot_id})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'single_subject_{subject_plot_id}_erp_electrode_61_sem.png'))
    # plt.show()
else:
    print(f"Warning: Subject index {subject_plot_idx} is out of range for plotting.")

# --- Plot average of channels 60 to 90 over time ---
# Assuming indices 60:90 means electrodes 61 through 90 (0-based index 60 to 89)
start_ch = 60
end_ch = 90 # Exclusive index
if 0 <= start_ch < end_ch <= num_electrodes:
    # Average across texts(0), channels(1), subjects(3)
    mean_condP_channels = condP[:, start_ch:end_ch, :, :].mean(axis=(0, 1, 3))
    mean_condN_channels = condN[:, start_ch:end_ch, :, :].mean(axis=(0, 1, 3))
    mean_condU_channels = condU[:, start_ch:end_ch, :, :].mean(axis=(0, 1, 3))

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, mean_condP_channels, color='blue', label='Pleasant')
    plt.plot(time_axis, mean_condN_channels, color='black', label='Neutral')
    plt.plot(time_axis, mean_condU_channels, color='red', label='Unpleasant')
    plt.title(f'Average ERP Over Time (Channels {start_ch+1}-{end_ch})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'average_erp_channels_60_90.png'))
    # plt.show()
else:
    print(f"Warning: Channel range [{start_ch}:{end_ch}] is invalid for plotting.")

# --- T-SNE visualization of the latent space (all subjects combined) ---
print("\nPerforming T-SNE on the combined latent space...")
# Reshape latent arrays: (n_texts, latent_dim, n_subjects) -> (n_texts * n_subjects, latent_dim)
latent_P_flat = condP_latent.transpose(0, 2, 1).reshape(-1, latent_dim)
latent_N_flat = condN_latent.transpose(0, 2, 1).reshape(-1, latent_dim)
latent_U_flat = condU_latent.transpose(0, 2, 1).reshape(-1, latent_dim)

# Concatenate the latent representations
condPNU_latent_flat = np.concatenate((latent_P_flat, latent_N_flat, latent_U_flat), axis=0)
# Create corresponding labels
labels = np.concatenate((
    np.full(latent_P_flat.shape[0], 'P'),
    np.full(latent_N_flat.shape[0], 'N'),
    np.full(latent_U_flat.shape[0], 'U')
))

# Perform T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000) # Standard params
start_tsne_all = time.time()
latent_2d = tsne.fit_transform(condPNU_latent_flat)
end_tsne_all = time.time()
print(f"T-SNE (all subjects) completed in {end_tsne_all - start_tsne_all:.2f} seconds.")

# Scatter plot
plt.figure(figsize=(10, 7))
colors = {'P': 'blue', 'N': 'black', 'U': 'red'}
for label_val in np.unique(labels):
    indices = labels == label_val
    plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], c=colors[label_val], label=label_val, alpha=0.3, s=10) # Smaller points
plt.legend()
plt.title('T-SNE Visualization of Generated Latent Space (All Subjects)')
plt.xlabel('T-SNE Dimension 1')
plt.ylabel('T-SNE Dimension 2')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'tsne_latent_space_all_subjects.png'))
# plt.show()

# --- T-SNE visualization for a single subject ---
subj_tsne_idx = 87 # Index (0-based) for the subject to visualize
if 0 <= subj_tsne_idx < num_subjects:
    subj_tsne_id = SUBJECTS_IDS[subj_tsne_idx]
    print(f"\nPerforming T-SNE for Subject {subj_tsne_id}...")
    # Extract latent representations for the selected subject
    latent_P_subj = condP_latent[:, :, subj_tsne_idx] # (n_texts, latent_dim)
    latent_N_subj = condN_latent[:, :, subj_tsne_idx]
    latent_U_subj = condU_latent[:, :, subj_tsne_idx]

    condPNU_latent_subject = np.concatenate((latent_P_subj, latent_N_subj, latent_U_subj), axis=0)
    labels_subject = np.concatenate((
        np.full(latent_P_subj.shape[0], 'P'),
        np.full(latent_N_subj.shape[0], 'N'),
        np.full(latent_U_subj.shape[0], 'U')
    ))

    # Perform T-SNE
    tsne_subject = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    start_tsne_single = time.time()
    latent_2d_subject = tsne_subject.fit_transform(condPNU_latent_subject)
    end_tsne_single = time.time()
    print(f"T-SNE (Subject {subj_tsne_id}) completed in {end_tsne_single - start_tsne_single:.2f} seconds.")


    # Scatter plot
    plt.figure(figsize=(10, 7))
    for label_val in np.unique(labels_subject):
        indices = labels_subject == label_val
        plt.scatter(latent_2d_subject[indices, 0], latent_2d_subject[indices, 1], c=colors[label_val], label=label_val, alpha=0.6)
    plt.legend()
    plt.title(f'T-SNE Visualization of Generated Latent Space (Subject {subj_tsne_id})')
    plt.xlabel('T-SNE Dimension 1')
    plt.ylabel('T-SNE Dimension 2')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'tsne_latent_space_subject_{subj_tsne_id}.png'))
    # plt.show()
else:
    print(f"Warning: Subject index {subj_tsne_idx} for T-SNE is out of range.")

print("\nScript finished.")