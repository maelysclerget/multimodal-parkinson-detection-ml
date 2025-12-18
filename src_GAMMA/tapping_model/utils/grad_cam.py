"""
Grad-CAM visualization for CNN2D_Heatmap classifier on tapping heatmaps

Generates a 3-panel figure showing:
1. Original heatmap image
2. Grad-CAM importance map
3. Overlay of Grad-CAM on original

Input:
    - CHECKPOINT_PATH : .pth file from CNN heatmap training
    - HEATMAP_PATH    : path to a heatmap PNG (224x224 grayscale)
    - LABEL_CSV       : CSV with columns ['healthCode', 'label_PD']

Output:
    - 3-panel Grad-CAM figure with predictions and true labels
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


# ===== Define CNN2D_Heatmap Model =====
class CNN2D_Heatmap(nn.Module):
    """
    2D Convolutional Neural Network for heatmap classification.
    
    Architecture:
    - Conv Block 1: 1 → num_channels channels, 224×224 → 112×112
    - Conv Block 2: num_channels → 2×num_channels, 112×112 → 56×56
    - Conv Block 3: 2×num_channels → 4×num_channels, 56×56 → 28×28
    - Global Average Pooling
    - FC: 4×num_channels → 128 → 64 → 2 (binary classification)
    
    Each conv block includes: Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d
    
    Args:
        num_channels (int): Base number of channels in first conv layer (default: 32)
        dropout_rate (float): Dropout rate for convolutional layers (default: 0.5)
    
    Input shape: (batch_size, 1, 224, 224) - grayscale images
    Output shape: (batch_size, 2) - logits for binary classification
    """
    def __init__(self, num_channels=32, dropout_rate=0.5):
        super(CNN2D_Heatmap, self).__init__()
        
        # Input: (batch, 1, 224, 224)
        # Conv block 1
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels * 4)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels * 4, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 2)
        
        # Store intermediate activations for Grad-CAM
        self.target_layer = None
    
    def forward(self, x):
        """
        Forward pass through CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Store activations from pool3 for Grad-CAM
        self.target_layer = x
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        return x


# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------
def load_model(checkpoint_path, device):
    """Load CNN2D_Heatmap model from checkpoint."""
    model = CNN2D_Heatmap(num_channels=32, dropout_rate=0.5)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# 2. Preprocess heatmap image for CNN2D_Heatmap
# ---------------------------------------------------------
def preprocess_image_for_model(img_path):
    """
    Preprocess heatmap image for CNN2D_Heatmap.
    
    Converts to grayscale, resizes to 224x224, normalizes to [-1, 1].
    
    Args:
        img_path (str): Path to heatmap PNG file
    
    Returns:
        torch.Tensor: Image tensor of shape (1, 1, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    image = Image.open(img_path).convert('L')  # Grayscale
    tensor = transform(image).unsqueeze(0)  # (1, 1, 224, 224)
    return tensor


# ---------------------------------------------------------
# 3. Grad-CAM on CNN2D_Heatmap (using pool3 activations)
# ---------------------------------------------------------
def compute_gradcam(model, input_tensor, target_class, device):
    """
    Compute Grad-CAM for CNN2D_Heatmap using pool3 layer activations.
    
    Grad-CAM visualizes which regions of the input contributed most to the
    predicted class.
    
    Args:
        model (CNN2D_Heatmap): Model in eval mode
        input_tensor (torch.Tensor): Input image tensor (1, 1, 224, 224)
        target_class (int): Target class (0=Healthy, 1=PD)
        device (torch.device): Device to compute on
    
    Returns:
        tuple: (cam_np, probs_np)
            - cam_np (np.ndarray): Grad-CAM heatmap (28x28), values in [0, 1]
            - probs_np (np.ndarray): Softmax probabilities for both classes
    """
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        """Store activations during forward pass."""
        activations["value"] = out.detach()
        out.retain_grad()

    def backward_hook(module, grad_inp, grad_out):
        """Store gradients during backward pass."""
        gradients["value"] = grad_out[0].detach()

    # Register hooks on pool3 layer (after 3rd conv block)
    handle_fwd = model.pool3.register_forward_hook(forward_hook)
    handle_bwd = model.pool3.register_full_backward_hook(backward_hook)

    # Enable gradients for backward pass
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    model.zero_grad()
    output = model(input_tensor)  # (1, 2)
    probs = torch.softmax(output, dim=1)[0]
    score = output[0, target_class]

    # Backward pass
    score.backward()

    # Retrieve activations and gradients
    act = activations["value"][0]  # (C, H, W) where H=W=28 for pool3
    grad = gradients["value"][0]   # (C, H, W)

    # Cleanup hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # Compute Grad-CAM: weighted sum of activations
    weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
    cam = (weights * act).sum(dim=0)               # (H, W)
    cam = torch.relu(cam)
    
    # Normalize to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    return cam.cpu().numpy(), probs.detach().cpu().numpy()


# ---------------------------------------------------------
# 4. 3-panel plotting: original, Grad-CAM, overlay
# ---------------------------------------------------------
def plot_triplet(heatmap_path, cam, healthcode, pred, true, prob,
                 out_path="gradcam_triplet.png"):
    """
    Create 3-panel figure: original heatmap, Grad-CAM, overlay.
    
    Args:
        heatmap_path (str): Path to original heatmap PNG
        cam (np.ndarray): Grad-CAM heatmap (28x28), values in [0, 1]
        healthcode (str): Patient healthCode ID
        pred (int): Predicted class (0=Healthy, 1=PD)
        true (int): True class label
        prob (float): Predicted probability for predicted class
        out_path (str): Output file path for figure
    """
    label_map = {0: "HC", 1: "PD"}

    # Load grayscale heatmap
    orig = Image.open(heatmap_path).convert('L')
    orig_np = np.array(orig, dtype=np.float32) / 255.0  # [0, 1]
    
    H, W = orig_np.shape
    
    # Resize CAM to match image size (28x28 -> HxW)
    cam_resized = cv2.resize(cam, (W, H))

    # Create heatmap overlay
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0
    
    alpha = 0.45
    overlay = (1 - alpha) * orig_np[..., np.newaxis] + alpha * heatmap_rgb

    # Prepare plot
    x0, x1 = 0, W
    y0, y1 = 0, H
    extent = [x0, x1, y0, y1]  # left, right, bottom, top
    origin = "lower"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plt.subplots_adjust(top=0.78, wspace=0.25)

    # Panel 1: Original heatmap
    axes[0].imshow(orig_np, cmap='gray', aspect="auto", origin=origin, extent=extent)
    axes[0].set_title("Original Heatmap", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    # Panel 2: Grad-CAM
    im1 = axes[1].imshow(cam_resized, cmap="jet", aspect="auto", origin=origin, extent=extent)
    axes[1].set_title("Grad-CAM Importance", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("X (pixels)")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Importance (0–1)")

    # Panel 3: Overlay
    axes[2].imshow(overlay, aspect="auto", origin=origin, extent=extent)
    axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("X (pixels)")

    # Title with prediction info
    fig.suptitle(
        f"HealthCode: {healthcode} | Pred: {label_map[pred]} (p={prob:.3f}) | True: {label_map[true]}",
        fontsize=13, fontweight='bold'
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved 3-panel Grad-CAM figure to: {out_path}")


# ---------------------------------------------------------
# 5. Main: glue everything together
# ---------------------------------------------------------
def main():
    """Main execution for Grad-CAM visualization on CNN2D_Heatmap."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    HEATMAP_PATH    = "/mloscratch/users/clerget/data/tapping_heatmaps/2b1f8af1-c928-4ca5-a240-fdd3ef6ead98/01_b25a3d62-c3ed-40af-b48b-5fab19819811_heatmap.png"
    LABEL_CSV       = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
    CHECKPOINT_PATH = "/mloscratch/users/clerget/data/saved_models/best_model_cnn_heatmap_fold0_class_weights.pth"

    # Create output directory
    output_dir = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results/Grad-CAM"
    os.makedirs(output_dir, exist_ok=True)

    # [1] Load model
    print("\n[1/5] Loading CNN2D_Heatmap model...")
    model = load_model(CHECKPOINT_PATH, device)
    print(f"✓ Model loaded from: {CHECKPOINT_PATH}")

    # [2] Get healthCode and true label
    print("\n[2/5] Loading labels...")
    fname = os.path.basename(HEATMAP_PATH)
    # Extract healthCode from path: .../healthCode/trial_id.png
    healthcode = os.path.basename(os.path.dirname(HEATMAP_PATH))
    
    label_df = pd.read_csv(LABEL_CSV)
    row = label_df[label_df["healthCode"] == healthcode]
    
    if row.empty:
        raise ValueError(f"HealthCode {healthcode} not found in {LABEL_CSV}")
    
    true_label = int(row.iloc[0]["label_PD"])
    print(f"✓ HealthCode: {healthcode}, True label: {true_label}")

    # [3] Prepare image tensor
    print("\n[3/5] Preprocessing heatmap image...")
    img_tensor = preprocess_image_for_model(HEATMAP_PATH)
    print(f"✓ Image shape: {img_tensor.shape}")

    # [4] Get prediction
    print("\n[4/5] Computing model prediction...")
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    
    pred_class = int(probs.argmax())
    prob_pred = float(probs[pred_class])
    print(f"✓ Prediction: class {pred_class}, probability {prob_pred:.4f}")

    # [5] Compute Grad-CAM
    print("\n[5/5] Computing Grad-CAM...")
    cam, _ = compute_gradcam(model, img_tensor, pred_class, device)
    print(f"✓ Grad-CAM computed (shape: {cam.shape})")

    # Plot triple figure
    output_path = os.path.join(output_dir, f"gradcam_triplet_{healthcode}.png")
    plot_triplet(
        heatmap_path=HEATMAP_PATH,
        cam=cam,
        healthcode=healthcode,
        pred=pred_class,
        true=true_label,
        prob=prob_pred,
        out_path=output_path
    )

    print("\n" + "="*80)
    print("✓ Grad-CAM visualization complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()