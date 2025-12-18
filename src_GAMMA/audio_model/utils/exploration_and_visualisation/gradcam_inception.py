"""
3-Panel Grad-CAM Visualization for InceptionV3 Mel-Spectrogram Classifier

This script generates a 3-panel Grad-CAM figure for a trained InceptionV3 model on mel-spectrogram audio classification. It visualizes the original mel-spectrogram, the Grad-CAM heatmap, and an overlay, helping to interpret which regions of the input most influenced the model's prediction.

Context:
--------
Grad-CAM is a popular technique for visualizing the spatial regions in an input image that contribute most to a deep learning model's decision. In this project, it is used to interpret the InceptionV3 classifier's predictions on mel-spectrograms, providing insight into what acoustic patterns the model relies on for Parkinson's detection.

Inputs:
    - CHECKPOINT_PATH : .pth file from your InceptionV3 training
    - MELSPEC_PATH    : path to a mel spectrogram .jpg (exact same format as training)
    - LABEL_CSV       : CSV with columns ['healthCode', 'label_PD']

Outputs:
    - triplet figure: original mel, Grad-CAM, overlay (PNG)

Usage:
------
Set the paths in the main() function and run as a script. The output figure will be saved to disk. This script is useful for model interpretability, reporting, and qualitative analysis of deep learning results.
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

# import your model
from inceptionv3_melspec_5fold import InceptionV3Classifier

# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------
def load_model(checkpoint_path, device):
    model = InceptionV3Classifier()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

# ---------------------------------------------------------
# 2. Preprocess mel-spectrogram image for InceptionV3
# ---------------------------------------------------------
def preprocess_image_for_model(img_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1,3,299,299)
    
    return tensor


# ---------------------------------------------------------
# 3. Grad-CAM on InceptionV3 backbone (Mixed_7c)
# ---------------------------------------------------------
def compute_gradcam(model, input_tensor, target_class, device):
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out
        out.retain_grad()  # ensure .grad will be populated

    target_layer = model.backbone.Mixed_7c
    handle_fwd = target_layer.register_forward_hook(forward_hook)

    # ---- IMPORTANT: enable grad through backbone just for this Grad-CAM call ----
    was_training = model.training
    model.eval()

    # Temporarily enable grads for backbone (so Mixed_7c output requires grad)
    old_req = []
    for p in model.backbone.parameters():
        old_req.append(p.requires_grad)
        p.requires_grad_(True)

    input_tensor = input_tensor.to(device)

    model.zero_grad(set_to_none=True)
    output = model(input_tensor)  # (1,2)
    probs = torch.softmax(output, dim=1)[0]
    score = output[0, target_class]

    score.backward()

    # Collect gradients from the hooked activation tensor
    act = activations["value"][0].detach()          # (C,H,W)
    grad = activations["value"].grad[0].detach()    # (C,H,W)

    # Cleanup hook + restore requires_grad flags
    handle_fwd.remove()
    for p, r in zip(model.backbone.parameters(), old_req):
        p.requires_grad_(r)
    if was_training:
        model.train()

    # Grad-CAM core
    weights = grad.mean(dim=(1, 2), keepdim=True)   # (C,1,1)
    cam = (weights * act).sum(dim=0)                # (H,W)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam.cpu().numpy(), probs.detach().cpu().numpy()


# ---------------------------------------------------------
# 4. 3-panel plotting: original, Grad-CAM, overlay
# ---------------------------------------------------------
def plot_triplet(melspec_path, cam, healthcode, pred, true, prob,
                 out_path="gradcam_triplet.png"):
    """
    3-panel figure that is consistent with the *stored* mel-spec JPGs
    (i.e., we do NOT claim true Hz/seconds since we are not recomputing).

    Axes:
      - X: time frames / image columns (0..W)
      - Y: mel-bin rows / image rows (0..H)

    Panels:
      1) Original mel-spectrogram JPG
      2) Grad-CAM heatmap
      3) Overlay
    """
    label_map = {0: "HC", 1: "PD"}

    # --- load JPG as float RGB in [0,1] ---
    orig = plt.imread(melspec_path)
    if orig.dtype != np.float32 and orig.dtype != np.float64:
        orig = orig.astype(np.float32) / 255.0
    if orig.shape[-1] == 4:  # RGBA -> RGB
        orig = orig[..., :3]

    H, W, _ = orig.shape

    # --- resize cam to image size ---
    cam_resized = cv2.resize(cam, (W, H))

    # --- heatmap + overlay ---
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0
    alpha = 0.45
    overlay = (1 - alpha) * orig + alpha * heatmap

    # --- axis in image coordinates ---
    x0, x1 = 0, W
    y0, y1 = 0, H
    extent = [x0, x1, y0, y1]  # left, right, bottom, top

    # Displaying with origin="lower" keeps bottom as "low" row index in the displayed plot.
    origin = "lower"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plt.subplots_adjust(top=0.78, wspace=0.25)

    # Panel 1: original
    axes[0].imshow(orig, aspect="auto", origin=origin, extent=extent)
    axes[0].set_title("Original mel-spectrogram")
    axes[0].set_xlabel("Time (frames / pixels)")
    axes[0].set_ylabel("Mel bins (pixels)")

    # Panel 2: Grad-CAM
    im1 = axes[1].imshow(cam_resized, cmap="jet", aspect="auto", origin=origin, extent=extent)
    axes[1].set_title("Grad-CAM importance")
    axes[1].set_xlabel("Time (frames / pixels)")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM (0–1)")

    # Panel 3: overlay
    axes[2].imshow(overlay, aspect="auto", origin=origin, extent=extent)
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Time (frames / pixels)")

    fig.suptitle(
        f"HealthCode: {healthcode} | Pred: {label_map[pred]} (p={prob:.2f}) | True: {label_map[true]}",
        fontsize=12
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3-panel Grad-CAM figure to: {out_path}")


# ---------------------------------------------------------
# 5. Main: glue everything together
# ---------------------------------------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MELSPEC_PATH    = "/mloscratch/users/gnahas/data/melSpec/dc600bb6-ad90-42ee-99ee-d665ab1a62ba_448b233e-de57-440a-a515-3d9616c24260_audio_audio_m4a.jpg"
    LABEL_CSV       = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
    CHECKPOINT_PATH = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/InceptionV3_MelSpec/V1/PTH/fold_3_best_auc_0.6580_epoch_10.pth"

    # 1) Load model
    model = load_model(CHECKPOINT_PATH, device)

    # 2) Get healthCode & true label
    fname = os.path.basename(MELSPEC_PATH)
    healthcode = fname.split("_")[0]   # prefix convention from your training
    label_df = pd.read_csv(LABEL_CSV, sep=",")
    row = label_df[label_df["healthCode"] == healthcode]
    if row.empty:
        raise ValueError(f"HealthCode {healthcode} not found in {LABEL_CSV}")
    true_label = int(row.iloc[0]["label_PD"])
    
    OUTPUT_PATH     = f"/mloscratch/users/gnahas/data/Grad-CAM/gradcam_triplet_{healthcode}.png"


    # 3) Prepare image tensor
    img_tensor = preprocess_image_for_model(MELSPEC_PATH)

    # 4) Forward + Grad-CAM
    logits = model(img_tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = int(probs.argmax())
    prob_pred = float(probs[pred_class])

    # Now recompute with gradients for Grad-CAM (no torch.no_grad here)
    cam, _ = compute_gradcam(model, img_tensor, pred_class, device)

    # 5) Plot triple figure
    plot_triplet(
        melspec_path=MELSPEC_PATH,
        cam=cam,
        healthcode=healthcode,
        pred=pred_class,
        true=true_label,
        prob=prob_pred,
        out_path=OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
