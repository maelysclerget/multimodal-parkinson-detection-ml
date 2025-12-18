import os
import glob
import random
import numpy as np
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from inceptionv3_melspec_5fold import InceptionV3Classifier

# -----------------------------
# Model + preprocessing
# -----------------------------
def load_model(checkpoint_path, device):
    """
    Load the InceptionV3Classifier model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
        model (InceptionV3Classifier): Loaded and ready-to-evaluate model.
    """
    model = InceptionV3Classifier()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def preprocess_image_for_model(img_path):
    """
    Preprocess an image for input to the model.

    Args:
        img_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, 299, 299).
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1,3,299,299)

def compute_gradcam(model, input_tensor, target_class, device):
    """
    Compute Grad-CAM for a given input and model.

    Args:
        model (torch.nn.Module): The model to use for Grad-CAM.
        input_tensor (torch.Tensor): Preprocessed input image tensor.
        target_class (int): Target class index for which Grad-CAM is computed.
        device (torch.device): Device to run computations on.

    Returns:
        np.ndarray: Grad-CAM heatmap of shape (H, W), normalized to [0, 1].
    """
    activations = {}

    def forward_hook(module, inp, out):
        activations["value"] = out
        out.retain_grad()

    target_layer = model.backbone.Mixed_7c
    handle_fwd = target_layer.register_forward_hook(forward_hook)

    was_training = model.training
    model.eval()

    old_req = []
    for p in model.backbone.parameters():
        old_req.append(p.requires_grad)
        p.requires_grad_(True)

    x = input_tensor.to(device)

    model.zero_grad(set_to_none=True)
    output = model(x)  # (1,2)
    score = output[0, target_class]
    score.backward()

    act = activations["value"][0].detach()          # (C,H,W)
    grad = activations["value"].grad[0].detach()    # (C,H,W)

    handle_fwd.remove()
    for p, r in zip(model.backbone.parameters(), old_req):
        p.requires_grad_(r)
    if was_training:
        model.train()

    weights = grad.mean(dim=(1, 2), keepdim=True)   # (C,1,1)
    cam = (weights * act).sum(dim=0)                # (H,W)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam.cpu().numpy()

def make_overlay_rgb(melspec_path, cam, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on the original mel-spectrogram image.

    Args:
        melspec_path (str): Path to the mel-spectrogram image (JPG/PNG).
        cam (np.ndarray): Grad-CAM heatmap (H, W), normalized to [0, 1].
        alpha (float): Transparency factor for the overlay.

    Returns:
        np.ndarray: RGB image with Grad-CAM overlay, values in [0, 1].
    """
    orig = plt.imread(melspec_path)
    if orig.dtype != np.float32 and orig.dtype != np.float64:
        orig = orig.astype(np.float32) / 255.0
    if orig.shape[-1] == 4:
        orig = orig[..., :3]

    H, W, _ = orig.shape
    cam_resized = cv2.resize(cam, (W, H))

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * orig + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0)


# -----------------------------
# Data selection from split CSV
# -----------------------------
def sample_healthcodes(split_csv, fold_iteration, subset, n=25, seed=0):
    """
    Sample a list of healthCodes from a split CSV for a given fold and subset.

    Args:
        split_csv (str): Path to the split CSV file. Must contain columns: fold_iteration, healthCode, subset.
        fold_iteration (int): Fold number to filter.
        subset (str): Subset to filter (e.g., 'test', 'val').
        n (int): Number of healthCodes to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        list[str]: List of sampled healthCode strings.
    """
    df = pd.read_csv(split_csv)

    # normalize types
    df["fold_iteration"] = df["fold_iteration"].astype(int)
    df["healthCode"] = df["healthCode"].astype(str)
    df["subset"] = df["subset"].astype(str)

    df = df[(df["fold_iteration"] == int(fold_iteration)) & (df["subset"] == str(subset))]

    healthcodes = sorted(df["healthCode"].unique().tolist())
    if len(healthcodes) < n:
        raise ValueError(f"Only {len(healthcodes)} unique healthCodes in fold={fold_iteration}, subset={subset}")

    rng = random.Random(seed)
    rng.shuffle(healthcodes)
    return healthcodes[:n]


def pick_one_melspec_per_healthcode(melspec_root, healthcodes, seed=0, recursive=True):
    """
    Find one mel-spectrogram JPG per healthCode by globbing for {healthCode}_*.jpg.

    Args:
        melspec_root (str): Root directory containing mel-spectrogram images.
        healthcodes (list[str]): List of healthCode strings.
        seed (int): Random seed for reproducibility.
        recursive (bool): Whether to search recursively in subdirectories.

    Returns:
        list[tuple[str, str]]: List of (healthCode, image_path) pairs.
    """
    rng = random.Random(seed)
    chosen = []
    missing = []

    for hc in healthcodes:
        pattern = os.path.join(melspec_root, "**", f"{hc}_*.jpg") if recursive else os.path.join(melspec_root, f"{hc}_*.jpg")
        matches = glob.glob(pattern, recursive=recursive)
        if not matches:
            missing.append(hc)
            continue
        matches.sort()
        chosen.append((hc, rng.choice(matches)))  # random one recording per patient

    if missing:
        print(f"[WARN] No mel JPG found for {len(missing)} healthCodes (showing up to 10): {missing[:10]}")

    if len(chosen) < 8:
        raise RuntimeError(f"Only found {len(chosen)} melspecs total; need at least 8 to build the final figure.")

    return chosen


# -----------------------------
# Labels
# -----------------------------
def load_labels(label_csv, sep=","):
    """
    Load healthCode labels from a CSV file.

    Args:
        label_csv (str): Path to the label CSV file. Must contain columns: healthCode, label_PD.
        sep (str): CSV separator (default: ',').

    Returns:
        dict[str, int]: Mapping from healthCode to label (0 or 1).
    """
    df = pd.read_csv(label_csv, sep=sep)
    df["healthCode"] = df["healthCode"].astype(str)
    return {hc: int(lbl) for hc, lbl in zip(df["healthCode"], df["label_PD"])}


@torch.no_grad()
def predict_one(model, img_path, device):
    """
    Predict the class and probability for a single image.

    Args:
        model (torch.nn.Module): Model for prediction.
        img_path (str): Path to the image file.
        device (torch.device): Device to run prediction on.

    Returns:
        tuple[int, float]: (predicted class index, predicted class probability)
    """
    x = preprocess_image_for_model(img_path).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    p_pred = float(probs[pred])
    return pred, p_pred


def select_good_bad(records, n_good=4, n_bad=4):
    """
    Select the most confident examples for each confusion matrix type: TN, TP, FP, FN.

    Args:
        records (list[dict]): List of record dicts with keys: 'true', 'pred', 'p_pred', 'healthcode', 'path'.
        n_good (int): Unused, kept for compatibility.
        n_bad (int): Unused, kept for compatibility.

    Returns:
        tuple[dict, dict, dict, dict]: (TN, TP, FP, FN) record dicts.
    """
    # Find one example for each: TN, TP, FP, FN
    # TN: true=0, pred=0
    # TP: true=1, pred=1
    # FP: true=0, pred=1
    # FN: true=1, pred=0
    tn = [r for r in records if r["true"] == 0 and r["pred"] == 0]
    tp = [r for r in records if r["true"] == 1 and r["pred"] == 1]
    fp = [r for r in records if r["true"] == 0 and r["pred"] == 1]
    fn = [r for r in records if r["true"] == 1 and r["pred"] == 0]

    # Sort by confidence (descending)
    tn = sorted(tn, key=lambda r: r["p_pred"], reverse=True)
    tp = sorted(tp, key=lambda r: r["p_pred"], reverse=True)
    fp = sorted(fp, key=lambda r: r["p_pred"], reverse=True)
    fn = sorted(fn, key=lambda r: r["p_pred"], reverse=True)

    if not (tn and tp and fp and fn):
        raise RuntimeError(f"Not enough examples for all confusion matrix types: TN={len(tn)}, TP={len(tp)}, FP={len(fp)}, FN={len(fn)}. Try increasing sampled healthCodes or switch subset/fold.")
    # Pick the most confident example for each
    return tn[0], tp[0], fp[0], fn[0]



def plot_gradcam_confusion_grid(tn, tp, fp, fn, overlays, out_path):
    """
    Plot a 2x2 confusion matrix grid with Grad-CAM overlays for TN, TP, FP, FN.

    Args:
        tn, tp, fp, fn (dict): Record dicts for each confusion matrix type.
        overlays (dict): Mapping from 'tn', 'tp', 'fp', 'fn' to overlay images (np.ndarray).
        out_path (str): Path to save the resulting figure.

    Returns:
        None. Saves the figure to out_path.
    """
    label_map = {0: "HC", 1: "PD"}
    # 2x2 confusion matrix: rows = true (0,1), cols = pred (0,1)
    # [[TN, FP], [FN, TP]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Top-left: TN
    ax = axes[0, 0]
    ax.imshow(overlays["tn"], origin="lower", aspect="auto")
    ax.set_title(f"True Negative (TN)\n{tn['healthcode']}\nPred {label_map[tn['pred']]} (p={tn['p_pred']:.2f}) | True {label_map[tn['true']]}", fontsize=10)
    ax.axis("off")

    # Top-right: FP
    ax = axes[0, 1]
    ax.imshow(overlays["fp"], origin="lower", aspect="auto")
    ax.set_title(f"False Positive (FP)\n{fp['healthcode']}\nPred {label_map[fp['pred']]} (p={fp['p_pred']:.2f}) | True {label_map[fp['true']]}", fontsize=10)
    ax.axis("off")

    # Bottom-left: FN
    ax = axes[1, 0]
    ax.imshow(overlays["fn"], origin="lower", aspect="auto")
    ax.set_title(f"False Negative (FN)\n{fn['healthcode']}\nPred {label_map[fn['pred']]} (p={fn['p_pred']:.2f}) | True {label_map[fn['true']]}", fontsize=10)
    ax.axis("off")

    # Bottom-right: TP
    ax = axes[1, 1]
    ax.imshow(overlays["tp"], origin="lower", aspect="auto")
    ax.set_title(f"True Positive (TP)\n{tp['healthcode']}\nPred {label_map[tp['pred']]} (p={tp['p_pred']:.2f}) | True {label_map[tp['true']]}", fontsize=10)
    ax.axis("off")

    # Set tick labels for columns (Predicted)
    axes[1, 0].set_xlabel('Negative (0)', fontsize=13)
    axes[1, 1].set_xlabel('Positive (1)', fontsize=13)

    # Set tick labels for rows (True)
    axes[0, 0].set_ylabel('Negative (0)', fontsize=13)
    axes[1, 0].set_ylabel('Positive (1)', fontsize=13)

    # Set main axis labels
    fig.text(0.5, 0.01, 'Predicted label', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'True label', va='center', rotation='vertical', fontsize=14)
    fig.suptitle("Grad-CAM overlays — Confusion Matrix", fontsize=16)
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- YOU SET THESE ----
    SPLIT_CSV       = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
    FOLD_ITERATION  = 0
    SUBSET          = "test"  # or "val" 
    N_HEALTHCODES   = 25
    SEED            = 42

    MELSPEC_ROOT    = "/mloscratch/users/gnahas/data/melSpec" 
    LABEL_CSV       = "/tremor2tensor/src_GAMMA/paired_healthcode.csv" 
    LABEL_SEP       = ","  # adjust if needed

    CHECKPOINT_PATH = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/InceptionV3_MelSpec/V5_class_weights_do0.4_lr0.001_bs64_wd0.001/PTH/fold_0_best_auc_0.7370_epoch_33.pth"
    OUT_PATH        = "/mloscratch/users/gnahas/data/Grad-CAM/gradcam_grid_fold3_test_TN_TP_FP_FN.png"
    # -----------------------

    model = load_model(CHECKPOINT_PATH, device)
    labels = load_labels(LABEL_CSV, sep=LABEL_SEP)

    # 1) sample 25 healthCodes from fold+subset
    healthcodes = sample_healthcodes(SPLIT_CSV, FOLD_ITERATION, SUBSET, n=N_HEALTHCODES, seed=SEED)

    # 2) pick one mel JPG per healthCode
    hc_paths = pick_one_melspec_per_healthcode(MELSPEC_ROOT, healthcodes, seed=SEED, recursive=True)

    # 3) predict on those (fast)
    records = []
    missing_label = 0
    for hc, img_path in hc_paths:
        if hc not in labels:
            missing_label += 1
            continue
        true = labels[hc]
        pred, p_pred = predict_one(model, img_path, device)
        records.append({
            "healthcode": hc,
            "path": img_path,
            "true": true,
            "pred": pred,
            "p_pred": p_pred
        })

    print(f"Candidates: {len(hc_paths)} | With labels: {len(records)} | Missing labels: {missing_label}")

    # 4) select TN, TP, FP, FN
    tn, tp, fp, fn = select_good_bad(records)

    # 5) compute Grad-CAM overlays for these 4
    overlays = {}
    for name, rec in zip(["tn", "tp", "fp", "fn"], [tn, tp, fp, fn]):
        img_tensor = preprocess_image_for_model(rec["path"])
        cam = compute_gradcam(model, img_tensor, rec["pred"], device)
        overlays[name] = make_overlay_rgb(rec["path"], cam, alpha=0.45)

    # 6) final figure
    plot_gradcam_confusion_grid(tn, tp, fp, fn, overlays, OUT_PATH)


if __name__ == "__main__":
    main()
