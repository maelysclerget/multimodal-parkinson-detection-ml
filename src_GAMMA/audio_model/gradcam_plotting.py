#!/usr/bin/env python3
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
    model = InceptionV3Classifier()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def preprocess_image_for_model(img_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1,3,299,299)

def compute_gradcam(model, input_tensor, target_class, device):
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
    split_csv columns: fold_iteration, healthCode, subset
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
    Finds one JPG per healthCode by globbing for: {healthCode}_*.jpg
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
def load_labels(label_csv, sep=";"):
    df = pd.read_csv(label_csv, sep=sep)
    df["healthCode"] = df["healthCode"].astype(str)
    return {hc: int(lbl) for hc, lbl in zip(df["healthCode"], df["label_PD"])}


@torch.no_grad()
def predict_one(model, img_path, device):
    x = preprocess_image_for_model(img_path).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    p_pred = float(probs[pred])
    return pred, p_pred


def select_good_bad(records, n_good=4, n_bad=4):
    good = [r for r in records if r["pred"] == r["true"]]
    bad  = [r for r in records if r["pred"] != r["true"]]

    good = sorted(good, key=lambda r: r["p_pred"], reverse=True)
    bad  = sorted(bad,  key=lambda r: r["p_pred"], reverse=True)  # high-conf wrong

    if len(good) < n_good or len(bad) < n_bad:
        raise RuntimeError(f"Not enough examples: good={len(good)}, bad={len(bad)}. "
                           f"Try increasing sampled healthCodes or switch subset/fold.")
    return good[:n_good], bad[:n_bad]


def plot_gradcam_grid(good, bad, overlays_good, overlays_bad, out_path):
    label_map = {0: "HC", 1: "PD"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.08, hspace=0.28, top=0.90)

    for i in range(4):
        ax = axes[0, i]
        ax.imshow(overlays_good[i], origin="lower", aspect="auto")
        r = good[i]
        ax.set_title(
            f"{r['healthcode']}\nPred {label_map[r['pred']]} (p={r['p_pred']:.2f}) | True {label_map[r['true']]}",
            fontsize=10
        )
        ax.axis("off")

    for i in range(4):
        ax = axes[1, i]
        ax.imshow(overlays_bad[i], origin="lower", aspect="auto")
        r = bad[i]
        ax.set_title(
            f"{r['healthcode']}\nPred {label_map[r['pred']]} (p={r['p_pred']:.2f}) | True {label_map[r['true']]}",
            fontsize=10
        )
        ax.axis("off")

    fig.suptitle("Grad-CAM overlays — well-predicted (top) vs misclassified (bottom)", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- YOU SET THESE ----
    SPLIT_CSV       = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"   # has fold_iteration,healthCode,subset
    FOLD_ITERATION  = 3
    SUBSET          = "test"                          # or "val" (must match CSV exactly)
    N_HEALTHCODES   = 25
    SEED            = 42

    MELSPEC_ROOT    = "/mloscratch/users/gnahas/data/melSpec"  # folder containing all melspec JPGs
    LABEL_CSV       = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"  # healthCode,label_PD
    LABEL_SEP       = ";"  # adjust if needed

    CHECKPOINT_PATH = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/InceptionV3_MelSpec/V1/PTH/fold_3_best_auc_0.6580_epoch_10.pth"
    OUT_PATH        = "/mloscratch/users/gnahas/data/Grad-CAM/gradcam_grid_fold3_test_good_vs_bad_2.png"
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

    # 4) choose 4 good + 4 bad
    good, bad = select_good_bad(records, n_good=4, n_bad=4)

    # 5) compute Grad-CAM overlays only for selected 8
    overlays_good, overlays_bad = [], []

    for r in good:
        img_tensor = preprocess_image_for_model(r["path"])
        cam = compute_gradcam(model, img_tensor, r["pred"], device)
        overlays_good.append(make_overlay_rgb(r["path"], cam, alpha=0.45))

    for r in bad:
        img_tensor = preprocess_image_for_model(r["path"])
        cam = compute_gradcam(model, img_tensor, r["pred"], device)
        overlays_bad.append(make_overlay_rgb(r["path"], cam, alpha=0.45))

    # 6) final figure
    plot_gradcam_grid(good, bad, overlays_good, overlays_bad, OUT_PATH)


if __name__ == "__main__":
    main()
