[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jZYLDMog)

# Multimodal Parkinson's Disease Detection — mPower (Audio & Tapping)

This repository contains the code used to train **unimodal** (audio, tapping) and **multimodal** models for Parkinson's disease detection using the mPower dataset.  
All development is organized under `src_GAMMA/`, with responsibilities split across team members by modality.

---

> ⚠️ **IMPORTANT: Data Confidentiality**  
> - The mPower dataset is **confidential** and **not included** in this repository.
> - Access requires approval via Synapse and compliance with data-use agreements.

---

## Data Access (mPower)

The mPower dataset is governed by a rigorous and participant-centred data governance framework designed to balance scientific openness with strong privacy protections.

🔗 **mPower Study Access**  
https://www.synapse.org/Synapse:syn4993293/wiki/247860

To access the data, you need:
1. A Synapse account  
2. Approval for the mPower study  
3. Compliance with Synapse data-use requirements  

---

## Repository Structure

All source code is organized under `src_GAMMA/`. Responsibilities are split across team members and modules.

| Path | Description | Owner |
|------|------------|-------|
| `src_GAMMA/audio_model/` | Audio unimodal models (features, spectrograms, waveforms) | Person A |
| `src_GAMMA/tapping_model/` | Tapping unimodal models (features, heatmaps, CNN/MLP) | Person B |
| `src_GAMMA/FusionModels/` | Multimodal fusion model definitions | Person C |
| `src_GAMMA/EmbeddingExtractor/` | Shared embedding extraction utilities | Person C |
| `src_GAMMA/FusionPipelines/` | Early, intermediate, and late fusion pipelines | Person C |
| `healthcode_5fold_train.csv` | Train patient split for each fold | — |
| `healthcode_5fold_val_test.csv` | Validation and test patient splits for each fold | — |

### Development Responsibilities
- **Person A**: Audio preprocessing, feature extraction, spectrogram models, and audio unimodal classifiers.
- **Person B**: Finger-tapping feature extraction, heatmap generation, unimodal MLP/CNN models, and interpretability.
- **Person C**: Multimodal fusion strategies, embedding extractors, and fusion pipelines.

---

## Reproducibility and Patient Splits

This repository includes all files required to **reproduce patient-level experiments deterministically** (except the data).

### Patient Pairing
`paired_healthcode.csv` specifies which patients have **both audio and tapping data** and are eligible for paired multimodal experiments.

### Cross-Validation Splits
To ensure consistent evaluation, **fixed patient-level splits** are provided for **5-fold cross-validation**:

| File | Description |
|------|-------------|
| `healthcode_5fold_train.csv` | Patients assigned to the **training set** for each fold |
| `healthcode_5fold_val_test.csv` | Patients assigned to the **validation** and **test** sets for each fold |

Each file explicitly indicates, **for every fold**, whether a patient belongs to the **train**, **validation**, or **test** set. No patient appears in more than one split within a fold.

> **Note:** All experiments can be reproduced directly using these CSV files, as patient splitting is fully specified and independent of the modeling code.

---
## Audio Module — Overview
### `audio_model/results_audio/`

Contains output figures and model performance summaries from audio-based experiments. 
> These results are not included in the final report but are shown for additional exploratory insights.

### `audio_model`

| File | Description |
|------|-------------|
| `cnn_melspec_model.py` | CNN model for mel-spectrogram classification |
| `inceptionv3_melspec_model.py` | InceptionV3-based model for mel-spectrogram analysis |
| `mlp_features_model.py` | MLP model using extracted audio features |
| `resnet1d_features_model.py` | 1D ResNet for audio feature-based classification |
| `resnet_melspec_model.py` | ResNet model for mel-spectrogram classification |
| `tcn_waveform_model.py` | Temporal Convolutional Network for raw waveform processing |

### `audio_model/utils/`
Contains 3 subfolders that allows to 
- Explore and visualize
- Extract features
- Convert files to different files

---

## Tapping Module — Overview

### `tapping_model/results_tapping/`
Contains output figures and model performance summaries from tapping-based experiments. 
> These results are not included in the final report but are shown for additional exploratory insights.

### `tapping_model\utils`

| File | Description |
|------|-------------|
| `advanced-features.py` | Computes advanced tapping features (fatigue, variability, fluctuations) |
| `data-loading.py` | Dataset loading and preprocessing utilities |
| `data-exploration.ipynb` | Exploratory data analysis |
| `tapping-heatmaps.py` | Generates spatial heatmaps from tap coordinates |
| `shap_values.py` | SHAP value computation |
| `shap_aggregate.py` | Patient-level SHAP aggregation |
| `grad_cam.py` | Grad-CAM for CNN interpretability |
| `grad_cam_panel.py` | Grad-CAM visualization utilities |

### `tapping_model`
| File | Description |
|------|-------------|
| `CNN-heatmaps.py` | CNN for heatmap-based tapping classification |
| `mlp-v1.py` | Initial MLP baseline experiments |
| `mlp-v2.py` | **Final MLP model** with class imbalance handling |
---
## Fusion Module — Overview

### `FusionPipelines/results/`

Contains output figures and model performance summaries from multimodal fusion experiments.

### `FusionPipelines/`

| File | Description |
|------|-------------|
| `EarlyFeatureFusionPipeline.ipynb` | Early fusion pipeline combining audio and tapping features |
| `IntermediateFeatureFusionPipeline.ipynb` | Intermediate fusion using feature-level embeddings |
| `IntermediateImageFusionPipeline.ipynb` | Intermediate fusion using image-based embeddings |
| `LateFeatureFusionPipeline.ipynb` | Late fusion combining predictions from feature-based unimodal models |

### `FusionModels/`

| File | Description |
|------|-------------|
| `IntermediateFusionMLP.py` | MLP for intermediate fusion strategy |
| `LateFusionMLP.py` | MLP for late fusion strategy |
| `MLP.py` | Base MLP architecture for fusion experiments |
| `utils.py` | Utility functions for fusion pipelines |

### `EmbeddingExtractor/`

| File | Description |
|------|-------------|
| `ImageEmbeddingExtractor.py` | Extracts embeddings from image-based representations (spectrograms, heatmaps) |

---
##  Computing Environment

All experiments were developed and executed on the **EPFL Research Computing Platform (RCP)**. Paths and scripts assume a cluster environment and may require adaptation for local execution.

---

## ▶ How to Run the Code

Before running any script:

1. **Update all data paths** inside the Python files to match your local dataset location.
   - **All paths starting with `/mloscratch/users` must be replaced with your own data directory path.**

2. **Ensure required CSV files are present** in the project root directory:
   - `paired_healthcode.csv`
   - `healthcode_5fold_train.csv`
   - `healthcode_5fold_val_test.csv`

3. **Activate your Python environment** with required dependencies:
   - PyTorch
   - Timm (for image embedding extractions)

### Example — Run the final tapping MLP model
```bash
cd src_GAMMA/tapping_model
python mlp-v2.py
```
---

## Citation

If you use this code in academic work:
- Cite the **mPower study dataset**
- Cite any related coursework or publications based on this repository