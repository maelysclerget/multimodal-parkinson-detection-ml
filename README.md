# Multimodal Parkinson's Disease Detection — mPower (Audio & Tapping)

This repository contains the code developed for a **three-person collaborative project** on Parkinson's disease detection using the mPower dataset. The project investigates both **unimodal** approaches (audio and finger tapping) and **multimodal fusion** strategies combining the two modalities.

All development is organized under `src_GAMMA/`, with responsibilities divided across the three team members according to modality and modeling tasks.

I, **Maëlys Clerget**, was primarily responsible for the **tapping modality**, including tapping preprocessing, feature extraction, spatial heatmap generation, unimodal MLP/CNN modeling, class-imbalance handling, and model interpretability using methods such as SHAP and Grad-CAM. I also contributed to broader project components including **data processing, evaluation metrics, model comparison, experimental design, and interpretation of results**.

---

## Report

A detailed description of the project methodology, experiments, results, and conclusions is available in the **final project report**.

📄 **[Read the full project report](Multimodal_PD_report.pdf)**

The report provides additional context on:

- Dataset preprocessing and patient-level splitting
- Audio and tapping feature engineering
- Unimodal model architectures
- Multimodal fusion strategies
- Evaluation metrics
- Experimental results
- Model comparison and interpretation
- Limitations and future directions

---

> ⚠️ **IMPORTANT: Data Confidentiality**
>
> - The mPower dataset is **confidential** and **not included** in this repository.
> - Access requires approval via Synapse and compliance with the corresponding data-use agreements.

---

## Data Access — mPower

The mPower dataset is governed by a participant-centred data governance framework designed to enable scientific research while protecting participant privacy.

🔗 **mPower Study Access**  
https://www.synapse.org/Synapse:syn4993293/wiki/247860

To access the data, you need:

1. A Synapse account
2. Approval for access to the mPower study
3. Compliance with the applicable Synapse data-use requirements

---

## Project Collaboration

This project was developed collaboratively by **three team members**, with each person focusing on complementary parts of the pipeline.

| Role | Main Responsibilities |
|---|---|
| **Maëlys Clerget — Tapping** | Tapping preprocessing, feature extraction, spatial heatmaps, tapping unimodal MLP/CNN models, class-imbalance handling, interpretability, evaluation metrics, data processing, model comparison, and general experimental analysis |
| **Person B — Audio** | Audio preprocessing, feature extraction, spectrogram generation, waveform processing, and audio unimodal models |
| **Person C — Multimodal Fusion** | Multimodal fusion strategies, embedding extraction, and early/intermediate/late fusion pipelines |

The final results therefore reflect a collaborative effort across **audio modeling, tapping modeling, and multimodal fusion**.

---

## Repository Structure

All source code is organized under `src_GAMMA/`.

| Path | Description | Main Contributor |
|---|---|---|
| `src_GAMMA/audio_model/` | Audio unimodal models using features, spectrograms, and waveforms | Person B |
| `src_GAMMA/tapping_model/` | Tapping unimodal models using engineered features, heatmaps, CNNs, and MLPs | **Maëlys Clerget** |
| `src_GAMMA/FusionModels/` | Multimodal fusion model definitions | Person C |
| `src_GAMMA/EmbeddingExtractor/` | Shared embedding extraction utilities | Person C |
| `src_GAMMA/FusionPipelines/` | Early, intermediate, and late fusion pipelines | Person C |
| `healthcode_5fold_train.csv` | Train patient split for each fold | Shared |
| `healthcode_5fold_val_test.csv` | Validation and test patient splits for each fold | Shared |

---

## Reproducibility and Patient Splits

This repository includes the files required to reproduce the patient-level experimental splits deterministically, excluding the confidential mPower data itself.

### Patient Pairing

`paired_healthcode.csv` specifies which patients have **both audio and tapping data** and are therefore eligible for paired multimodal experiments.

### Cross-Validation Splits

To ensure consistent evaluation, fixed patient-level splits are provided for **5-fold cross-validation**.

| File | Description |
|---|---|
| `healthcode_5fold_train.csv` | Patients assigned to the **training set** for each fold |
| `healthcode_5fold_val_test.csv` | Patients assigned to the **validation** and **test** sets for each fold |

Each file explicitly specifies, for every fold, whether a patient belongs to the training, validation, or test set.

No patient appears in more than one split within the same fold.

> **Note:** These predefined splits allow the experiments to be reproduced consistently across different modeling approaches.

---

## Audio Module — Overview

### `audio_model/results_audio/`

Contains output figures and model performance summaries from audio-based experiments.

> These results include additional exploratory analyses that were not necessarily included in the final report.

### `audio_model/`

| File | Description |
|---|---|
| `cnn_melspec_model.py` | CNN model for mel-spectrogram classification |
| `inceptionv3_melspec_model.py` | InceptionV3-based model for mel-spectrogram analysis |
| `mlp_features_model.py` | MLP model using extracted audio features |
| `resnet1d_features_model.py` | 1D ResNet for audio feature-based classification |
| `resnet_melspec_model.py` | ResNet model for mel-spectrogram classification |
| `tcn_waveform_model.py` | Temporal Convolutional Network for raw waveform processing |

### `audio_model/utils/`

This directory contains utilities for:

- Audio data exploration and visualization
- Audio preprocessing
- Feature extraction
- Spectrogram generation
- File-format conversion
- Supporting evaluation and analysis workflows

---

## Tapping Module — Overview

The tapping component was primarily developed by **Maëlys Clerget**.

The objective was to investigate behavioral tapping patterns and compare multiple machine learning and deep learning approaches for Parkinson's disease classification.

### `tapping_model/results_tapping/`

Contains output figures and model performance summaries from tapping-based experiments.

> These results provide additional exploratory insights beyond the final report.

### `tapping_model/utils/`

| File | Description |
|---|---|
| `advanced-features.py` | Computes advanced tapping features such as fatigue, variability, and fluctuations |
| `data-loading.py` | Dataset loading and preprocessing utilities |
| `data-exploration.ipynb` | Exploratory data analysis |
| `tapping-heatmaps.py` | Generates spatial heatmaps from tap coordinates |
| `shap_values.py` | SHAP value computation |
| `shap_aggregate.py` | Patient-level SHAP aggregation |
| `grad_cam.py` | Grad-CAM for CNN interpretability |
| `grad_cam_panel.py` | Grad-CAM visualization utilities |

### `tapping_model/`

| File | Description |
|---|---|
| `CNN-heatmaps.py` | CNN for heatmap-based tapping classification |
| `mlp-v1.py` | Initial MLP baseline experiments |
| `mlp-v2.py` | **Final MLP model** with class imbalance handling |

In addition to developing the tapping models, I contributed to project-wide tasks such as **evaluation metric selection, performance comparison, data processing decisions, and interpretation of experimental results**.

---

## Fusion Module — Overview

### `FusionPipelines/results/`

Contains output figures and model performance summaries from multimodal fusion experiments.

### `FusionPipelines/`

| File | Description |
|---|---|
| `EarlyFeatureFusionPipeline.ipynb` | Early fusion pipeline combining audio and tapping features |
| `IntermediateFeatureFusionPipeline.ipynb` | Intermediate fusion using feature-level embeddings |
| `IntermediateImageFusionPipeline.ipynb` | Intermediate fusion using image-based embeddings |
| `LateFeatureFusionPipeline.ipynb` | Late fusion combining predictions from feature-based unimodal models |

### `FusionModels/`

| File | Description |
|---|---|
| `IntermediateFusionMLP.py` | MLP for intermediate fusion strategy |
| `LateFusionMLP.py` | MLP for late fusion strategy |
| `MLP.py` | Base MLP architecture for fusion experiments |
| `utils.py` | Utility functions for fusion pipelines |

### `EmbeddingExtractor/`

| File | Description |
|---|---|
| `ImageEmbeddingExtractor.py` | Extracts embeddings from image-based representations such as spectrograms and tapping heatmaps |

---

## Computing Environment

All experiments were developed and executed on the **EPFL Research Computing Platform (RCP)**.

Some paths and scripts therefore assume a cluster-based environment and may require adaptation for local execution.

---

## ▶ How to Run the Code

Before running any script:

1. **Update the data paths** inside the Python files to match your local dataset location.
   - All paths starting with `/mloscratch/users` should be replaced with your own data directory.

2. **Ensure that the required CSV files are present** in the project root directory:
   - `paired_healthcode.csv`
   - `healthcode_5fold_train.csv`
   - `healthcode_5fold_val_test.csv`

3. **Activate a Python environment** containing the required dependencies, including:
   - PyTorch
   - Timm
   - NumPy
   - Pandas
   - scikit-learn

### Example — Run the final tapping MLP model

```bash
cd src_GAMMA/tapping_model
python mlp-v2.py
