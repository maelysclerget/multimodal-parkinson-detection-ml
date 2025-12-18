[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jZYLDMog)

# Multimodal Parkinson’s Disease Detection — mPower (Audio & Tapping)

This repository contains the code used to train **unimodal** (audio, tapping) and **multimodal** models for Parkinson’s disease detection using the mPower dataset.  
All development is organized under `src_GAMMA/`, with responsibilities split across team members by modality.

---

> ⚠️ Raw mPower data are **not included**. Access requires approval via Synapse.

---

## Data Access (mPower)

The mPower dataset used in this project is governed by a rigorous and participant-centred data governance framework designed to balance scientific openness with strong privacy protections. The mPower dataset can be obtained from Synapse:

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
| `paired_healthcode.csv` | Audio–tapping patient pairing file (reproducibility) | — |
| `healthcode_5fold_train.csv` | Train patient split for each fold | — |
| `healthcode_5fold_val_test.csv` | Validation and test patient splits for each fold | — |

---

## Reproducibility and Patient Splits

This repository includes all files required to **reproduce patient-level experiments deterministically** except the data. 

### Patient pairing
- `paired_healthcode.csv` specifies which patients have **both audio and tapping data** and are therefore eligible for paired multimodal experiments.

### Cross-validation splits
To ensure consistent evaluation, **fixed patient-level splits** are provided:

| File | Description |
|------|-------------|
| `healthcode_5fold_train.csv` | Patients assigned to the **training set** for each fold |
| `healthcode_5fold_val_test.csv` | Patients assigned to the **validation** and **test** sets for each fold |

Each file explicitly indicates, **for every fold**, whether a patient belongs to the **train**, **validation**, or **test** set.

All experiments use **5-fold cross-validation at the patient level**, ensuring that no patient appears in more than one split within a fold.

> **Note:** TAs can re-run all experiments directly using these CSV files, as patient splitting is fully specified and independent of the modeling code.

### Development Responsibilities
- **Person A**: `audio_model/`  
  Audio preprocessing, feature extraction, spectrogram models, and audio unimodal classifiers.
- **Person B**: `tapping_model/`  
  Finger-tapping feature extraction, heatmap generation, unimodal MLP/CNN models, and interpretability.
- **Person C**: `fusion_models/`  
  Multimodal fusion strategies, embedding extractors, and fusion pipelines.

---

## Tapping Module — Overview

### `tapping_model/`

| Path | Type | Description |
|-----|------|-------------|
| `results_tapping/` | folder | Output figures and model summaries |
| `01_basic_baseline_Basic_Features_results.png` | image | Results using basic tapping features |
| `02_advanced_baseline_Advanced_Features_results.png` | image | Results using advanced tapping features |
| `03_combined_Combined_Features_results.png` | image | Results using combined features |
| `04_cnn_heatmap_Heatmap_CNN_results.png` | image | CNN performance on tapping heatmaps |
| `mlp_v2_best_model_summary_undersampling.png` | image | Best MLP with undersampling |
| `mlp_v2_best_model_summary_weighted_sampler.png` | image | Best MLP with weighted sampling |

These results are not included in the final report but are shown for additional exploratory insights.---

### `tapping_model/utils/`

| File | Description |
|------|-------------|
| `advanced-features.py` | Computes advanced tapping features (fatigue, variability, fluctuations) |
| `data-loading.py` | Dataset loading and preprocessing utilities |
| `data-exploration.ipynb` | Exploratory data analysis |
| `tapping-heatmaps.py` | Generates spatial heatmaps from tap coordinates |
| `CNN-heatmaps.py` | CNN for heatmap-based tapping classification |
| `mlp-v1.py` | Initial MLP baseline experiments |
| `mlp-v2.py` | **Final MLP model** with class imbalance handling |
| `shap_values.py` | SHAP value computation |
| `shap_aggregate.py` | Patient-level SHAP aggregation |
| `grad_cam.py` | Grad-CAM for CNN interpretability |
| `grad_cam_panel.py` | Grad-CAM visualization utilities |

---

## How to Run the Code

Before running any script:
1. **Update all data paths** inside the Python files to match your local dataset location.
   - **All paths starting with `/mloscratch/users` must be replaced with your own data directory path.**
2. Activate your Python environment with required dependencies: 
   - PyTorch 
   - timm (for image embedding extractions)

## Citation
If you use this code in academic work:
- Cite the **mPower study dataset**
- Cite any related coursework or publications based on this repository

### Example — Run the final tapping MLP model
```markdown
```bash

cd src_GAMMA/tapping_model
python mlp-v2.py



