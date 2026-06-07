# Meat Freshness Classification Using CNNs

**Machine Learning – Phase 2 Project**

A deep learning pipeline that classifies meat images as **Fresh** or **Spoiled** using three CNN architectures: a custom-designed network, MobileNetV2 (transfer learning), and EfficientNetB0 (transfer learning). The project includes Grad-CAM visualisations, error analysis, and an ablation study on data augmentation.

---

## Dataset

**Meat Freshness Image Dataset** — available on Kaggle:
https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset

Download the dataset and place it at the path configured in the notebook, or update `KAGGLE_INPUT` in the code to match your local path.

---

## Project Structure

```
meat-freshness-cnn/
├── meat_freshness_classification.ipynb   # Main Kaggle notebook
├── meat_freshness_classification.py      # Equivalent Python script
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
└── outputs/                              # Generated figures and CSVs (auto-created)
```

---

## Requirements

Python 3.9 or later is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
Pillow>=9.4.0
```

> **GPU note:** The code runs on CPU but is significantly faster with a CUDA-enabled GPU. On Kaggle, enable the GPU accelerator under *Settings → Accelerator → GPU T4 x2*.

---

## How to Run

### Option A — Kaggle (recommended)

1. Open the notebook on Kaggle.
2. Attach the *Meat Freshness Image Dataset* under *Add Data*.
3. Enable GPU under *Settings → Accelerator*.
4. Click **Run All**.

### Option B — Local

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/meat-freshness-cnn.git
cd meat-freshness-cnn

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and update KAGGLE_INPUT in the script

# 5. Run
python meat_freshness_classification.py
```

---

## Models

| Model | Strategy | Parameters (approx.) |
|---|---|---|
| Custom CNN | Trained from scratch | ~1.5 M |
| MobileNetV2 | Transfer learning + fine-tuning (top 30 layers) | ~2.3 M |
| EfficientNetB0 | Transfer learning + fine-tuning (top 20 layers) | ~4.1 M |

---

## Outputs

All figures and result files are saved to `/kaggle/working/outputs/` (Kaggle) or `./outputs/` (local):

| File | Description |
|---|---|
| `fig01_class_distribution.png` | Class distribution bar chart |
| `fig02_sample_images.png` | Sample images per class |
| `fig_curves_*.png` | Training/validation accuracy and loss curves |
| `fig_cm_*.png` | Confusion matrices |
| `fig_roc_*.png` | ROC curves with AUC scores |
| `fig_gradcam_*.png` | Grad-CAM heat-map overlays |
| `fig_errors_*.png` | Misclassified sample grids |
| `fig_model_comparison.png` | Grouped bar chart comparing all models |
| `fig_ablation.png` | Augmentation ablation comparison |
| `model_comparison.csv` | Numeric metrics for all three models |
| `ablation_results.csv` | Augmentation ablation numeric results |
| `best_custom_cnn.keras` | Best Custom CNN checkpoint |
| `best_mv2_ft.keras` | Best MobileNetV2 checkpoint (fine-tuned) |
| `best_eff_ft.keras` | Best EfficientNetB0 checkpoint (fine-tuned) |

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 px |
| Batch size | 32 |
| Max epochs (initial) | 30 |
| Fine-tuning epochs | 10 |
| Base learning rate | 1 × 10⁻³ |
| Fine-tuning learning rate | 1 × 10⁻⁵ |
| Random seed | 42 |

---

## Reproducibility

The random seed is fixed at `42` for Python, NumPy, and TensorFlow. Results may vary slightly across different hardware and TensorFlow versions due to non-deterministic GPU operations.

---

## License

This project is submitted for academic purposes. The dataset is subject to its original Kaggle licence — please review the dataset page before any commercial use.
