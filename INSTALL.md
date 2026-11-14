# Installation and Run Guide

## Requirements

- Python **3.10 – 3.12**
- About 1 GB of free disk space (core app)
- Optional, for the AI-written report: about 4 GB extra disk space and 8 GB RAM (an Apple Silicon Mac or a CUDA GPU makes it faster)

## 1. Get the code

```bash
git clone https://github.com/annapoorna-a-k/Oral-Cancer-Detection-and-Multiclass-Classification.git
cd Oral-Cancer-Detection-and-Multiclass-Classification/app
```

## 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

## 3. Install the core dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, ONNX Runtime, XGBoost, scikit-learn, OpenCV, NumPy, Pillow and Matplotlib. It is enough to run segmentation, classification, Grad-CAM, SHAP and the standard (template) report.

### macOS only

XGBoost needs the OpenMP library:

```bash
brew install libomp
```

## 4. Run the app

```bash
streamlit run app.py
```

Streamlit opens the app at <http://localhost:8501>. Upload an intra-oral photo, optionally fill in the patient details, and press **Run screening**.

## 5. Optional: AI-written report (Qwen2.5-1.5B + LoRA)

```bash
pip install -r requirements-llm.txt
```

- On first use the base model `Qwen/Qwen2.5-1.5B-Instruct` (about 3 GB) is downloaded from Hugging Face; after that it runs offline.
- To use the fine-tuned LoRA adapter, place the adapter folder (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files) at `app/models/llm/qwen_report_lora/`. The app detects it automatically. The adapter is not included in this repository because of its size.
- The language model runs in a separate process (`llm_worker.py`) to avoid an OpenMP library conflict between PyTorch and XGBoost on macOS.
- Every generated report is checked by a rule-based safety validator; if it fails twice, the standard template report is shown instead.

If PyTorch or Transformers are not installed, the app hides the AI-report option and uses the template report.

## Model files

| File | Purpose |
|---|---|
| `models/segmentation/lw_adapters/lw_adapters.onnx` | Mobile U-Net with adapters (oral-cavity segmentation) |
| `models/classifier/EfficientNet-B0/Image_only/effb0_384.onnx` | EfficientNet-B0 image classifier (384 × 384) |
| `models/classifier/EfficientNet-B0/Image_only/effb0_384_head.npz` | classifier-head weights used for the Grad-CAM heatmap |
| `models/classifier/EfficientNet-B0/Image_only/config.json` | image preprocessing settings |
| `models/classifier/Metadata_only/xgboost.json` | metadata-only XGBoost model |
| `models/classifier/EfficientNet-B0/late_fusion/L4_xgboost.json` | final XGBoost stacking model (image probability + metadata) |
| `models/classifier/metadata_encoding.json` | metadata encoding (age scaling, one-hot categories) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `XGBoost Library (libxgboost.dylib) could not be loaded` | `brew install libomp` (macOS) |
| `sklearn needs to be installed` | `pip install scikit-learn` |
| `No module named 'peft'` | `pip install -r requirements-llm.txt` |
| Python crashes when the AI report is enabled (macOS) | make sure you run the provided `llm_worker.py`, which keeps PyTorch in a separate process |
| Changes to the code do not show | stop Streamlit (Ctrl + C) and start it again |
