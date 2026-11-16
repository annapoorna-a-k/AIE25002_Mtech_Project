# Notebooks

Training and experiment notebooks, in pipeline order. Most were run on Kaggle (GPU) and are stored with their outputs.

| # | Folder | Contents |
|---|---|---|
| 01 | `01_eda` | Metadata analysis and patient-level risk-factor plots |
| 02 | `02_patient_level_split` | Patient-level stratified train / validation / test split |
| 03 | `03_segmentation_model_comparison` | U-Net + ResNet34, DeepLabV3+ and Mobile U-Net; comparison table and plots |
| 04 | `04_mobile_unet_finetuning` | Hyperparameter sweeps and fine-tuning strategies (adapters, top layers, full, head only) |
| 05 | `05_lesion_segmentation_experiments` | Lesion-segmentation experiments with Mobile U-Net variants |
| 06 | `06_crop_generation` | Oral-cavity segmentation inference and crop generation for classification |
| 07 | `07_augmentation` | Image augmentation |
| 08 | `08_four_class_backbone_comparison` | EfficientNet-B0, EfficientNetV2-B0, MobileNetV2 and MobileNetV3-Small (four classes) |
| 09 | `09_four_class_efficientnet_tuning` | Stage-wise tuning of EfficientNet-B0 (four classes) |
| 10 | `10_attention_experiments` | Attention inside the image model: SE, CNN-Transformer hybrid and axial attention |
| 11 | `11_yolo_detection_classification` | YOLO26n and YOLO26s: detect the lesion, crop it and classify the crop |
| 12 | `12_binary_metadata_fusion` | Binary screening with metadata fusion at 224 px and 384 px (final model) |
| 13 | `13_explainability` | Grad-CAM, Grad-CAM++ and SHAP analysis of the final model |
| 14 | `14_synthetic_data_generation` | Class-conditioned diffusion (LoRA) and StarGAN translation |
| 15 | `15_report_generation_llm` | LoRA fine-tuning of Qwen2.5-1.5B for report generation |

## Notes

- In `03_segmentation_model_comparison`, notebooks `01`–`03` are the original training runs for each model; `04_model_comparison.ipynb` combines their results, saved in `results/`.
- The trained U-Net + ResNet34 and DeepLabV3+ weight files (about 300 MB each) are not included because of GitHub's file-size limit.
- The dataset is not included; see the main README for the source.
