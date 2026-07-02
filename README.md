# steel-defect-classification
CNN-based steel surface defect classification (NEU-DET) comparing a custom CNN, MobileNetV2, and ResNet50, with Grad-CAM/SHAP explainability.
# Steel Surface Defect Classification (NEU-DET)

CNN-based image classification project comparing a custom CNN, MobileNetV2, and 
ResNet50 for classifying 6 types of steel surface defects.

## Dataset
NEU-DET, 1800 images, 6 classes (crazing, inclusion, patches, pitted_surface, 
rolled-in_scale, scratches). Split: 1260 train / 270 val / 270 test.

## Results
| Model | Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 96.67% | 0.9664 |
| MobileNetV2 | 97.78% | 0.9775 |
| ResNet50 | 100% | 1.0 |

## Contents
- `notebook/` — full training and evaluation notebook
- `results/figures/` — training curves, confusion matrices, Grad-CAM, SHAP visualizations
- `results/tables/` — model comparison and class-wise performance CSVs

## Links
- Kaggle notebook: [add link]
- Web demo: [add link]
