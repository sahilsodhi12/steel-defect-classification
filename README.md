
# Steel Surface Defect Classification (NEU-DET)

CNN-based image classification project comparing a custom CNN, MobileNetV2, and ResNet50 for classifying 6 types of steel surface defects, with Grad-CAM and SHAP explainability to visualize what each model is actually learning.

## Dataset

NEU-DET, 1,800 grayscale images, 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) — 300 images per class, perfectly balanced.

**Split:** Stratified 70/15/15 → 1,260 train / 270 val / 270 test (recombined from the dataset's original train/validation-only release to avoid data leakage and enable honest generalization testing).

**Augmentation (train only):** rotation, width/height shift, zoom, brightness jitter, horizontal + vertical flip. Vertical flip is valid here since defect orientation on a rolled steel strip carries no semantic meaning.

## Models

| Model | Approach |
|---|---|
| Custom CNN | 4-block Conv2D + BatchNorm + MaxPooling, trained from scratch |
| MobileNetV2 | ImageNet-pretrained backbone, 2-stage transfer learning (frozen head → fine-tune last 30 layers) |
| ResNet50 | ImageNet-pretrained backbone, 2-stage transfer learning (frozen head → fine-tune last 30 layers) |

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 96.67% | 0.9664 |
| MobileNetV2 | 97.78% | 0.9775 |
| ResNet50 | 100% | 1.0 |

**Best model:** ResNet50 achieves the highest raw accuracy, but MobileNetV2 offers the best accuracy-to-efficiency tradeoff — it's ~8x smaller and trains/infers faster than ResNet50, making it the more practical choice for deployment on constrained hardware.

> **Note on ResNet50's 100% accuracy:** a perfect score on a 270-image test set is a strong result, but with a relatively small, tightly-controlled dataset (300 images per class, consistent imaging conditions), it likely reflects limited appearance variation within each class rather than perfect real-world generalization. Validating on an external or more diverse test set would give a more reliable estimate of production performance.

## Explainability

To go beyond raw accuracy, the notebook generates:
- **Grad-CAM** heatmaps — showing which image regions each model focuses on when making a prediction
- **SHAP** pixel-level attributions — showing which individual pixels push the prediction toward or away from a class
- **Misclassified examples** — representative failure cases for qualitative error analysis

These are useful for sanity-checking that models are keying off actual defect patterns (cracks, scale, pitting) rather than incidental artifacts like lighting or background texture — an important check before trusting a model in a real inspection pipeline.

## Repository Structure

```
steel-defect-classification/
├── notebook/
│   └── steel_classification_nb_final.ipynb
├── results/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── class_samples.png
│   │   ├── augmentation_examples.png
│   │   ├── custom_cnn_training_curves.png
│   │   ├── mobilenetv2_training_curves.png
│   │   ├── resnet50_training_curves.png
│   │   ├── confusion_matrix_custom_cnn.png
│   │   ├── confusion_matrix_mobilenetv2.png
│   │   ├── confusion_matrix_resnet50.png
│   │   ├── gradcam_examples.png
│   │   ├── shap_examples.png
│   │   └── misclassified_examples.png
│   └── tables/
│       ├── model_comparison.csv
│       ├── classwise_performance.csv
│       └── run_summary.json
└── README.md
```


## Tech Stack

TensorFlow / Keras · scikit-learn · SHAP · Grad-CAM · pandas · matplotlib / seaborn

## How to Run

1. Download the NEU-DET dataset and place it under a path discoverable by the notebook (originally run on Kaggle with `/kaggle/input`).
2. Open `notebook/` and run the training notebook top to bottom.
3. Trained models, figures, and tables will be written to `results/`.

## Links

- Kaggle notebook: https://www.kaggle.com/code/sodhisahil/steel-classification-nb
- Overleaf report: https://www.overleaf.com/read/gncjmznpmnvz#a77db9

- ## Live Demo
Try the deployed classifier here: https://huggingface.co/spaces/sahil-sodhi/steel-defect-classifier
(interface source code in `webapp/`)

