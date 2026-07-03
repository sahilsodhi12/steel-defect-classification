import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "mobilenetv2_final.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

CLASS_INFO = {
    "crazing": "Fine network of surface cracks caused by thermal or mechanical stress during rolling.",
    "inclusion": "Foreign particles embedded in the steel surface during the manufacturing process.",
    "patches": "Irregular, blotchy discoloured regions on the steel surface.",
    "pitted_surface": "Small, shallow depressions scattered across the surface.",
    "rolled-in_scale": "Oxide scale pressed into the surface during hot rolling.",
    "scratches": "Linear surface marks caused by mechanical contact during handling.",
}

model = tf.keras.models.load_model(MODEL_PATH)


def predict(image):
    if image is None:
        return None
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}


custom_css = """
.gradio-container {max-width: 1000px !important; margin: auto !important;}
#hero {text-align: center; padding: 10px 0 20px 0;}
#hero-title {font-size: 2.4em; font-weight: 800; margin-bottom: 4px;}
#hero-sub {color: #a0a0a0; font-size: 1.05em; margin-bottom: 10px;}
.badge-row {display: flex; justify-content: center; gap: 10px; margin-top: 10px; flex-wrap: wrap;}
.badge {background: #2d3748; color: #fbbf24; padding: 5px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 600;}
.feature-card {background: #1f2430; border-radius: 12px; padding: 16px; text-align: center;}
.feature-card h4 {margin: 6px 0 4px 0;}
.feature-card p {color: #999; font-size: 0.85em; margin: 0;}
footer {visibility: hidden}
"""

theme = gr.themes.Soft(primary_hue="orange", secondary_hue="slate").set(
    body_background_fill="*neutral_950", block_background_fill="*neutral_900"
)

with gr.Blocks(css=custom_css, theme=theme, title="Steel Surface Defect Classifier") as demo:

    gr.HTML("""
    <div id="hero">
        <div id="hero-title">Steel Surface Defect Classifier</div>
        <div id="hero-sub">Upload a steel surface image and instantly identify the defect type using a MobileNetV2 CNN trained on NEU-DET.</div>
        <div class="badge-row">
            <span class="badge">97.78% Test Accuracy</span>
            <span class="badge">6 Defect Classes</span>
            <span class="badge">24.6 MB Model</span>
            <span class="badge">Real-Time Inference</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("Classify"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload a steel surface image")
                    with gr.Row():
                        clear_btn = gr.ClearButton([image_input])
                        submit_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    output_label = gr.Label(num_top_classes=6, label="Predicted Defect Class")

            submit_btn.click(fn=predict, inputs=image_input, outputs=output_label)

            gr.HTML("<br>")
            gr.Markdown("### Defect Classes This Model Recognizes")
            with gr.Row():
                for cls, desc in CLASS_INFO.items():
                    gr.HTML(f"""
                    <div class="feature-card">
                        <h4>{cls.replace('_',' ').replace('-',' ').title()}</h4>
                        <p>{desc}</p>
                    </div>
                    """)

        with gr.TabItem("About"):
            gr.Markdown(
                """
### About This Project

This tool classifies steel surface defects into six categories using a deep learning model trained on the **NEU-DET** (Northeastern University Surface Defect Database), a benchmark dataset of 1,800 grayscale images used to develop and evaluate automated visual inspection systems for steel manufacturing.

**Why this matters:** manual inspection of steel surfaces is slow, inconsistent, and difficult to scale to modern high-speed production lines. Automated, vision-based classification helps catch defects like cracks, inclusions, and scale before they reach downstream production.

### Methodology

Three CNN-based approaches were trained and compared: a custom CNN built from scratch, and two ImageNet-pretrained transfer-learning models (MobileNetV2 and ResNet50), each fine-tuned in two stages (frozen backbone, then partial fine-tuning). Models were evaluated on a stratified 70/15/15 train/validation/test split using accuracy, macro-averaged precision/recall/F1, and AUC, alongside deployment-relevant efficiency metrics.

To validate that models learn genuine defect features rather than spurious correlations, **Grad-CAM** and **SHAP** explainability methods were applied to visualize which image regions and pixels drive each prediction.

This interface runs **MobileNetV2**, selected for deployment because it offers the best accuracy-to-efficiency trade-off among the three models tested — comparable accuracy to the top-performing ResNet50 at roughly one-eighth the model size.

### Tech Stack

TensorFlow / Keras &middot; Gradio &middot; scikit-learn &middot; SHAP &middot; Grad-CAM &middot; Hugging Face Spaces

### Links

- GitHub Repository: https://github.com/sahilsodhi12/steel-defect-classification.git
- Kaggle Notebook: https://www.kaggle.com/code/sodhisahil/steel-classification-nb
- Full Report: https://www.overleaf.com/read/gncjmznpmnvz#a77db9
                """
            )

        with gr.TabItem("Model Comparison"):
            gr.Markdown("### Performance Across All Three Trained Models")
            gr.HTML(
                """
                <table style="width:100%; border-collapse: collapse; text-align:center;">
                <tr style="background:#2d3748; color:white;">
                    <th style="padding:10px;">Model</th>
                    <th style="padding:10px;">Accuracy</th>
                    <th style="padding:10px;">Macro F1</th>
                    <th style="padding:10px;">Model Size</th>
                    <th style="padding:10px;">Inference Time</th>
                </tr>
                <tr style="background:#1f2430;">
                    <td style="padding:8px;">Custom CNN</td>
                    <td style="padding:8px;">96.67%</td>
                    <td style="padding:8px;">0.9664</td>
                    <td style="padding:8px;">5.31 MB</td>
                    <td style="padding:8px;">7.15 ms</td>
                </tr>
                <tr style="background:#26314a; color:#fbbf24; font-weight:600;">
                    <td style="padding:8px;">MobileNetV2 (deployed here)</td>
                    <td style="padding:8px;">97.78%</td>
                    <td style="padding:8px;">0.9775</td>
                    <td style="padding:8px;">24.61 MB</td>
                    <td style="padding:8px;">29.70 ms</td>
                </tr>
                <tr style="background:#1f2430;">
                    <td style="padding:8px;">ResNet50</td>
                    <td style="padding:8px;">100.00%</td>
                    <td style="padding:8px;">1.0000</td>
                    <td style="padding:8px;">206.72 MB</td>
                    <td style="padding:8px;">28.44 ms</td>
                </tr>
                </table>
                """
            )
            gr.Markdown(
                "ResNet50 achieves the highest raw accuracy, but MobileNetV2 is deployed here as the more "
                "practical choice for real-time, resource-constrained inference."
            )

    gr.HTML(
        "<hr><center style='color:#888; font-size:0.85em;'>Built for the Machine Learning and Smart Systems Project &middot; NEU-DET Steel Surface Defect Dataset</center>"
    )

demo.launch()
