import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gradio as gr
import spaces

def create_model(num_classes):
    net = models.efficientnet_b0(weights=None)
    net.classifier = nn.Sequential(
        nn.Dropout(0.4), 
        nn.Linear(net.classifier[1].in_features, num_classes)
    )
    return net

CHECKPOINT_PATH = "fold_0_best.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Successfully loaded model! Classes: {class_names}")
else:
    class_names = ["Sparrow", "Bluejay", "Cardinal", "Robin", "Finch", "Crow"]
    model = None
    print("Warning: fold_0_best.pth not found.")
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@spaces.GPU
def predict_animal(image):
    if image is None:
        return "Please provide an image or take a photo.", {}
    
    if model is None:
        return "Model checkpoint not loaded on server.", {}

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    max_prob, predicted_idx = torch.max(probabilities, dim=0)
    max_prob_val = max_prob.item()

    CONFIDENCE_THRESHOLD = 0.60
    if max_prob_val < CONFIDENCE_THRESHOLD:
        warning_msg = f"Unrecognized Species (Confidence: {max_prob_val*100:.1f}%). This falls below our verification threshold."
        return warning_msg, {}

    top_class = class_names[predicted_idx.item()]
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    status_msg = f"Possible Species: {top_class.title()} ({max_prob_val*100:.1f}% match)"
    return status_msg, confidences

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Ranchers&display=swap');

/* Force uniform app background and clean margin spacing */
html, body, gradio-app {
    background-color: #1E3B2F !important;
    font-family: 'Ranchers', sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* Fix outer margins so it has a real gap top and bottom instead of gluing to screen edges */
gradio-app {
    display: flex !important;
    justify-content: center !important;
    align-items: flex-start !important;
    padding: 50px 20px 80px 20px !important;
    box-sizing: border-box !important;
}

/* Completely neutralize Gradio's default container card styling */
.gradio-container {
    max-width: 820px !important;
    width: 100% !important;
    margin: 0 auto !important;
    background-color: #7F8C6A !important;
    border-radius: 20px !important;
    box-shadow: 0 12px 35px rgba(10, 20, 15, 0.4) !important;
    padding: 36px !important;
    border: 1px solid #C8B48A !important;
    font-family: 'Ranchers', sans-serif !important;
}

.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3, 
.gradio-container p,
button.primary-btn {
    font-family: 'Ranchers', sans-serif !important;
}

.custom-status-box {
    background-color: #F5F0E6 !important;
    padding: 18px 20px !important;
    border: 1px solid #C8B48A !important;
    border-radius: 16px !important;
    font-weight: 400 !important;
    color: #1E3B2F !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    margin-bottom: 16px !important;
    word-break: break-word !important;
}

.custom-status-box *, 
.custom-status-box p, 
.custom-status-box span {
    color: #1E3B2F !important;
}

/* Fix Species Predictions box and target the big inner text / values */
#species-output {
    background-color: #F5F0E6 !important;
    border: 1px solid #C8B48A !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    color: #1E3B2F !important;
}

#species-output *,
#species-output .label-wrap,
#species-output span,
#species-output div,
#species-output .value,
#species-output [data-testid="label"] {
    color: #1E3B2F !important;
    font-family: 'Ranchers', sans-serif !important;
}

#image-input {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid #C8B48A !important;
    background-color: #F5F0E6 !important;
}

#image-input > div,
#image-input .image-container {
    border: none !important;
    border-radius: 16px !important;
    background-color: #F5F0E6 !important;
}

button.primary-btn {
    background: #C8B48A !important;
    color: #1E3B2F !important;
    border-radius: 16px !important;
    font-weight: 400 !important;
    font-size: 1rem !important;
    border: 1px solid #C8B48A !important;
    padding: 14px 24px !important;
    margin-top: 12px !important;
    box-shadow: none !important;
    cursor: pointer !important;
}

button.primary-btn:hover {
    background: #d4be95 !important;
    transform: none !important;
}
"""

with gr.Blocks() as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 2.5rem;">
            <h1 style="font-size: 2.6rem; margin: 0 0 10px 0; color: #1E3B2F !important; font-weight: 400; font-family: 'Ranchers', sans-serif;">Wetlands Birds Detector</h1>
            <p style="color: #F5F0E6 !important; font-size: 1.1rem; font-weight: 400; margin: 0; line-height: 1.5; font-family: 'Ranchers', sans-serif;">
                Snap a photo or upload an image of a bird species the wetlands ecosystem for identification.
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                sources=["webcam", "upload"], 
                type="pil", 
                label="Capture or Upload Species",
                elem_id="image-input"
            )
            submit_btn = gr.Button("Identify Bird", elem_classes=["primary-btn"])
            
        with gr.Column(scale=1):
            status_output = gr.HTML(value='<div class="custom-status-box">Awaiting species input...</div>')
            output_label = gr.Label(
                num_top_classes=3, 
                label="Species Predictions",
                elem_id="species-output"
            )

    submit_btn.click(
        fn=predict_animal, 
        inputs=input_image, 
        outputs=[status_output, output_label]
    )

if __name__ == "__main__":
    demo.launch(css=custom_css)
