import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Path resolution to ensure models can be imported from any location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.model_a import ModelA
from models.model_b import ModelB

def classify_image_both_models(image_path, model_a, model_b, class_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], device='cpu'):
    """
    Loads any image of any resolution, resizes it to 64x64,
    and displays predicted class probabilities for both Model A and Model B.
    """
    # 1. Load image and apply assignment-standard preprocessing (64x64 resize + normalization)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)  # Add batch dimension

    # 2. Forward pass through Model A
    model_a.eval()
    model_a.to(device)
    with torch.no_grad():
        outputs_a = model_a(input_tensor)
        probs_a = F.softmax(outputs_a, dim=1).squeeze(0).cpu().numpy()

    # 3. Forward pass through Model B
    model_b.eval()
    model_b.to(device)
    with torch.no_grad():
        outputs_b = model_b(input_tensor)
        probs_b = F.softmax(outputs_b, dim=1).squeeze(0).cpu().numpy()

    # 4. Display original image alongside both models' predictions
    fig, (ax_img, ax_a, ax_b) = plt.subplots(1, 3, figsize=(15, 4))

    # Show input image
    ax_img.imshow(raw_img)
    ax_img.set_title("Input Image")
    ax_img.axis('off')

    # Model A Bar Plot
    bars_a = ax_a.barh(class_names, probs_a * 100, color='skyblue')
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel('Probability (%)')
    ax_a.set_title('Model A (Standard CNN)')

    for bar in bars_a:
        width = bar.get_width()
        ax_a.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=9)

    # Model B Bar Plot
    bars_b = ax_b.barh(class_names, probs_b * 100, color='lightgreen')
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel('Probability (%)')
    ax_b.set_title('Model B (Lightweight Depthwise)')

    for bar in bars_b:
        width = bar.get_width()
        ax_b.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate both model architectures
    model_a = ModelA(num_classes=5)
    model_b = ModelB(num_classes=5)

    # 2. Define paths to saved weights
    weights_a_path = os.path.join(PROJECT_ROOT, "saved_models", "model_a_weights.pth")
    weights_b_path = os.path.join(PROJECT_ROOT, "saved_models", "model_b_weights.pth")

    # 3. Load Model A weights
    if os.path.exists(weights_a_path):
        model_a.load_state_dict(torch.load(weights_a_path, map_location=device))
        print("Successfully loaded Model A weights.")
    else:
        print(f"Warning: Model A weights not found at '{weights_a_path}'. Running with untrained weights.")

    # 4. Load Model B weights
    if os.path.exists(weights_b_path):
        model_b.load_state_dict(torch.load(weights_b_path, map_location=device))
        print("Successfully loaded Model B weights.")
    else:
        print(f"Warning: Model B weights not found at '{weights_b_path}'. Running with untrained weights.")

    # 5. Run prediction on target image
    test_image_path = r"C:\Users\jhdka\Pictures\Screenshots\Screenshot 2026-10-06 171902.png"
    
    if os.path.exists(test_image_path):
        classify_image_both_models(test_image_path, model_a, model_b, device=device)
    else:
        print(f"Error: Image path not found: {test_image_path}")