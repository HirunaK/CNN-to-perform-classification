import argparse
from pathlib import Path

import cv2
import torch

from models.model_a import ModelA


def main():
    parser = argparse.ArgumentParser(description="Predict the flower class in an image.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to a flower image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/model_a.pth"),
        help="Path to the trained Model A checkpoint",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    image_path = args.image or Path(input("Enter the flower image path: ").strip().strip('"'))
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_dir / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}. Run train_model_a.py first."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    image_size = checkpoint.get("image_size", 64)
    image = cv2.resize(image, (image_size, image_size))
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    model = ModelA(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    predicted_index = probabilities.argmax().item()
    predicted_class = idx_to_class[predicted_index]
    confidence = probabilities[predicted_index].item()
    print(f"Flower: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
