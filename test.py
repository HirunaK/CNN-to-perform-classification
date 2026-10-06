import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.model_a import ModelA
from train_model_a import FlowerDataset, find_data_dir


def main():
	parser = argparse.ArgumentParser(description="Evaluate a trained Model A on the test split.")
	parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_a.pth"))
	parser.add_argument("--data-dir", type=Path)
	parser.add_argument("--batch-size", type=int, default=32)
	args = parser.parse_args()

	project_dir = Path(__file__).resolve().parent
	data_dir = args.data_dir or find_data_dir(project_dir)
	test_dir = data_dir / "test"
	if not test_dir.is_dir():
		raise FileNotFoundError(f"Test split was not found: {test_dir}")

	checkpoint_path = args.checkpoint
	if not checkpoint_path.is_absolute():
		checkpoint_path = project_dir / checkpoint_path
	if not checkpoint_path.is_file():
		raise FileNotFoundError(
			f"Checkpoint not found: {checkpoint_path}. Run train_model_a.py first."
		)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
	class_to_idx = checkpoint["class_to_idx"]
	dataset = FlowerDataset(test_dir, class_to_idx)
	loader = DataLoader(dataset, batch_size=args.batch_size)
	model = ModelA(num_classes=len(class_to_idx)).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	loss_function = nn.CrossEntropyLoss()
	total_loss = 0.0
	total_correct = 0
	with torch.no_grad():
		for images, labels in loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			total_loss += loss_function(outputs, labels).item() * labels.size(0)
			total_correct += (outputs.argmax(dim=1) == labels).sum().item()

	accuracy = total_correct / len(dataset)
	average_loss = total_loss / len(dataset)
	print(f"Test images: {len(dataset)}")
	print(f"Test loss: {average_loss:.4f}")
	print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
	main()