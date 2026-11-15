import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from model import DigitClassifierCNN

# Constants for normalization used during MNIST training
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Load the trained model weights
def load_trained_model(model_path="model.pt", device=None):
    if device is None:
        device = torch.device("cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"!! Model weights at '{model_path}' not found. Run train.py !!"
        )

    model = DigitClassifierCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device

def preprocess_image(image: Image.Image) -> torch.Tensor:

    # Convert image to grayscale
    image = image.convert("L")

    # Find the bounding box of the drawn digit
    # Crop the canvas accordingle
    bbox = image.getbbox()
    if bbox is None: # Canvas is empty
        return torch.zeros((1, 1, 28, 28))
    digit_cropped = image.crop(bbox)

    # Resize the bounding box to fit inside a 20x20 box but preserving the aspect ratio
    w, h = digit_cropped.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit_resized = digit_cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Place resized digit into a 28x28 black background centered
    canvas_28 = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas_28.paste(digit_resized, (paste_x, paste_y))

    # Transform to normalized tensor like train.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    tensor = transform(canvas_28)
    return tensor.unsqueeze(0)  # Shape (1, 1, 28, 28)


# Predicts the digit from the given Image and returns the digit [0 - 9]
# and a confidence in float for each digit
def predict(image: Image.Image, model=None, device=None, model_path="model.pt"):

    if model is None:
        model, device = load_trained_model(model_path=model_path, device=device)

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
        predicted_digit = int(torch.argmax(outputs, dim=1).item())

    return predicted_digit, probabilities


# Incase someone wants to predict the digit from a file for some reason
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image_file>")
        print("Disclaimer: Model will only work with white ink on black canvas for now!")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path)
    digit, probs = predict(img)
    print(f"Predicted Digit: {digit} (Confidence: {probs[digit] * 100:.2f}%)")
    print("Probabilities per class:")
    for d, p in enumerate(probs):
        print(f"  {d}: {p * 100:5.2f}%")
