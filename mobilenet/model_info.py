import torch
import torchvision
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from PIL import Image

DEVICE = "cpu"
model = torchvision.models.mobilenet.mobilenet_v2(
    weights=MobileNet_V2_Weights.IMAGENET1K_V1
)

model.classifier = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

# Load and preprocess image
img_path = 'mobilenet/cat.jpg'
try:
    img = Image.open(img_path)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(input_tensor)

    print(f"Input image: {img_path}")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector (first 20 elements):")
    print(features[0][:20])
    
    # Save features to a file if needed for comparison later
    # torch.save(features, 'mobilenet/cat_features.pt')

except FileNotFoundError:
    print(f"Error: {img_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
