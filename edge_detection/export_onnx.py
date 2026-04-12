import torch
import torch.onnx
from detection import Net, sobel, conv_rgb_core_sobel, conv_rgb_core_sobel_vertical, conv_rgb_core_sobel_horizontal
import numpy as np

def export_model(model_name, kernel):
    print(f"Exporting {model_name}...")
    net = Net()
    sobel(net, kernel)
    net.eval()

    # Create dummy input
    # Shape: (1, 3, 224, 224) - typical for image models
    # The user mentioned input_size = (224, 224, 3) in their snippet, which implies HWC, 
    # but PyTorch uses NCHW. The conversion happens in detection.py manually or via library.
    # We export taking standard NCHW input.
    dummy_input = torch.randn(1, 3, 224, 224)

    output_path = f"{model_name}.onnx"
    
    # Export
    torch.onnx.export(net, 
                      dummy_input, 
                      output_path, 
                      verbose=True,
                      input_names=['input'], 
                      output_names=['output'],
                      opset_version=11)
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    # Export standard sobel
    export_model("sobel_int8", conv_rgb_core_sobel)
    
    # Optional: export others if needed
    # export_model("sobel_vertical", conv_rgb_core_sobel_vertical)
    # export_model("sobel_horizontal", conv_rgb_core_sobel_horizontal)
