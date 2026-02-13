import os
import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_inference(model_path, image_path):
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.resize(img, (224, 224))
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocessing: Normalize to [-1, 1] and HWC -> CHW
    input_tensor = (input_img.astype(np.float32) - 127.5) / 128.0
    input_tensor = input_tensor.transpose(2, 0, 1) # HWC -> CHW
    input_tensor = input_tensor[np.newaxis, ...] # Add batch dim -> NCHW
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    output = outputs[0] # (1, 3, 222, 222)
    
    # Postprocessing
    output = output[0].transpose(1, 2, 0) # CHW -> HWC
    
    # Normalize for display: shift to positive and scale
    # Taking absolute value as edge detection often produces negative values
    output_disp = np.abs(output)
    output_disp = (output_disp * 255.0 / output_disp.max()).astype(np.uint8)

    # Visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("ONNX Output (Sobel)")
    plt.imshow(output_disp)
    plt.axis('off')
    
    plt.tight_layout()
    # Save relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "out", "onnx_result.png")
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
    
    # Try to show if possible (might fail in headless but saving works)
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display window: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "sobel_int8.onnx")
    image_path = os.path.join(base_dir, "out", "test.jpg")
    
    try:
        run_inference(model_path, image_path)
    except Exception as e:
        print(f"Error: {e}")
