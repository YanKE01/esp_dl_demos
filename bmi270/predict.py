import re
import numpy as np
from tensorflow import keras


# Load trained model from H5
def load_model_h5(filepath):
    """
    Load the model from H5 format
    """
    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model


# Extract and process log data
def extract_and_process_data(log_file='log.txt'):
    print("Extracting and processing log data...")

    # Read log file
    with open(log_file, 'r') as file:
        content = file.read()

    # Regular expression to extract data part
    pattern = r"=== Data Collection Results ===(.*?)=== End of Data Collection ==="
    matches = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(matches)} data collections")

    # Process extracted data into numerical values
    all_data = []
    for i, match in enumerate(matches, start=1):
        print(f"Processing data collection {i}...")
        match = match.strip()  # Remove extra spaces or newlines
        data_points = match.split(",")  # Split by comma

        # Convert to float data
        data_points = [float(dp) for dp in data_points]
        all_data.append(data_points)

    # Convert to NumPy array (no standardization, consistent with training)
    data_array = np.array(all_data)
    print(f"Original data shape: {data_array.shape}")

    # Reshape data: from [samples, 1200] to [samples, 400, 3]
    # Data arrangement: gry_x1, gry_y1, gry_z1, gry_x2, gry_y2, gry_z2, ...
    # Need to reshape to: each sample has 400 timesteps, each timestep has 3 axes of data
    num_samples = data_array.shape[0]
    num_timesteps = 200  # 400 timesteps
    num_axes = 3  # x, y, z three axes

    # Reshape data: reshape 1200 data points in each row into 400x3 matrix
    X_reshaped = data_array.reshape(num_samples, num_timesteps, num_axes)
    print(f"Reshaped data shape: {X_reshaped.shape}")

    # Normalize each sample (consistent with training)
    print("Performing per-sample normalization...")
    X_normalized = np.zeros_like(X_reshaped)
    for i in range(num_samples):
        # Normalize x, y, z axes separately for each sample
        for axis in range(num_axes):
            data_axis = X_reshaped[i, :, axis]
            # Min-Max normalization: (x - min) / (max - min)
            min_val = np.min(data_axis)
            max_val = np.max(data_axis)
            print("Min:{},Max:{}".format(min_val,max_val))
            # Avoid division by zero error
            if max_val - min_val > 1e-8:
                X_normalized[i, :, axis] = (data_axis - min_val) / (max_val - min_val)
            else:
                X_normalized[i, :, axis] = data_axis - min_val

    print("Data normalization completed")

    # Convert to numpy array (Keras works with numpy arrays)
    X_array = np.array(X_reshaped, dtype=np.float32)

    return X_array, X_reshaped


# Save normalized data to C file
def save_to_c_file(X_normalized, filename="normalized_data.c"):
    """
    Save normalized data to C file format as flat [1200] arrays
    
    Args:
        X_normalized (numpy.ndarray): Normalized data array
        filename (str): Output C file name
    """
    num_samples, num_timesteps, num_axes = X_normalized.shape
    data_per_sample = num_timesteps * num_axes  # Should be 1200
    
    print(f"Saving normalized data to {filename}...")
    print(f"Data shape: {X_normalized.shape}")
    print(f"Each sample flattened to: {data_per_sample} values")
    
    with open(filename, 'w') as f:
        # Write header
        f.write("// Normalized sensor data for gesture recognition\n")
        f.write("// Generated automatically from predict.py\n")
        f.write(f"// Data shape: {num_samples} samples x {data_per_sample} values each\n")
        f.write(f"// Data arrangement: gry_x1, gry_y1, gry_z1, gry_x2, gry_y2, gry_z2, ...\n\n")
        
        # Write includes
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")
        
        # Write data array - flatten each sample to [1200]
        f.write(f"// Normalized data array - each sample is flattened to {data_per_sample} values\n")
        f.write(f"const float normalized_data[{num_samples}][{data_per_sample}] = {{\n")
        
        for sample_idx in range(num_samples):
            f.write(f"    // Sample {sample_idx}\n")
            f.write("    {")
            
            # Flatten the sample data: [400, 3] -> [1200]
            flat_data = X_normalized[sample_idx].flatten()
            
            for i, value in enumerate(flat_data):
                f.write(f"{value:.6f}f")
                if i < len(flat_data) - 1:
                    f.write(", ")
                # Add line breaks every 10 values for readability
                if (i + 1) % 10 == 0 and i < len(flat_data) - 1:
                    f.write("\n        ")
            
            f.write("}")
            if sample_idx < num_samples - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        
        # Write metadata
        f.write("// Data metadata\n")
        f.write(f"const size_t NUM_SAMPLES = {num_samples};\n")
        f.write(f"const size_t DATA_PER_SAMPLE = {data_per_sample};\n")
        f.write(f"const size_t NUM_TIMESTEPS = {num_timesteps};\n")
        f.write(f"const size_t NUM_AXES = {num_axes};\n")
        f.write(f"const size_t TOTAL_DATA_POINTS = {num_samples * data_per_sample};\n\n")
        
        # Write helper function to get data
        f.write("// Helper function to get data for a specific sample\n")
        f.write("const float* get_sample_data(size_t sample_index) {\n")
        f.write("    if (sample_index >= NUM_SAMPLES) {\n")
        f.write("        return NULL;\n")
        f.write("    }\n")
        f.write("    return normalized_data[sample_index];\n")
        f.write("}\n\n")
        
        # Write helper function to get specific timestep data
        f.write("// Helper function to get data for a specific timestep of a sample\n")
        f.write("const float* get_timestep_data(size_t sample_index, size_t timestep_index) {\n")
        f.write("    if (sample_index >= NUM_SAMPLES || timestep_index >= NUM_TIMESTEPS) {\n")
        f.write("        return NULL;\n")
        f.write("    }\n")
        f.write("    return &normalized_data[sample_index][timestep_index * NUM_AXES];\n")
        f.write("}\n")
    
    print(f"Successfully saved {num_samples} samples to {filename}")
    print(f"Total data points: {num_samples * data_per_sample}")
    print(f"File size: {num_samples * data_per_sample * 4} bytes")


# Save normalized data to header file
def save_to_h_file(X_normalized, filename="normalized_data.h"):
    """
    Save normalized data to C header file format
    
    Args:
        X_normalized (numpy.ndarray): Normalized data array
        filename (str): Output header file name
    """
    num_samples, num_timesteps, num_axes = X_normalized.shape
    data_per_sample = num_timesteps * num_axes  # Should be 1200
    
    print(f"Saving normalized data to {filename}...")
    
    with open(filename, 'w') as f:
        # Write header guard
        f.write("#ifndef NORMALIZED_DATA_H\n")
        f.write("#define NORMALIZED_DATA_H\n\n")
        
        # Write includes
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")
        
        # Write metadata constants
        f.write("// Data metadata\n")
        f.write(f"#define NUM_SAMPLES {num_samples}\n")
        f.write(f"#define DATA_PER_SAMPLE {data_per_sample}\n")
        f.write(f"#define NUM_TIMESTEPS {num_timesteps}\n")
        f.write(f"#define NUM_AXES {num_axes}\n")
        f.write(f"#define TOTAL_DATA_POINTS {num_samples * data_per_sample}\n\n")
        
        # Write function declarations
        f.write("// Function declarations\n")
        f.write("extern const float normalized_data[NUM_SAMPLES][DATA_PER_SAMPLE];\n")
        f.write("const float* get_sample_data(size_t sample_index);\n")
        f.write("const float* get_timestep_data(size_t sample_index, size_t timestep_index);\n\n")
        
        f.write("#endif // NORMALIZED_DATA_H\n")
    
    print(f"Successfully saved header file: {filename}")


# Inference function (output probabilities)
def infer(model, X_array):
    """
    Use Keras model for prediction
    """
    # X_array shape is already [batch_size, 400, 3], no additional adjustment needed

    # Use model for prediction
    probabilities = model.predict(X_array, verbose=0)

    # Get predicted class (argmax)
    predicted = np.argmax(probabilities, axis=1)

    return probabilities, predicted


def main():
    # Load model from H5 file
    model = load_model_h5("simple_1dcnn_model.h5")

    # Extract and process log data
    X_array, X_normalized = extract_and_process_data('shuttle_dataset/test')

    # Save normalized data to C files
    # print("\n=== Saving Normalized Data to C Files ===")
    # save_to_c_file(X_normalized, "normalized_data.c")
    # save_to_h_file(X_normalized, "normalized_data.h")

    # Use model for prediction
    probabilities, predictions = infer(model, X_array)

    # Output prediction results and probabilities
    for i, prob in enumerate(probabilities):
        print(
            f"Sample {i + 1} prediction probabilities: Class 0 (o): {prob[0]:.4f}, Class 1 (v): {prob[1]:.4f}, Class 2 (unknown): {prob[2]:.4f}")


if __name__ == "__main__":
    main()
