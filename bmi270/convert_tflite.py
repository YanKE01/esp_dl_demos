import os

import tensorflow as tf
from tensorflow import keras


def convert_keras_to_tflite(model_path, output_path="simple_1dcnn_model.tflite", quantization_type="hybrid"):
    """
    Convert Keras model to TensorFlow Lite format with different quantization options
    
    Args:
        model_path (str): Path to the Keras model file (.h5)
        output_path (str): Path for the output TensorFlow Lite model (.tflite)
        quantization_type (str): Type of quantization - "hybrid" (weights INT8, I/O float32) or "full_int8" (all INT8)
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            return False

        print(f"Loading Keras model from: {model_path}")

        # Load the Keras model
        model = keras.models.load_model(model_path)

        # Print model summary
        print("Model Summary:")
        model.summary()

        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Configure quantization based on type
        if quantization_type == "hybrid":
            # Hybrid quantization: weights INT8, input/output float32
            print("Using hybrid quantization (weights INT8, input/output float32)")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]  # Keep input/output as float32
            converter.inference_input_type = tf.float32  # Input type
            converter.inference_output_type = tf.float32  # Output type
        elif quantization_type == "full_int8":
            # Full INT8 quantization: all operations use INT8
            print("Using full INT8 quantization (all operations INT8)")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            # No quantization
            print("No quantization applied")
            pass

        # Convert the model
        tflite_model = converter.convert()

        # Save the converted model
        with open(output_path, "wb") as f:
            flatbuffer_size = f.write(tflite_model)

        print(f"Conversion completed successfully!")
        print(f"TensorFlow Lite model saved to: {output_path}")
        print(f"The size of the converted flatbuffer is: {flatbuffer_size} bytes")
        print(f"Size in KB: {flatbuffer_size / 1024:.2f} KB")
        print(f"Size in MB: {flatbuffer_size / (1024 * 1024):.2f} MB")

        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def test_tflite_model(tflite_path, test_data=None):
    """
    Test the converted TensorFlow Lite model
    
    Args:
        tflite_path (str): Path to the TensorFlow Lite model
        test_data (numpy.ndarray): Test data for inference (optional)
    """
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\nTensorFlow Lite Model Details:")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")

        if test_data is not None:
            print(f"\nTesting with provided data...")
            print(f"Test data shape: {test_data.shape}")

            # Set input data
            interpreter.set_tensor(input_details[0]['index'], test_data)

            # Run inference
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(f"Output shape: {output_data.shape}")
            print(f"Sample output: {output_data[0] if len(output_data) > 0 else 'No output'}")

        return True

    except Exception as e:
        print(f"Error testing TFLite model: {str(e)}")
        return False


def main():
    """
    Main function to convert Keras model to TensorFlow Lite with different quantization options
    """
    print("=== Keras to TensorFlow Lite Converter ===")

    # Model paths
    keras_model_path = "simple_1dcnn_model.h5"

    # Convert with hybrid quantization (weights INT8, input/output float32)
    print("\n" + "=" * 60)
    print("Converting with HYBRID quantization (weights INT8, I/O float32)")
    print("=" * 60)

    tflite_hybrid_path = "simple_1dcnn_model_hybrid.tflite"
    success_hybrid = convert_keras_to_tflite(keras_model_path, tflite_hybrid_path, "hybrid")

    if success_hybrid:
        # Test the hybrid model
        print("\n" + "=" * 50)
        test_tflite_model(tflite_hybrid_path)


if __name__ == "__main__":
    main()
