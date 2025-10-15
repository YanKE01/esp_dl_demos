import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


# Data extraction function
def extract_data():
    """
    Extract data from log_fling.csv, log_z.csv, log_fling_left.csv and log_unknown.csv files
    fling data label is 0, z data label is 1, fling_left data label is 2, unknown data label is 3
    """
    # Read data files
    o_data = pd.read_csv('shuttle_dataset/o.csv', sep=',', header=None)
    v_data = pd.read_csv('shuttle_dataset/v.csv', sep=',', header=None)
    unknown_data = pd.read_csv('shuttle_dataset/unknown.csv', sep=',', header=None)


    # Create labels
    o_label = np.zeros(o_data.shape[0], dtype=int)
    v_label = np.ones(v_data.shape[0], dtype=int)
    unknown_label = np.full(unknown_data.shape[0], 2, dtype=int)

    # Combine feature data and labels (use .values for all to ensure numpy arrays)
    X_raw = np.vstack([o_data.values, v_data.values, unknown_data.values])
    y = np.concatenate([o_label, v_label, unknown_label])

    # Display the number of samples for each class
    print(f"o samples: {o_label.shape[0]}")
    print(f"v samples: {v_label.shape[0]}")
    print(f"unknown samples: {unknown_label.shape[0]}")

    print(f"Total samples: {len(y)}")

    # Reshape data: from [samples, 1200] to [samples, 400, 3]
    # Data arrangement: gry_x1, gry_y1, gry_z1, gry_x2, gry_y2, gry_z2, ...
    # Need to reshape to: each sample has 400 timesteps, each timestep has 3 axes of data
    num_samples = X_raw.shape[0]
    num_timesteps = 200  # 400 timesteps
    num_axes = 3  # x, y, z three axes

    # Reshape data: reshape 1200 data points in each row into 400x3 matrix
    X = X_raw.reshape(num_samples, num_timesteps, num_axes)

    # Normalize each sample (per-sample normalization)
    print("Performing per-sample normalization...")
    X_normalized = np.zeros_like(X)
    for i in range(num_samples):
        # Normalize x, y, z axes separately for each sample
        for axis in range(num_axes):
            data_axis = X[i, :, axis]
            # Min-Max normalization: (x - min) / (max - min)
            min_val = np.min(data_axis)
            max_val = np.max(data_axis)
            # Avoid division by zero error
            if max_val - min_val > 1e-8:
                X_normalized[i, :, axis] = (data_axis - min_val) / (max_val - min_val)
            else:
                X_normalized[i, :, axis] = data_axis - min_val

    print("Data normalization completed")
    return X, y


# 1D CNN model definition using Keras (matching PyTorch architecture)
def create_1d_cnn_model():
    model = keras.Sequential([
        keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation='relu', input_shape=(200, 3)),
        keras.layers.MaxPooling1D(pool_size=4),

        keras.layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(pool_size=4),

        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(3, activation='softmax')
    ])

    return model


# Function to calculate and display model size
def get_model_size_info(model):
    """
    Calculate and display model size information
    """
    # Count total parameters
    total_params = model.count_params()
    
    # Calculate size in bytes (assuming float32 = 4 bytes per parameter)
    size_bytes = total_params * 4
    
    print(f"\nModel Size Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {size_bytes / (1024*1024):.2f} MB")
    print(f"Model size: {size_bytes / (1024*1024*1024):.4f} GB")
    
    return total_params, size_bytes


# Training function using Keras
def train_model(model, X_train, y_train, X_test, y_test, epochs=30):
    """
    Train the model using Keras fit method
    """
    # Compile the model with same settings as PyTorch
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Same lr as PyTorch
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print("Model Summary:")
    model.summary()

    # Train the model with same batch size as PyTorch
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=64,  # Same batch size as PyTorch
        verbose=1,
        shuffle=True
    )

    return history


def plot_training_curves(history, save_path="training_curves.png"):
    """
    Plot training loss and accuracy curves (train + validation)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    axes[0].plot(epochs, history.history['loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history.history['accuracy'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history.history['val_accuracy'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining curves saved to {save_path}")
    plt.show()


# Testing function using Keras
def test_model(model, X_test, y_test):
    """
    Test the model using Keras evaluate method
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    return test_accuracy


# Save model as H5
def save_model_h5(model, filepath):
    """
    Save the model in H5 format
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


# Load model from H5
def load_model_h5(filepath):
    """
    Load the model from H5 format
    """
    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model


def main():
    """
    Main function - extract data and perform training using Keras
    """
    print("=== BMI270 Gesture Recognition Training Script (Keras) ===")

    try:
        # Extract data
        print("\n=== Data Extraction Phase ===")
        X, y = extract_data()

        # Convert to numpy arrays (Keras works with numpy arrays)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)  # Classification task, labels are integer type

        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

        # Create model
        print("\n=== Model Creation ===")
        model = create_1d_cnn_model()

        # Display model size information
        get_model_size_info(model)

        # Train model
        print("\n=== Starting Training ===")
        history = train_model(model, X_train, y_train, X_test, y_test, epochs=300)

        # Plot training curves
        print("\n=== Plotting Training Curves ===")
        plot_training_curves(history)

        # Test model
        print("\n=== Starting Testing ===")
        test_accuracy = test_model(model, X_test, y_test)

        # Save model as H5
        print("\n=== Saving Model ===")
        save_model_h5(model, "simple_1dcnn_model.h5")
        print("Training completed!")

    except FileNotFoundError as e:
        print(f"Error: File not found {str(e)}")
    except Exception as e:
        print(f"Error occurred during data extraction: {str(e)}")


if __name__ == "__main__":
    main()
