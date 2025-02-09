import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from HAR.build_model import *


def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    features = train_df.columns[:-2]
    target = 'Activity'

    label_to_index = {label: idx for idx, label in enumerate(train_df[target].astype('category').cat.categories)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    train_df[target] = train_df[target].map(label_to_index)
    test_df[target] = test_df[target].map(label_to_index)

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, mean, std, label_to_index, index_to_label


if __name__ == '__main__':
    train_loader, test_loader, mean, std, label_to_index, index_to_label = load_and_preprocess_data(
        "./dataset/train.csv",
        "./dataset/test.csv")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    model = HARModel().to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    num_epochs = 100

    # 训练和验证循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印每个epoch的平均损失
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_inputs, test_labels in test_loader:
                test_inputs, test_labels = test_inputs.to(DEVICE), test_labels.to(DEVICE)
                outputs = model(test_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the validation data: {accuracy:.2f}%")

    torch.save(model.state_dict(), "models/final_model_1.pth")
