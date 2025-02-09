from train import *

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, mean, std, label_to_index, index_to_label = load_and_preprocess_data(
        "./dataset/train.csv",
        "./dataset/test.csv")

    model = HARModel().to(device)
    model.load_state_dict(torch.load("models/final_model_1.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，减少内存占用并加速计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到目标设备（CPU或GPU）
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")
