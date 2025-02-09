import torch
import torchvision

from build_model import *

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

    testData = torchvision.datasets.MNIST("./dataset", train=False, transform=transform)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=256, shuffle=False)

    model = Net().to(device)
    model.load_state_dict(torch.load("./models/final_model.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，减少内存占用并加速计算
        for inputs, labels in testDataLoader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到目标设备（CPU或GPU）

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")

