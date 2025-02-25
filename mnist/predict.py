import torch
import torchvision
from PIL import Image

from build_model import *

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((30, 25)),  # 调整大小
        torchvision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        torchvision.transforms.Lambda(lambda x: x.convert('L')),  # 转换为灰度图
        torchvision.transforms.Lambda(lambda x: Image.eval(x, lambda px: 255 if px > 127.5 else 0)),  # 二值化处理
        torchvision.transforms.ToTensor(),  # 转换为tensor
    ])
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

