import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保所有图像都是单通道灰度图
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # 随机旋转平移
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 根据需要调整归一化参数
])

# 加载自定义数据集
dataset = datasets.ImageFolder(root='./dataset/extra', transform=transform)

# 统计每个类别的样本数
class_counts = Counter([label for _, label in dataset])

# 获取类别名称
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = [idx_to_class[i] for i in class_counts.keys()]
counts = list(class_counts.values())

# 绘图
plt.figure(figsize=(10, 6))
plt.bar(class_names, counts)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Classes in Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # 定义类别数量和布局
# num_classes = 10
# rows, cols = 2, 5
#
# # 用于存储每个类别的一张图片
# class_images = [None] * num_classes
#
# # 遍历数据集并收集每个类别的一张图像
# for img, label in dataset:
#     if class_images[label] is None:
#         class_images[label] = img
#     if all(img is not None for img in class_images):
#         break
#
# # 创建图像展示（2行5列）
# fig, axes = plt.subplots(rows, cols, figsize=(12, 5))
# for i in range(num_classes):
#     row = i // cols
#     col = i % cols
#     ax = axes[row, col]
#     image = class_images[i].squeeze()
#     ax.imshow(image, cmap='gray')
#     ax.set_title(f"Class {i}")
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()
#
#
# # 划分训练集和测试集
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#
# train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#
#             nn.Flatten(),
#             nn.Linear(in_features=7 * 6 * 64, out_features=256),  # 根据输入尺寸计算
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(in_features=256, out_features=len(dataset.classes)),  # 输出层大小为类别数
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         output = self.model(x)
#         return output
#
#
# # 初始化模型、损失函数和优化器
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = Net().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # 训练和测试函数
# def train_epoch(model, train_loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     epoch_loss = running_loss / len(train_loader)
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc
#
#
# def test_epoch(model, test_loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     epoch_loss = running_loss / len(test_loader)
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc
#
#
# # 训练循环
# num_epochs = 100
# train_acc_array = []
# test_acc_array = []
# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#     test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], '
#           f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
#           f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
#     train_acc_array.append(train_acc)
#     test_acc_array.append(test_acc)
#
# # 保存模型
# torch.save(model.state_dict(), './models/final_model.pth')
#
#
# plt.figure(figsize=(10, 5))
# plt.plot(train_acc_array, label='Train Accuracy', color='blue')
# plt.plot(test_acc_array, label='Test Accuracy', color='red')
# plt.title('Train and Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.show()
