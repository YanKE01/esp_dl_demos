from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

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
print(f"Total number of images in the dataset: {len(dataset)}")

# 获取类别名称
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = [idx_to_class[i] for i in class_counts.keys()]
counts = list(class_counts.values())

# 设置风格
plt.style.use('ggplot')

# 绘图
plt.figure(figsize=(12, 7))
bars = plt.bar(class_names, counts, edgecolor='black')

# 设置颜色渐变（根据数量）
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height),
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 美化轴和标题
plt.xlabel('Class', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.title('Distribution of Classes in Dataset', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 显示图形
plt.show()
