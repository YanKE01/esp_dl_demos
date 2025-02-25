import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

# 下载 MNIST 数据集
mnist_dataset = MNIST(root='./dataset/MNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_image_tensor, _ = mnist_dataset[0]
mnist_image = transforms.ToPILImage()(mnist_image_tensor)

# 缩放 MNIST 图像为 30x25
mnist_image_resized = mnist_image.resize((30, 25), Image.BILINEAR)

# 设置本地图像路径
local_image_path = 'D:/project/espressif/esp_dl_demos/mnist/dataset/extra/5'

# 查找图像文件
local_image_file = next((os.path.join(local_image_path, f)
                         for f in os.listdir(local_image_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))), None)

if not local_image_file:
    raise FileNotFoundError("未在本地路径中找到图像文件。")

local_image = Image.open(local_image_file).convert('L')

# 缩放本地图像为 30x25（如有需要）
local_image_resized = local_image.resize((30, 25), Image.BILINEAR)

# 可视化对比
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(mnist_image_resized, cmap='gray')
plt.title("MNIST Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(local_image_resized, cmap='gray')
plt.title("Touchpad Interpolated Image")
plt.axis('off')

plt.tight_layout()
plt.show()
