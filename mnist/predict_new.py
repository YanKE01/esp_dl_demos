import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义与训练时一致的转换操作
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保图像是单通道灰度图
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 根据训练时使用的归一化参数调整
])


# 加载模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 6 * 64, out_features=256),  # 根据输入尺寸计算
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=256, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.model(x)
        return output


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Net().to(device)
model.load_state_dict(torch.load('./models/final_model.pth', map_location=device))
model.eval()


# 图像预处理和预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image_array = np.array(image)
    image_array[image_array == 255] = 1
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度(batch_size=1)，并移动到目标设备
    input_numpy = input_tensor.squeeze(0)
    input_numpy = input_numpy.cpu().detach().numpy()

    print(input_numpy.shape)
    c_array_code = "float input_data[{}][{}] = {{\n".format(input_numpy.shape[1], input_numpy.shape[2])  # 修改这里交换宽高
    for i in range(input_numpy.shape[1]):  # 遍历高度
        c_array_code += "    {"
        for j in range(input_numpy.shape[2]):  # 遍历宽度
            c_array_code += "{}".format(int(input_numpy[0][i][j]))  # 确保这里的索引与形状匹配
            if j < input_numpy.shape[2] - 1:
                c_array_code += ", "
        c_array_code += "}"
        if i < input_numpy.shape[1] - 1:
            c_array_code += ",\n"
    c_array_code += "\n};"
    print(c_array_code)

    with torch.no_grad():  # 关闭梯度计算以节省内存和加速计算
        output = model(input_tensor)

    print(output)
    predicted_class = output.argmax(dim=1).item()
    return predicted_class


# 示例：对一张新图像进行预测
image_path = './dataset/extra/9/20250225_140331.png'
predicted_label = predict_image(image_path)
print(f"预测的类别是: {predicted_label}")
