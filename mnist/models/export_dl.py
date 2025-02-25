import torch
from PIL import Image
from ppq.api import espdl_quantize_torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms, datasets

from mnist.build_model import Net

DEVICE = "cpu"


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


class FeatureOnlyDataset(Dataset):
    def __init__(self, original_dataset):
        self.features = []
        for item in original_dataset:
            self.features.append(item[0])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_fn2(batch):
    features = torch.stack(batch)
    return features.to(DEVICE)


if __name__ == '__main__':
    BATCH_SIZE = 32
    INPUT_SHAPE = [1, 25, 30]
    TARGET = "esp32p4"
    NUM_OF_BITS = 8
    ESPDL_MODLE_PATH = "./p4/touch_recognition.espdl"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.ImageFolder(root="../dataset/extra", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    image = Image.open("../dataset/extra/9/20250225_140331.png").convert('L')
    input_tensor = transform(image).unsqueeze(0)
    print(input_tensor)

    feature_only_test_data = FeatureOnlyDataset(test_dataset)

    testDataLoader = torch.utils.data.DataLoader(dataset=feature_only_test_data, batch_size=BATCH_SIZE, shuffle=False,
                                                 collate_fn=collate_fn2)

    model = Net().to(DEVICE)
    model.load_state_dict(torch.load("./final_model.pth", map_location=DEVICE))
    model.eval()

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=testDataLoader,
        calib_steps=8,
        input_shape=[1] + INPUT_SHAPE,
        inputs=[input_tensor],
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        dispatching_override=None
    )
