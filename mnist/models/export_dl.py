import torch
import torchvision
from ppq.api import espdl_quantize_torch
from torch.utils.data import Dataset

from mnist.build_model import Net

DEVICE = "cpu"


class FeatureOnlyMNIST(Dataset):
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
    INPUT_SHAPE = [1, 28, 28]
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    ESPDL_MODLE_PATH = "./s3/mnist.espdl"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    testData = torchvision.datasets.MNIST("../dataset", train=False, transform=transform)
    test_input, test_label = testData[100]
    test_input = test_input.unsqueeze(0)
    print(test_input.shape)
    print(f"Result:{test_label}")
    feature_only_test_data = FeatureOnlyMNIST(testData)

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
        inputs=[test_input],
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        dispatching_override=None
    )
