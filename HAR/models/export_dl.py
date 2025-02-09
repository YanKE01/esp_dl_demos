import pandas as pd
import torch
from ppq.api import espdl_quantize_torch
from torch.utils.data import Dataset, DataLoader

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

    return X_test, y_test


class HARFeature(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


BATCH_SIZE = 1
DEVICE = "cpu"
TARGET = "esp32s3"
NUM_OF_BITS = 8
input_shape = [561]
ESPDL_MODLE_PATH = "./s3/har.espdl"


def collate_fn2(batch):
    return batch.to(DEVICE)


if __name__ == '__main__':
    x_test, y_test = load_and_preprocess_data(
        "../dataset/train.csv",
        "../dataset/test.csv")

    test_input = x_test[500]
    test_input = test_input.unsqueeze(0)
    print("Test label:", y_test[500])

    model = HARModel()
    model.load_state_dict(torch.load("final_model_1.pth", map_location="cpu"))
    model.eval()

    har_dataset = HARFeature(x_test)
    har_dataloader = DataLoader(har_dataset, batch_size=BATCH_SIZE, shuffle=False)

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=har_dataloader,
        calib_steps=8,
        input_shape=[1] + input_shape,
        inputs=[test_input],
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn2,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        dispatching_override=None
    )
