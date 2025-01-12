import json

import numpy as np
import pandas as pd
import torch
from ppq.api import espdl_quantize_torch
from torch.utils.data import DataLoader

from boston.build_model import *
from boston.dataset import *


def load_scaling_params(json_path):
    """从 JSON 文件中加载缩放参数"""
    with open(json_path, 'r') as f:
        scaling_params = json.load(f)
    min_vals = np.array(scaling_params['min'])
    max_vals = np.array(scaling_params['max'])

    return min_vals, max_vals


def apply_min_max_scale(features, min_vals, max_vals):
    """使用预定义的最小值和最大值对特征数据进行缩放"""
    range_vals = max_vals - min_vals
    # 防止除以零的情况发生
    range_vals[range_vals == 0] = 1.0

    scaled_features = (features - min_vals) / range_vals
    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=0.0)

    return scaled_features


def load_and_preprocess_data(df, scaling_params_path):
    # 确保 DataFrame 中有 'medv' 列作为目标变量
    if 'medv' not in df.columns:
        raise ValueError("DataFrame must contain a 'medv' column for the target variable.")

    features = df.drop(columns=['medv']).values
    targets = df['medv'].values

    # 加载缩放参数
    min_vals, max_vals = load_scaling_params(scaling_params_path)

    # 使用预定义的最小值和最大值对特征数据进行缩放
    features_scaled = apply_min_max_scale(features, min_vals, max_vals)

    return features_scaled, targets


def collate_fn2(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


if __name__ == '__main__':
    BATCH_SIZE = 32
    INPUT_SHAPE = [13]
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
    TARGET = "esp32p4"
    NUM_OF_BITS = 8
    ESPDL_MODLE_PATH = "p4/house_price.espdl"

    model = BostonHousingModel(13)
    model.load_state_dict(torch.load("./boston_house_price.pth"))
    model.eval()

    scaling_params_path = '../scaling_params.json'

    df = pd.read_csv("../BostonHousing.csv")

    features_scaled, targets = load_and_preprocess_data(df, scaling_params_path)
    print(features_scaled[1])

    test_data = torch.tensor(features_scaled[1], dtype=torch.float32).unsqueeze(0)
    print(test_data.shape)

    features_scaled = torch.tensor(features_scaled, dtype=torch.float32)

    dataset = BostonHousingDatasetFeature(features_scaled)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 原始数据
    test_data = np.array(
        [2.35922539e-04, 0.00000000e+00, 2.42302053e-01, 0.00000000e+00, 1.72839506e-01, 5.47997701e-01, 7.82698249e-01,
         3.48961980e-01, 4.34782609e-02, 1.04961832e-01, 5.53191489e-01, 1.00000000e+00, 2.04470199e-01])
    test_data = test_data.reshape(1, -1)  # 转换为 (1, 13) 形状

    # 转换为 torch tensor
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # 将 test_data 放入列表中
    inputs = [test_data]

    try:
        quant_ppq_graph = espdl_quantize_torch(
            model=model,
            espdl_export_file=ESPDL_MODLE_PATH,
            calib_dataloader=data_loader,
            calib_steps=8,
            input_shape=[1] + INPUT_SHAPE,
            inputs=inputs,
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
    except RuntimeError as e:
        print(f"Error: {e}")
        print(f"Inputs shape: {test_data.shape}")
        print(f"Inputs: {test_data}")
        raise
