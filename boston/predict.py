import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from build_model import BostonHousingModel
from dataset import BostonHousingDataset


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


def load_model(model_path):
    input_dim = 13
    model = BostonHousingModel(input_dim)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def predict(model, data_loader):
    predictions = []
    with torch.no_grad():
        for batch_features, _ in data_loader:
            outputs = model(batch_features)
            predictions.extend(outputs.squeeze().numpy())

    return predictions


if __name__ == '__main__':
    scaling_params_path = 'scaling_params.json'

    # 或者你可以直接使用一个已经存在的 DataFrame df
    df = pd.read_csv("./BostonHousing.csv")

    features_scaled, targets = load_and_preprocess_data(df, scaling_params_path)
    print(features_scaled[1], targets[1])

    features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
    dataset = BostonHousingDataset(features_scaled, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model_path = 'models/boston_house_price.pth'
    model = load_model(model_path)

    predictions = predict(model, data_loader)

    print("Predicted vs Actual values:")
    for i in range(len(predictions)):
        print(f"Predict: {predictions[i]:.2f}, Actual: {targets[i]}")
