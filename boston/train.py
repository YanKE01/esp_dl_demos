import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from build_model import *
from dataset import *

if __name__ == '__main__':
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    x_train, x_test, y_train, y_test = train_test_split(
        data_scaled, target, test_size=0.2, random_state=42
    )

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    train_dataset = BostonHousingDataset(x_train, y_train)
    test_dataset = BostonHousingDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = x_train.shape[1]
    model = BostonHousingModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train
    train_losses = []
    test_losses = []
    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_features.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                epoch_test_loss += loss.item() * batch_features.size(0)
        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.4f}, '
                  f'Test Loss: {epoch_test_loss:.4f}')

    model.eval()
    with torch.no_grad():
        predictions = model(x_test).squeeze().numpy()
        actual = y_test.squeeze().numpy()

    mse = np.mean((predictions - actual) ** 2)
    rmse = np.sqrt(mse)
    print(f'\nTest MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions, c='crimson')
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlabel('Actual Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title('Actual vs Predicted Values', fontsize=20)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'b--')
    plt.show()

    torch.save(model.state_dict(), 'models/boston_house_price.pth')

    scaling_params = {
        "min": scaler.data_min_.tolist(),
        "max": scaler.data_max_.tolist()
    }

    with open('scaling_params.json', 'w') as f:
        json.dump(scaling_params, f)
