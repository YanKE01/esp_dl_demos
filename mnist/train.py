import torch
import torchvision

from build_model import Net

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

    trainData = torchvision.datasets.MNIST("./dataset", train=True, transform=transform, download=True)
    testData = torchvision.datasets.MNIST("./dataset", train=False, transform=transform)
    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=256, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=256, shuffle=False)

    net = Net().to(device)
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    best_accuracy = 0.0
    num_epochs = 100

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for batch_features, batch_targets in trainDataLoader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            outputs = net(batch_features)
            loss = lossF(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainDataLoader)}")

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_features, test_targets in testDataLoader:
                test_features, test_targets = test_features.to(device), test_targets.to(device)
                outputs = net(test_features)
                _, predicted = torch.max(outputs.data, 1)
                total += test_targets.size(0)
                correct += (predicted == test_targets).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the test images: {accuracy}%")

        # 如果当前模型在测试集上的准确率是最优的，则保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), "./models/best_model.pth")
            print(f"New best model saved with accuracy: {best_accuracy}%")

    torch.save(net.state_dict(), "./models/final_model.pth")
