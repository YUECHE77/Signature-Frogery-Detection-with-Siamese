import torch
import torch.nn.functional as F


def evaluate_model(data_loader, model, loss_func, device):
    """Evaluate model during training -> use val_loader"""
    total = 0
    total_loss = 0
    correct = 0

    for data in data_loader:
        img1, img2, label = data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        outputs = model(img1, img2)

        loss = loss_func(outputs, label)
        total_loss += loss.item() * len(label)

        pred = (F.sigmoid(outputs) > 0.8).type(torch.float)
        correct += (pred == label).sum().item()

        total += len(label)

    total_loss /= total
    accuracy = correct * 100 / total

    return total_loss, accuracy


def test_model(data_loader, model, device):
    """Test model performance after model training -> use test_loader"""
    total = 0
    correct = 0

    for data in data_loader:
        img1, img2, label = data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        outputs = model(img1, img2)

        pred = (F.sigmoid(outputs) > 0.9).type(torch.float)
        correct += (pred == label).sum().item()

        total += len(label)

    accuracy = correct * 100 / total
    print(f'The accuracy of the fine-tuned model on the testing set is {accuracy}')

    return accuracy
