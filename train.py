import utils.utils as utils
from utils.utils_model import evaluate_model
from nets.Siamese import Siamese

import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   data_dir: Path to your dataset(images) folder
    #   data_csv: Path to your csv file
    # ----------------------------------------------------#
    data_dir = 'dataset/Chinese_no_kuang_data'
    data_csv = 'dataset/More_data.csv'
    # ----------------------------------------------------#
    #   Training parameters
    #   epoch_num       Epoch number
    #   batch_size      Batch size
    #   lr              Learning rate
    # ----------------------------------------------------#
    epoch_num = 5
    batch_size = 2
    lr = 1e-3
    # ----------------------------------------------------#
    #   Set up transform
    #   transform       How you process training images
    # ----------------------------------------------------#
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #     transforms.ColorJitter(contrast=0.3),  # contrast enhancement
        #     transforms.RandomRotation(15),  # rotations
        #     transforms.RandomHorizontalFlip(p=0.3),  # flips
        transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # ----------------------------------------------------#
    #   Read in the data and get the dataloaders
    # ----------------------------------------------------#
    data = pd.read_csv(data_csv)
    train_loader, test_loader, val_loader = utils.load_dataset(data_dir=data_dir, data_csv=data, transform=transform,
                                                               use_part=True, num_data=20, batch_size=batch_size)
    # ----------------------------------------------------#
    #   Get the model and put it on GPU
    # ----------------------------------------------------#
    model = Siamese()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # ----------------------------------------------------#
    #   Optimizer and Loss function
    # ----------------------------------------------------#
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCEWithLogitsLoss()
    # ----------------------------------------------------#
    #   Start training
    # ----------------------------------------------------#
    for epoch in range(epoch_num):
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for data in train_loader:
                model.train()

                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                outputs = model(img1, img2)
                loss = loss_func(outputs, label)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        with torch.no_grad():
            model.eval()

            loss, accuracy = evaluate_model(val_loader, model, loss_func, device)
            print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Accuracy: {accuracy:.3f}%')

            if (epoch + 1) % 5 == 0:
                save_path = f'logs/model_loss_{loss:.4f}.pth'
                torch.save(model.state_dict(), save_path)

    print('Finished Training')
