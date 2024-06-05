import os
import numpy as np
import pandas as pd
import random
import itertools

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from .SiameseDataset import SiameseDataset


def generate_dataset(base_path, num_folder=15, num_cross_samples=2500, file_type='.jpg'):
    data = []
    cross_samples = []

    # 获取所有姓名编号的文件夹路径（扩展到15个）
    true_paths = [os.path.join(base_path, f'{name_id:03}') for name_id in range(1, num_folder+1)]
    forg_paths = [os.path.join(base_path, f'{name_id:03}_forg') for name_id in range(1, num_folder+1)]

    for i, true_path in enumerate(true_paths):
        if not os.path.exists(true_path):
            print(f"True path does not exist: {true_path}")
            continue

        true_files = [f for f in os.listdir(true_path) if f.endswith(file_type)]

        # 生成负样本：同一文件夹内的两两配对
        for true_file1, true_file2 in itertools.combinations(true_files, 2):
            true_file1_path = os.path.join(true_path, true_file1)
            true_file2_path = os.path.join(true_path, true_file2)
            data.append([true_file1_path, true_file2_path, 0])

        # 生成正样本：同一文件夹内的真实签名与伪造签名配对
        forg_path = forg_paths[i]
        if not os.path.exists(forg_path):
            print(f"Forged path does not exist: {forg_path}")
            continue

        forg_files = [f for f in os.listdir(forg_path) if f.endswith(file_type)]
        for true_file in true_files:
            true_file_path = os.path.join(true_path, true_file)
            for forg_file in forg_files:
                forg_file_path = os.path.join(forg_path, forg_file)
                data.append([true_file_path, forg_file_path, 1])

        # 生成交叉正样本：与其他文件夹配对
        for j, other_true_path in enumerate(true_paths):
            if i == j or not os.path.exists(other_true_path):
                continue

            other_true_files = [f for f in os.listdir(other_true_path) if f.endswith(file_type)]
            for true_file in true_files:
                true_file_path = os.path.join(true_path, true_file)
                for other_true_file in other_true_files:
                    other_true_file_path = os.path.join(other_true_path, other_true_file)
                    cross_samples.append([true_file_path, other_true_file_path, 1])

            other_forg_path = forg_paths[j]
            if not os.path.exists(other_forg_path):
                continue

            other_forg_files = [f for f in os.listdir(other_forg_path) if f.endswith(file_type)]
            for true_file in true_files:
                true_file_path = os.path.join(true_path, true_file)
                for other_forg_file in other_forg_files:
                    other_forg_file_path = os.path.join(other_forg_path, other_forg_file)
                    cross_samples.append([true_file_path, other_forg_file_path, 1])

    # 随机选择指定数量的交叉正样本
    if len(cross_samples) > num_cross_samples:
        cross_samples = random.sample(cross_samples, num_cross_samples)

    data.extend(cross_samples)

    return data


def create_dataset_csv_file(dataset, save_path):
    """
    :param dataset: The output of function "generate_dataset"
    :param save_path: The path to save the csv file
    :return: The DataFrame
    """
    df = pd.DataFrame(dataset, columns=['Image1', 'Image2', 'Label'])
    df.to_csv(save_path, index=False)

    return df


def load_dataset(data_dir, data_csv, transform=None, use_part=False, num_data=4000, train_size=0.7, test_size=0.15, batch_size=64):
    """
    :param data_dir: The directory path of data (images) -> it is a string
    :param data_csv: The csv file define data pairs -> it is a dataframe
    :param transform: Do transform or not
    :param use_part: Use part of the dataset or not
    :param num_data: If use_part -> how many you want to use
    :param train_size: proportion of training data
    :param test_size: proportion of test data
    :param batch_size: batch size
    :return: The data loaders
    """
    dataset = SiameseDataset(data_dir=data_dir, data_csv=data_csv, transform=transform, use_part=use_part, num_data=num_data)

    train_num = int(train_size * len(dataset))
    test_num = int(test_size * len(dataset))
    val_num = len(dataset) - train_num - test_num

    train_data, val_data, test_data = random_split(dataset, [train_num, val_num, test_num])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def visualize_dataloader_shape(dataloader):
    """Visualize the shape of dataloaders"""
    img1, img2, label = next(iter(dataloader))

    print(f'The size of Image1 is {img1.shape}')
    print(f'The size of Image2 is {img2.shape}')
    print(f'The size of Label is {label.shape}')


def get_images_and_label(dataloader, index):
    """Get specific pair of images and corresponding label in a dataloader"""
    img1, img2, label = next(iter(dataloader))

    return img1[index].numpy(), img2[index].numpy(), label[index].item()


def visualize_images(img1, img2, label, gray=True, normalized=False):
    """Visualize the images"""

    img1_single = img1.transpose(1, 2, 0)
    img2_single = img2.transpose(1, 2, 0)

    if normalized:
        if gray:
            img1_single = img1_single * np.array([0.5], dtype=float) + np.array([0.5], dtype=float)
            img2_single = img2_single * np.array([0.5], dtype=float) + np.array([0.5], dtype=float)
            img1_single = np.clip(img1_single, 0, 1)
            img2_single = np.clip(img2_single, 0, 1)
        else:
            img1_single = img1_single * np.array([0.229, 0.224, 0.225], dtype=float) + np.array([0.485, 0.456, 0.406], dtype=float)
            img2_single = img2_single * np.array([0.229, 0.224, 0.225], dtype=float) + np.array([0.485, 0.456, 0.406], dtype=float)
            img1_single = np.clip(img1_single, 0, 1)
            img2_single = np.clip(img2_single, 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img1_single, cmap='gray')
    ax[0].axis('off')

    ax[1].imshow(img2_single, cmap='gray')
    ax[1].axis('off')

    fig.suptitle(f'Label: {label}')

    plt.show()


def set_requires_grad(siamese, stop_layer_):
    """Set up requires_grad for parameters -> used for transfer learning"""

    def freeze_before(model, stop_layer):
        """
        Freeze all the layers before stop layer
        :param model: Should be Siamese.vgg in this case
        :param stop_layer: An integer
        """
        frozen = True

        for index, child in enumerate(model.children()):
            if index == stop_layer:
                frozen = False
            for param in child.parameters():
                param.requires_grad = not frozen

    freeze_before(model=siamese.vgg, stop_layer=stop_layer_)

    # Unfreeze all the fully connected layers
    for param in siamese.fully_connect1.parameters():
        param.requires_grad = True
    for param in siamese.fully_connect2.parameters():
        param.requires_grad = True


def show_net(model):
    for index, child in enumerate(model.children()):
        print('\n', index, child)


