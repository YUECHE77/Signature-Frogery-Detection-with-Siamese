import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, data_dir, data_csv, transform=None, use_part=False, num_data=4000):
        """
        :param data_dir: The directory path of data (images) -> it is a string
        :param data_csv: The csv file define data pairs -> it is a dataframe
        :param transform: Do transform or not
        :param use_part: Use only part of the dataset
        :param num_data: The number of sub-dataset
        """
        super(SiameseDataset, self).__init__()

        self.data = data_csv
        if use_part:
            self.data = self.only_use_partial(num_data=num_data)

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img1_path = os.path.join(self.data_dir, self.data.iat[index, 0])
        img2_path = os.path.join(self.data_dir, self.data.iat[index, 1])

        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
        except IOError:
            print(f"Error opening one of the images at index {index}")
            return None

        img1 = img1.convert('L')  # Mode L -> reduce the number of channels to 1 -> gray images
        img2 = img2.convert('L')  # Mode RGB -> do nothing -> color images

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.from_numpy(np.array([int(self.data.iat[index, 2])], dtype=np.float32))
        # label = torch.tensor([int(self.train_data.iat[idx, 2])], dtype=torch.float32)

        return img1, img2, label

    def only_use_partial(self, num_data=4000):
        """num_data: The size of your subset"""
        subset_index = np.random.choice(len(self.data), size=num_data, replace=False)

        subset_data = self.data.loc[subset_index].reset_index(drop=True)
        return subset_data
