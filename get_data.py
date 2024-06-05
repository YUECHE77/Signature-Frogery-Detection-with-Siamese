import utils.utils as utils
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   base_path: The folder contains 'xxx' and 'xxx_forg'
    #   num_folder: How many pairs of folders do you have
    #   num_cross_samples: Num of cross samples
    #   file_type: Type of your images

    #   output_path: Path to save the final csv file
    # ----------------------------------------------------#
    base_path = 'dataset/Chinese_no_kuang_data'
    num_folder = 15
    num_cross_samples = 2500
    file_type = '.jpg'

    output_path = 'dataset/More_data.csv'

    # ----------------------------------------------------#
    #   Generate data points -> df
    #   I keep everything default here
    #   Read generate_dataset in utils.utils for more information
    # ----------------------------------------------------#
    dataset = utils.generate_dataset(base_path)
    df = pd.DataFrame(dataset, columns=['Image1', 'Image2', 'Label'])

    # ----------------------------------------------------#
    #   You need to process the format
    #   Usually, it is r'base_path\\'
    # ----------------------------------------------------#
    df['Image1'] = df['Image1'].str.replace(r'dataset/Chinese_no_kuang_data\\', '', regex=True).str.replace('\\', '/')
    df['Image2'] = df['Image2'].str.replace(r'dataset/Chinese_no_kuang_data\\', '', regex=True).str.replace('\\', '/')

    # ----------------------------------------------------#
    #   base_path: The folder contains 'xxx' and 'xxx_forg'
    # ----------------------------------------------------#
    df.to_csv(output_path, index=False)
