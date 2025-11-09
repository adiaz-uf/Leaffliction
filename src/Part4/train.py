import os
import sys
import numpy as np
import pandas as pd


def augment_data(dir_info, max_images):
    augmentations_per_image = (max_images // dir_info['len']) - 1
    pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Error! Usage is: ./train.py <folder_path>')
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print('Error! Path must be a directory')
        sys.exit(2)

    values = []
    max_images = 1000  # minimum number of images per disease category

    """ get files in each subdirectory """
    for root, dirs, files in os.walk(folder_path):
        if len(files) > 0:
            values.append(
                {
                    "dir": os.path.split(root)[1],
                    "full_path": root,
                    "files": files,
                    "len": len(files)
                }
            )
            if len(files) > max_images:
                max_images = len(files)

    if len(values) == 0:
        print('Error! No files found in the specified directory')
        sys.exit(3)

    for item in values:
        augment_data(item, max_images)
        print(
            f"Directory: {item['dir']}, Number of files: {len(item['files'])}")
