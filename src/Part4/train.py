import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Part2.Augmentation import make_augmentations

generated_images = 0


def augment_data(dir_info, max_images):
    global generated_images
    print(f"Augmenting {dir_info['dir']} data...")
    print(f"Needed images: {max_images}, Current images: {dir_info['len']}\n")
    needed_images = max_images - dir_info['len']
    augmented_path = os.path.join(
        './data/augmented/augmented_to_train', dir_info['dir'])

    # Copy original images to augmented directory
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)
    for file in dir_info['files']:
        src_file_path = os.path.join(dir_info['full_path'], file)
        dst_file_path = os.path.join(augmented_path, file)
        try:
            if not os.path.exists(dst_file_path):
                cv2.imwrite(dst_file_path, cv2.imread(src_file_path))
        except Exception as e:
            print(
                f'Error copying file {src_file_path}')

    if needed_images <= 0:
        return

    # Generate augmented images if needed
    for file in dir_info['files']:
        file_path = os.path.join(dir_info['full_path'], file)
        result = make_augmentations(file_path, verbose=False)
        if result is None:
            continue
        if needed_images <= 0:
            break

        try:
            for i, image_name in enumerate(result):
                if image_name == "Original":
                    continue
                if needed_images <= 0:
                    return
                save_path = os.path.join(
                    augmented_path, f'{os.path.splitext(file)[0]}_{image_name}.jpg')
                cv2.imwrite(save_path, result[image_name])
                needed_images -= 1
                generated_images += 1
        except Exception as e:
            print(f'Error saving file {file_path}: {e}')


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

    """ augment data to balance number of images """
    print("-- Starting data augmentation... --\n")
    for item in values:
        augment_data(item, max_images)
    print(f'New generated images total: {generated_images}')
