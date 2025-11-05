import sys
import os
import cv2
import random
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

# Data path
data_path = './data/original/'
# Library const values
border_mode = border_mode = cv2.BORDER_CONSTANT
border_value = (0, 0, 0)


def flip_image(image):
    flip_type = random.choice(['horizontal', 'vertical', 'both'])

    if flip_type == "both":
        return cv2.flip(image, -1)
    elif flip_type == "horizontal":
        return cv2.flip(image, 1)
    elif flip_type == "vertical":
        return cv2.flip(image, 0)


def rotate_image(image):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderMode=border_mode,
        borderValue=border_value
    )
    return rotated


def skew_image(image):
    height, width = image.shape[:2]
    skew_type = random.choice(['horizontal', 'vertical', 'both'])

    skew_amount = random.uniform(0.1, 0.3)

    if skew_type == 'horizontal' or skew_type == 'both':
        pts1 = np.float32([[0, 0], [width, 0], [0, height]])
        pts2 = np.float32(
            [[int(width * skew_amount), 0], [width, 0], [0, height]])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (width, height),
                               borderMode=border_mode, borderValue=border_value)

    if skew_type == 'vertical' or skew_type == 'both':
        pts1 = np.float32([[0, 0], [width, 0], [0, height]])
        pts2 = np.float32(
            [[0, int(height * skew_amount)], [width, 0], [0, height]])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (width, height),
                               borderMode=border_mode, borderValue=border_value)

    return image


def shear_image(image):
    height, width = image.shape[:2]
    shear_amount = random.uniform(0.1, 0.3)

    direction = random.choice(['horizontal', 'vertical'])

    if direction == 'horizontal':
        M = np.float32([
            [1, shear_amount, 0],
            [0, 1, 0]
        ])
    else:
        M = np.float32([
            [1, 0, 0],
            [shear_amount, 1, 0]
        ])

    sheared = cv2.warpAffine(image, M, (width, height),
                             borderMode=border_mode, borderValue=border_value)

    return sheared


def crop_image(image):
    height, width = image.shape[:2]

    crop_height = int(height * random.uniform(0.8, 0.9))
    crop_width = int(width * random.uniform(0.8, 0.9))

    y_start = random.randint(0, height - crop_height)
    x_start = random.randint(0, width - crop_width)

    cropped = image[y_start:y_start +
                    crop_height, x_start:x_start + crop_width]

    resized = cv2.resize(cropped, (width, height),
                         interpolation=cv2.INTER_LINEAR)

    return resized


def distort_image(image):
    alpha = 8
    sigma = 4

    height, width = image.shape[:2]

    dx = gaussian_filter(
        (np.random.rand(height, width) * 2 - 1), sigma) * alpha
    dy = gaussian_filter(
        (np.random.rand(height, width) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    if len(image.shape) == 3:
        distorted = np.copy(image)
        for channel in range(image.shape[2]):
            distorted[:, :, channel] = map_coordinates(
                image[:, :, channel],
                indices,
                order=1,
                mode='reflect'
            ).reshape(image.shape[:2])
    else:
        distorted = map_coordinates(
            image,
            indices,
            order=1,
            mode='reflect'
        ).reshape(image.shape)

    return distorted.astype(np.uint8)


def main(filePath):
    print(f'Processing file: {filePath}')
    image = cv2.imread(filePath)

    all_images = {}

    # Image augmentations
    all_images["Flip"] = flip_image(image)

    all_images["Rotate"] = rotate_image(image)

    all_images["Skew"] = skew_image(image)

    all_images["Shear"] = shear_image(image)

    all_images["Crop"] = crop_image(image)

    all_images["Distort"] = distort_image(image)

    return all_images


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Error! Usage is: ./Augmentation.py <file_path>')
        sys.exit(1)

    subdir = sys.argv[1]

    if os.path.isdir(data_path + subdir):
        file_count = 0

        for root, dirs, files in os.walk(os.path.join(data_path, subdir)):
            for file in files:
                filePath = os.path.join(root, file)
                if os.path.isfile(filePath):
                    file_count += 1
                    images = main(filePath)
                    # save file to augmented data folder
                    augmented_path = './data/augmented/' + subdir
                    try:
                        for image in images:
                            if not os.path.exists(augmented_path):
                                os.makedirs(augmented_path)
                            save_path = os.path.join(
                                augmented_path, f'{os.path.splitext(file)[0]}_{image}.jpg')
                            cv2.imwrite(save_path, images[image])
                    except Exception as e:
                        print(f'Error saving file {filePath}: {e}')
        print(f'Processed {file_count} files.')
    elif os.path.isfile(data_path + subdir):
        images = main(filePath)
    else:
        print('Error! Invalid file or directory')
        sys.exit(2)
