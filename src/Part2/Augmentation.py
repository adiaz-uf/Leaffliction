import sys
import os
import cv2
import random
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib.pyplot as plt

# Data path
data_path = './data/original/'
# Library const values
border_mode = cv2.BORDER_CONSTANT
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


def projective_transform(image):
    height, width = image.shape[:2]

    shift_amount = random.uniform(0.15, 0.25)

    # Original corner points
    pts1 = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    transform_type = random.choice(
        ['tilt_left', 'tilt_right', 'tilt_top', 'tilt_bottom'])

    if transform_type == 'tilt_left':
        pts2 = np.float32([
            [int(width * shift_amount), int(height * shift_amount)],
            [width, 0],
            [width, height],
            [int(width * shift_amount), height - int(height * shift_amount)]
        ])
    elif transform_type == 'tilt_right':
        pts2 = np.float32([
            [0, 0],
            [width - int(width * shift_amount), int(height * shift_amount)],
            [width - int(width * shift_amount), height -
             int(height * shift_amount)],
            [0, height]
        ])
    elif transform_type == 'tilt_top':
        pts2 = np.float32([
            [int(width * shift_amount), int(height * shift_amount)],
            [width - int(width * shift_amount), int(height * shift_amount)],
            [width, height],
            [0, height]
        ])
    else:  # tilt_bottom
        pts2 = np.float32([
            [0, 0],
            [width, 0],
            [width - int(width * shift_amount), height -
             int(height * shift_amount)],
            [int(width * shift_amount), height - int(height * shift_amount)]
        ])

    # Get perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    projected = cv2.warpPerspective(
        image,
        M,
        (width, height),
        borderMode=border_mode,
        borderValue=border_value
    )

    return projected


def display_images(original_image, augmented_images):
    num_images = len(augmented_images) + 1
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle('Image Augmentations', fontsize=16, fontweight='bold')

    axes = axes.flatten() if num_images > 1 else [axes]

    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Display original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Display augmented images
    for idx, (name, img) in enumerate(augmented_images.items(), start=1):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(name, fontsize=12)
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()


def main(filePath):
    print(f'Processing file: {filePath}')
    image = cv2.imread(filePath)

    if image is None:
        print(f'Error: Could not read image from {filePath}')
        return None

    all_images = {}

    # Image augmentations
    all_images["Flip"] = flip_image(image)
    all_images["Rotate"] = rotate_image(image)
    all_images["Skew"] = skew_image(image)
    all_images["Shear"] = shear_image(image)
    all_images["Crop"] = crop_image(image)
    all_images["Distort"] = distort_image(image)
    all_images["Projective"] = projective_transform(image)

    return image, all_images


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
                    result = main(filePath)
                    if result is None:
                        continue

                    original, images = result
                    # save file to augmented data folder
                    augmented_path = './data/augmented/' + subdir
                    try:
                        for image_name in images:
                            if not os.path.exists(augmented_path):
                                os.makedirs(augmented_path)
                            save_path = os.path.join(
                                augmented_path, f'{os.path.splitext(file)[0]}_{image_name}.jpg')
                            cv2.imwrite(save_path, images[image_name])
                    except Exception as e:
                        print(f'Error saving file {filePath}: {e}')
        print(f'Processed {file_count} files.')

    elif os.path.isfile(data_path + subdir):
        images = main(data_path + subdir)
        if images is not None:
            original, images = images
            display_images(original, images)
            augmented_path = './data/augmented/' + os.path.dirname(subdir)
            try:
                for image_name in images:
                    if not os.path.exists(augmented_path):
                        os.makedirs(augmented_path)
                    save_path = f'{augmented_path}/{os.path.splitext(os.path.basename(subdir))[0]}_{image_name}.jpg'
                    cv2.imwrite(save_path, images[image_name])
                print("Success! File saved")
            except Exception as e:
                print(f'Error saving file: {e}')
    else:
        print('Error! Invalid file or directory')
        sys.exit(2)
