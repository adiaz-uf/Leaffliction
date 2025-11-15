import sys
import os
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib.pyplot as plt

from img_functions import flip_image, rotate_image, skew_image, shear_image, crop_image, distort_image, projective_transform

# Data path
data_path = './data/original/'
# Library const values
BORDER_DARK = cv2.BORDER_CONSTANT
BORDER_VALUE_DARK = (0, 0, 0)
BORDER_REFLECT = cv2.BORDER_REFLECT
BORDER_VALUE_REFLECT = None

border_mode = BORDER_DARK
border_value = BORDER_VALUE_DARK


def display_images(augmented_images):
    num_images = len(augmented_images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle('Image Augmentations', fontsize=16, fontweight='bold')

    axes = axes.flatten() if num_images > 1 else [axes]

    # Display augmented images
    for idx, (name, img) in enumerate(augmented_images.items(), start=0):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(name, fontsize=12)
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()


def make_augmentations(filePath, verbose=True):
    if verbose:
        print(f'Processing file: {filePath}')
    image = cv2.imread(filePath)

    if image is None:
        print(f'Error: Could not read image from {filePath}')
        return None

    all_images = {}

    # Image augmentations
    all_images["Original"] = image
    all_images["Flip"] = flip_image(image)
    all_images["Rotate"] = rotate_image(image)
    all_images["Skew"] = skew_image(image)
    all_images["Shear"] = shear_image(image)
    all_images["Crop"] = crop_image(image)
    all_images["Distort"] = distort_image(image)
    all_images["Projective"] = projective_transform(image)

    return all_images


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Error! Usage is: ./Augmentation.py <file_path>')
        sys.exit(1)

    subdir = sys.argv[1]

    if "-r" in sys.argv:
        border_mode = BORDER_REFLECT
        border_value = BORDER_VALUE_REFLECT

    if os.path.isdir(data_path + subdir):
        file_count = 0

        for root, dirs, files in os.walk(os.path.join(data_path, subdir)):
            for file in files:
                filePath = os.path.join(root, file)
                if os.path.isfile(filePath):
                    file_count += 1
                    result = make_augmentations(filePath)
                    if result is None:
                        continue

                    images = result
                    # save file to augmented data folder
                    augmented_path = './data/augmented/' + subdir
                    try:
                        for img_name in images:
                            if not os.path.exists(augmented_path):
                                os.makedirs(augmented_path)
                            save_path = os.path.join(
                                augmented_path,
                                f'{os.path.splitext(file)[0]}_{img_name}.jpg')
                            cv2.imwrite(save_path, images[img_name])
                    except Exception as e:
                        print(f'Error saving file {filePath}: {e}')
        print(f'Processed {file_count} files.')

    elif os.path.isfile(data_path + subdir):
        images = make_augmentations(data_path + subdir)
        if images is not None:
            display_images(images)
            augmented_path = './data/augmented/' + os.path.dirname(subdir)
            try:
                for image_name in images:
                    if not os.path.exists(augmented_path):
                        os.makedirs(augmented_path)
                    save_path = f'{augmented_path}/ \
                    {os.path.splitext(os.path.basename
                                      (subdir))[0]}_{image_name}.jpg'
                    cv2.imwrite(save_path, images[image_name])
                print("Success! File saved")
            except Exception as e:
                print(f'Error saving file: {e}')
    else:
        print('Error! Invalid file or directory')
        sys.exit(2)
