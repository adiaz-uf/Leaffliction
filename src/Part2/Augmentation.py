import sys
import os
import cv2
import matplotlib.pyplot as plt

import img_functions
from img_functions import flip_image, rotate_image, skew_image, shear_image
from img_functions import crop_image, distort_image, projective_transform

generated_images = 0


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
    plt.savefig("img/augmentations.png", bbox_inches='tight')
    plt.close()


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


def augment_train_data(dir_info, max_images):
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
        except Exception:
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
                    augmented_path,
                    f'{os.path.splitext(file)[0]}_{image_name}.jpg')
                cv2.imwrite(save_path, result[image_name])
                needed_images -= 1
                generated_images += 1
        except Exception as e:
            print(f'Error saving file {file_path}: {e}')


if __name__ == "__main__":
    use_for_train = False  # Indicate if augmentations are for training data
    if len(sys.argv) < 2:
        print('Error! Usage is: ./Augmentation.py <file_path>')
        print('Use -h for all options')
        sys.exit(1)

    if "-h" in sys.argv:
        print('Usage: ./Augmentation.py <file_path or directory> [options]\n')
        print("If a file path is provided,"
              " the augmented image will be displayed.")
        print(
            "If a directory path is provided, all images in the directory will"
            " be processed and saved to './data/augmented/<subdir>/'\n")
        print('Options:')
        print('  -r    Use reflect border mode for augmentations')
        print('  -t    Make augmentations suitable for training')
        print('  -h    Show this help message')
        sys.exit(0)

    folder_path = sys.argv[1]

    if "-r" in sys.argv:
        img_functions.border_mode = img_functions.BORDER_REFLECT
        img_functions.border_value = img_functions.BORDER_VALUE_REFLECT

    # If just a file, show augmented images
    if os.path.isfile(folder_path):
        images = make_augmentations(folder_path)
        if images is not None:
            display_images(images)
            exit(0)
        else:
            print('Error: images could not be processed.')
            exit(2)

    if "-t" in sys.argv:
        use_for_train = True

    print(folder_path)
    file_count = 0
    if os.path.isdir(folder_path):
        values = []
        max_images = 1000

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

        if use_for_train:
            """ augment data for training """
            print("-- Starting data augmentation... --\n")
            for item in values:
                augment_train_data(item, max_images)
            print(f'New generated images total: {generated_images}')
            exit(0)

        else:
            """ save augmented images to augmented data folder """
            for item in values:
                augmented_path = os.path.join(
                    './data/augmented/', item['dir'])
                for file in item['files']:
                    file_count += 1
                    file_path = os.path.join(item['full_path'], file)
                    print(f'Processing file: {file_path}')
                    result = make_augmentations(file_path, verbose=False)
                    if result is None:
                        continue

                    # Save augmented image
                    try:
                        for i, img_name in enumerate(result):
                            save_path = os.path.join(
                                augmented_path,
                                f'{os.path.splitext(file)[0]}_{img_name}.jpg')
                            cv2.imwrite(save_path, result[img_name])
                            generated_images += 1
                    except Exception:
                        print(f'Error saving file {file_path}')
            print(f'Processed {file_count} files.')
            print(f'Saved images  total: {generated_images}')
            exit(0)

    print('Error! Invalid file or directory')
    sys.exit(2)
