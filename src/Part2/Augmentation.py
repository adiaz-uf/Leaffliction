import sys
import os
import cv2

# Data path
data_path = './data/original/'


def flip_image(image):
    return cv2.flip(image, 1)


def rotate_image(image):
    pass


def skew_image(image):
    pass


def shear_image(image):
    pass


def crop_image(image):
    pass


def distort_image(image):
    pass


def main(filePath):
    print(f'Processing file: {filePath}')
    image = cv2.imread(filePath)

    all_images = []

    # Image augmentations
    all_images.append(flip_image(image))

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
                            save_path = os.path.join(augmented_path, f'aug_{file_count}_{file}')
                            cv2.imwrite(save_path, image)
                    except Exception as e:
                        print(f'Error saving file {filePath}: {e}')
        print(f'Processed {file_count} files.')
    elif os.path.isfile(data_path + subdir):
        main(data_path + subdir)
    else:
        print('Error! Invalid file or directory')
        sys.exit(2)
