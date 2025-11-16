import cv2
import random
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

# Library const values
BORDER_DARK = cv2.BORDER_CONSTANT
BORDER_VALUE_DARK = (0, 0, 0)
BORDER_REFLECT = cv2.BORDER_REFLECT
BORDER_VALUE_REFLECT = None

border_mode = BORDER_DARK
border_value = BORDER_VALUE_DARK


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
    angle = random.choice([-30, -15, 15, 30])
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

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
                               borderMode=border_mode,
                               borderValue=border_value)

    if skew_type == 'vertical' or skew_type == 'both':
        pts1 = np.float32([[0, 0], [width, 0], [0, height]])
        pts2 = np.float32(
            [[0, int(height * skew_amount)], [width, 0], [0, height]])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (width, height),
                               borderMode=border_mode,
                               borderValue=border_value)

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
