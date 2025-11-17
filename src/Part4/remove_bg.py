import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")

    return image


def remove_background(image_path):
    og_image = load_image(image_path)

    image_blurred = cv2.medianBlur(og_image, 5)

    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

    # Define the range for 'green'.
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Create a mask of "green" pixels
    hint_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Use morphological operations to close small holes and remove noise
    kernel = np.ones((7, 7), np.uint8)
    hint_mask = cv2.morphologyEx(hint_mask, cv2.MORPH_CLOSE, kernel,
                                 iterations=2)
    hint_mask = cv2.morphologyEx(hint_mask, cv2.MORPH_OPEN, kernel,
                                 iterations=1)

    # Find the largest contour in our "hint" mask
    contours, _ = cv2.findContours(hint_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No green contours found in image."
              "Try adjusting HSV range.")
        return None, None

    # Get the bounding box of the largest "green" object
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Add a small padding to the rectangle
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(og_image.shape[1] - x, w + padding * 2)
    h = min(og_image.shape[0] - y, h + padding * 2)
    rect = (x, y, w, h)

    # Execute GrabCut
    gc_mask = np.zeros(og_image.shape[:2], np.uint8)

    # Initialize background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut using the bounding box as the initial guess
    try:
        cv2.grabCut(og_image, gc_mask, rect, bgdModel, fgdModel,
                    5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"Error during GrabCut: {e}")
        return None, None

    # Create a final binary mask where 1 and 3 are foreground (1),
    # and 0 and 2 are background (0).
    final_mask = np.where((gc_mask == cv2.GC_FGD) |
                          (gc_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    # Apply Mask to Original Image
    white_background = np.full(og_image.shape, 255,
                               dtype=np.uint8)

    # Convert the 0/1 mask to a boolean mask for NumPy indexing
    boolean_mask = final_mask.astype(bool)

    # Use the boolean mask to copy pixels from the *original* image
    result_image = white_background
    result_image[boolean_mask] = og_image[boolean_mask]

    return result_image


def remove_bg(image_path):
    blank_background = remove_background(image_path)
    return blank_background
