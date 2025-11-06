import cv2
import numpy as np

class Transform:
    def __init__(self, img_path):
        self.img_path = img_path

        # Original image
        self.original_img = None

        # Gaussian blur image        
        self.gaussian_blur_img = None

        # Mask image        
        self.masked_img = None

        # Roi objects image        
        self.roi_objects_img = None
        self.roi_objects_contours = 0

        # Analyze_object image        
        self.nalyze_object_img = None

        # Pseudolandmarks image        
        self.pseudolandmarks_img = None

    def load_image(self):
        image = cv2.imread(self.img_path)
        if image is None:
            print(f"Error: Could not read image from {self.img_path}")

        self.original_img = image
        return image


    """
    Creates a blurred binary mask (Leaf=white, Background=black) 
    using the Saturation (S) channel.
    """
    def gaussian_blur(self):
        # Convert image to HSV
        hsv_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)

        # Split HSV into separate channels
        h, s, v = cv2.split(hsv_img)

        # Apply Threshold on the S channel (Saturation)
        _, threshold_img = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)

        # Apply Gaussian blur
        blur_img = cv2.GaussianBlur(threshold_img, (5, 5), 0)

        self.gaussian_blur = blur_img
        return blur_img


    """ 
    The goal is to isolate the parts of the leaf that are diseased
    (the brown spots) and discard both the bottom (gray) and the healthy
    parts of the leaf (green).
    """
    def mask(self):
        if self.gaussian_blur is None:
            self.gaussian_blur()

        # 1. Get the 'b' channel from L*a*b* color space
        lab_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_img)

        # 2. Threshold the 'b' channel: pixels > 200 become white
        _, b_thresh = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)

        # 3. Combine  gaussian_blur mask with the b_thresh mask
        bs_mask = cv2.bitwise_or(self.gaussian_blur, b_thresh)

        # 4. Apply this final mask to the original image     
        # Create a white background
        background = np.full(self.original_img.shape, 255, dtype=self.original_img.dtype)

        # Get the inverse of the mask
        mask_inv = cv2.bitwise_not(bs_mask)

        # Get foreground (leaf) and background parts
        foreground = cv2.bitwise_and(self.original_img, self.original_img, mask=bs_mask)
        background_part = cv2.bitwise_and(background, background, mask=mask_inv)

        # Combine them to get the final image
        final_image = cv2.add(foreground, background_part)

        self.masked_img = final_image
        return final_image


    """
    Finds the contours (Regions of Interest) from the final binary mask.
    Paints the white parts of the binary image in green on the original image,
    then draws a blue rectangle around all the green parts.
    """
    def roi_objects(self):
        if self.masked_img is None:
            self.mask()

        # Convert masked image to grayscale for contour detection
        gray = cv2.cvtColor(self.masked_img, cv2.COLOR_BGR2GRAY)

        # Create binary image: diseased parts (not white) become white
        # White background (255) becomes black (0)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove small noise

        # Start with a copy of the original image
        roi_image = self.original_img.copy()

        # Paint the white parts (diseased regions) of binary image in GREEN
        # Where binary is white (255), paint green on roi_image
        roi_image[binary == 255] = [0, 255, 0]  # BGR: Green

        # Find the bounding box of all white pixels in binary image
        white_pixels = np.where(binary == 255)

        if len(white_pixels[0]) > 0:
            # Get min/max coordinates of white pixels
            y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
            x_min, x_max = white_pixels[1].min(), white_pixels[1].max()

            # Draw blue rectangle around all green parts
            cv2.rectangle(roi_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            self.roi_objects_img = roi_image
        else:
            print("No diseased regions found")
            self.roi_objects_img = roi_image

        return roi_image

    def analyze_object(self):
            # TODO: Implement analyze object logic
            pass

    def pseudolandmarks(self):
            # TODO: Implement pseudolandmarks logic
            pass

    def color_histogram(self):
            # TODO: Implement color histogram logic
            pass

def transform_image(image_path):
    transform = Transform(image_path)
    original_image = transform.load_image()

    if original_image is not None:
        cv2.imshow("Original Image", original_image)

    blurred_image = transform.gaussian_blur()
    if blurred_image is not None:
        cv2.imshow("Gaussian Blur", blurred_image)
        count = 1

    mask_image = transform.mask()
    if mask_image is not None:
        cv2.imshow("Mask Image", mask_image)

    roi_objects_img = transform.roi_objects()
    if roi_objects_img is not None:
        cv2.imshow("Roi Objects Image", roi_objects_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


