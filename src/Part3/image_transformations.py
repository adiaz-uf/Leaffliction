import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

matplotlib.use('Qt5Agg')

# !IMPORTANT: export QT_QPA_PLATFORM=wayland to show windows

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
        self.kept_mask = None
        self.roi_objects_img = None

        # Analyze_object image
        self.analyze_object_img = None

        # Pseudolandmarks image
        self.pseudolandmarks_img = None

        # Histogram
        self.color_histogram_img = None

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

        self.gaussian_blur_img = blur_img
        return blur_img

    """
    The goal is to isolate the parts of the leaf that are diseased
    (the brown spots) and discard both the bottom (gray) and the healthy
    parts of the leaf (green).
    """
    def mask(self):
        if self.gaussian_blur_img is None:
            self.gaussian_blur()

        # 1. Get the 'b' channel from L*a*b* color space
        lab_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_img)

        # 2. Threshold the 'b' channel: pixels > 200 become white
        _, b_thresh = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)

        # 3. Combine  gaussian_blur mask with the b_thresh mask
        bs_mask = cv2.bitwise_or(self.gaussian_blur_img, b_thresh)

        # 4. Apply this final mask to the original image
        # Create a white background
        background = np.full(self.original_img.shape, 255,
                             dtype=self.original_img.dtype)

        # Get the inverse of the mask
        mask_inv = cv2.bitwise_not(bs_mask)

        # Get foreground (leaf) and background parts
        foreground = cv2.bitwise_and(self.original_img, self.original_img,
                                     mask=bs_mask)
        background_part = cv2.bitwise_and(background, background,
                                          mask=mask_inv)

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

        # Fill small holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Start with a copy of the original image
        roi_image = self.original_img.copy()

        # Paint the white parts (diseased regions) of binary image in GREEN
        # Where binary is white (255), paint green on roi_image
        roi_image[binary == 255] = [0, 255, 0]  # BGR: Green

        self.kept_mask = roi_image

        # Find the bounding box of all white pixels in binary image
        white_pixels = np.where(binary == 255)

        if len(white_pixels[0]) > 0:
            # Get min/max coordinates of white pixels
            y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
            x_min, x_max = white_pixels[1].min(), white_pixels[1].max()

            # Draw blue rectangle around all green parts
            cv2.rectangle(roi_image, (x_min, y_min), (x_max, y_max),
                          (255, 0, 0), 2)

            self.roi_objects_img = roi_image
        else:
            print("No diseased regions found")
            self.roi_objects_img = roi_image

        return roi_image

    """
    Analyzes the leaf object using PlantCV to get shape properties
    (magenta lines) and overlays the disease contours (blue lines).
    """
    def analyze_object(self):
        if self.original_img is None:
            self.load_image()

        if self.gaussian_blur_img is None:
            self.gaussian_blur()

        # Set PlantCV parameters
        pcv.params.debug = None

        # Find contours of the leaf
        obj_contours, _ = cv2.findContours(self.gaussian_blur_img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)

        if len(obj_contours) == 0:
            print("Error: No leaf object found in gaussian_blur_img mask.")
            return self.original_img

        # Find the largest contour (the main leaf)
        largest_contour = max(obj_contours, key=cv2.contourArea)

        # Create a copy of the original image for drawing
        analysis_image = self.original_img.copy()

        # Use PlantCV to analyze size (calculates metrics)
        shape_img = pcv.analyze.size(img=analysis_image,
                                     labeled_mask=self.gaussian_blur_img,
                                     n_labels=1)

        # Draw the contour in magenta
        cv2.drawContours(shape_img, [largest_contour], -1, (255, 0, 255), 2)

        # Draw ellipse if possible
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            cv2.ellipse(shape_img, ellipse, (255, 0, 255), 2)

        analysis_image = shape_img

        if self.masked_img is None:
            self.mask()

        gray = cv2.cvtColor(self.masked_img, cv2.COLOR_BGR2GRAY)

        # Create a binary mask of just the diseased parts
        _, binary_disease = cv2.threshold(gray, 250, 255,
                                          cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        binary_disease = cv2.morphologyEx(binary_disease, cv2.MORPH_OPEN,
                                          kernel)
        binary_disease = cv2.morphologyEx(binary_disease, cv2.MORPH_CLOSE,
                                          kernel)

        # Find the contours of the diseased spots
        disease_contours, _ = cv2.findContours(binary_disease,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Draw the blue disease contours onto the analysis image
        cv2.drawContours(analysis_image, disease_contours, -1, (255, 0, 0), 2)

        # Store and return the final image
        self.analyze_object_img = analysis_image
        return self.analyze_object_img

    """
    Creates pseudolandmarks (landmark points) along the main contour
    and disease contours.
    - Blue points: Right side of leaf
    - Magenta points: Left side of leaf
    - Orange points: Disease contours
    """
    def pseudolandmarks(self):
        if self.original_img is None:
            self.load_image()
        if self.gaussian_blur_img is None:
            self.gaussian_blur()

        pseudolandmarks_img = self.original_img.copy()

        obj_contours, _ = cv2.findContours(self.gaussian_blur_img,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        
        if not obj_contours:
            print("Error: No main leaf contour found in gaussian_blur_img.")
            return self.original_img

        # Get the largest contour (the leaf)
        main_contour = max(obj_contours, key=cv2.contourArea)

        # Find top-most and bottom-most points
        top_index = main_contour[:, :, 1].argmin()
        bottom_index = main_contour[:, :, 1].argmax()

        # Split the contour into left and right segments
        if top_index < bottom_index:
            segment1 = main_contour[top_index:bottom_index]
            segment2 = np.concatenate((main_contour[bottom_index:],
                                       main_contour[:top_index]))
        else:
            segment1 = main_contour[bottom_index:top_index]
            segment2 = np.concatenate((main_contour[top_index:],
                                       main_contour[:bottom_index]))

        def sample_points_from_contour(contour, n_points):
            """Sample n_points evenly from a contour"""
            if len(contour) == 0:
                return []
            
            indices = np.linspace(0, len(contour) - 1, n_points, dtype=int)
            points = [tuple(contour[i][0]) for i in indices]
            return points

        # Sample 30 points from each segment
        points_segment1 = sample_points_from_contour(segment1, 30)
        points_segment2 = sample_points_from_contour(segment2, 30)

        if self.masked_img is None:
            self.mask()

        gray = cv2.cvtColor(self.masked_img, cv2.COLOR_BGR2GRAY)
        _, binary_disease = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((3, 3), np.uint8)
        binary_disease = cv2.morphologyEx(binary_disease, cv2.MORPH_OPEN, kernel)
        binary_disease = cv2.morphologyEx(binary_disease, cv2.MORPH_CLOSE, kernel)
        
        disease_contours, _ = cv2.findContours(binary_disease,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Draw segment 1 points in BLUE
        for point in points_segment1:
            cv2.circle(pseudolandmarks_img, point, 5, (255, 0, 0), -1)

        # Draw segment 2 points in MAGENTA
        for point in points_segment2:
            cv2.circle(pseudolandmarks_img, point, 5, (255, 0, 255), -1)

        # Draw disease contour points in ORANGE
        for spot_contour in disease_contours:
            if len(spot_contour) > 5:
                disease_points = sample_points_from_contour(spot_contour, 15)
                for point in disease_points:
                    cv2.circle(pseudolandmarks_img, point, 4, (0, 60, 255), -1)

        self.pseudolandmarks_img = pseudolandmarks_img
        return self.pseudolandmarks_img

    """
    Generates a 9-channel color histogram (Figure IV.7) for the
    masked leaf pixels using BGR, HSV, and L*a*b color spaces.
    """
    def color_histogram(self):
        if self.original_img is None:
            self.load_image()
        if self.gaussian_blur_img is None:
            self.gaussian_blur()
        
        # Convert to HSV
        hsv_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)
        # Convert to L*a*b*
        lab_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2Lab)
        
        # Split all channels
        b, g, r = cv2.split(self.original_img)
        h, s, v = cv2.split(hsv_img)
        l, a, b_lab = cv2.split(lab_img)
        
        # Data structure: (channel_data, label, color_code)
        channels_to_plot = [
            (b, 'blue', 'blue'),
            (b_lab, 'blue-yellow', 'yellow'),
            (g, 'green', 'green'),
            (a, 'green-magenta', 'magenta'),
            (h, 'hue', 'purple'),
            (l, 'lightness', 'gray'),
            (r, 'red', 'red'),
            (s, 'saturation', 'cyan'),
            (v, 'value', 'orange')
        ]

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 7))

        # --- Calculate and Plot Histograms ---
        
        # Get the total number of non-zero pixels in the mask (our 100%)
        total_pixels = np.count_nonzero(self.gaussian_blur_img)
        if total_pixels == 0:
            print("Error: Mask is empty, cannot calculate histogram.")
            plt.close(fig)
            return None

        for channel_data, label, color in channels_to_plot:
            # Calculate histogram ONLY for the masked area
            hist = cv2.calcHist(
                images=[channel_data], 
                channels=[0], 
                mask=self.gaussian_blur_img, # This is the key
                histSize=[256], 
                ranges=[0, 256]
            )
            
            # Normalize to get proportion (%)
            hist_percent = (hist / total_pixels) * 100
            
            # Plot the histogram line
            ax.plot(hist_percent, color=color, label=label, linewidth=2.5)

        # Format Plot
        ax.set_xlabel("Pixel intensity", fontsize=12)
        ax.set_ylabel("Proportion of pixels (%)", fontsize=12)
        ax.set_title("Figure IV.7: Color Histogram", fontsize=14, fontweight='bold')
        
        # Add legend outside the plot area
        legend = ax.legend(title="Color Channel", bbox_to_anchor=(1.02, 1), loc='upper left')
        legend.get_title().set_fontweight('bold')
        
        # Set axis limits
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        
        # Adjust layout to make space for the legend
        fig.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        # Store and return the figure
        self.color_histogram_fig = fig
        return self.color_histogram_fig


def transform_image(image_path):
    transform = Transform(image_path)
    
    # Process all images
    original_image = transform.load_image()
    blurred_image = transform.gaussian_blur()
    mask_image = transform.mask()
    roi_objects_img = transform.roi_objects()
    analyzed_img = transform.analyze_object()
    pseudolandmarks_img = transform.pseudolandmarks()
    color_hist_fig = transform.color_histogram()

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle('Leaf Image Transformation Pipeline', fontsize=14, fontweight='bold')

    # Convert BGR to RGB for matplotlib (OpenCV uses BGR, matplotlib uses RGB)
    def bgr_to_rgb(img):
        if img is not None and len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # Row 1
    axes[0, 0].imshow(bgr_to_rgb(original_image))
    axes[0, 0].set_title('1. Original Image', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(blurred_image, cmap='gray')
    axes[0, 1].set_title('2. Gaussian Blur Mask', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(bgr_to_rgb(mask_image))
    axes[0, 2].set_title('3. Masked Image', fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(bgr_to_rgb(roi_objects_img))
    axes[1, 0].set_title('4. ROI Objects', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(bgr_to_rgb(analyzed_img))
    axes[1, 1].set_title('5. Analyzed Object', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(bgr_to_rgb(pseudolandmarks_img))
    axes[1, 2].set_title('6. Pseudolandmarks', fontsize=10, fontweight='bold')
    axes[1, 2].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # Show both figures
    plt.show()


"""
Transforms all images in src_dir and saves each transformation
as .png files in dst_dir with descriptive names.

Args:
    src_dir: Source directory containing images
    dst_dir: Destination directory for transformed images
"""
def transform_directory(src_dir, dst_dir):
    import os
    from pathlib import Path
    
    # Validate source directory exists
    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return
    
    if not os.path.isdir(src_dir):
        print(f"Error: '{src_dir}' is not a directory.")
        return
    
    # Get list of image files
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in os.listdir(src_dir) 
                   if os.path.isfile(os.path.join(src_dir, f)) 
                   and os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"Error: No valid images found in '{src_dir}'.")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Destination directory: {dst_dir}")
    
    # Process each image
    total_images = len(image_files)
    print(f"\nProcessing {total_images} images...\n")
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(src_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        print(f"[{idx}/{total_images}] Processing: {image_file}")
        
        try:
            # Create transform instance
            transform = Transform(image_path)
            
            # Process all transformations
            original_image = transform.load_image()
            if original_image is None:
                print(f"Skipped: Could not read image")
                continue
            
            blurred_image = transform.gaussian_blur()
            mask_image = transform.mask()
            roi_objects_image = transform.roi_objects()
            analyzed_image = transform.analyze_object()
            pseudolandmarks_image = transform.pseudolandmarks()
            histogram_fig = transform.color_histogram()
            
            # Save all transformations
            transformations = [
                (original_image, f"{base_name}_1_original.png"),
                (blurred_image, f"{base_name}_2_gaussian_blur.png"),
                (mask_image, f"{base_name}_3_mask.png"),
                (roi_objects_image, f"{base_name}_4_roi_objects.png"),
                (analyzed_image, f"{base_name}_5_analyzed.png"),
                (pseudolandmarks_image, f"{base_name}_6_pseudolandmarks.png")
            ]
            
            for img, filename in transformations:
                if img is not None:
                    output_path = os.path.join(dst_dir, filename)
                    cv2.imwrite(output_path, img)
            
            # Save histogram as PNG
            if histogram_fig is not None:
                histogram_path = os.path.join(dst_dir, f"{base_name}_7_color_histogram.png")
                histogram_fig.savefig(histogram_path, dpi=150, bbox_inches='tight')
                plt.close(histogram_fig)
            
            print(f"Saved 7 transformations (6 images + 1 histogram)")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue
    
    print(f"\nProcessing complete! All images saved to: {dst_dir}")

