#!/usr/bin/env python3
"""
Leaffliction - Disease Prediction
Predicts leaf disease from an image using the trained model.
"""

import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt

# Import remove_bg function from Part3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Part3.image_transformations import remove_bg

matplotlib.use('Qt5Agg')

def load_class_indices(json_path='class_indices.json'):
    """
    Load the class indices mapping from JSON file.
    
    Args:
        json_path: Path to class_indices.json file
        
    Returns:
        Dictionary mapping class names to indices
    """
    if not os.path.exists(json_path):
        print(f"Error: Class indices file '{json_path}' not found.", file=sys.stderr)
        print("Make sure you have trained the model first.", file=sys.stderr)
        sys.exit(1)
    
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    
    # Invert the mapping (index -> class_name)
    index_to_class = {v: k for k, v in class_indices.items()}
    
    return index_to_class


def preprocess_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        img_size: Target size for the image (default: 224x224)
        
    Returns:
        Preprocessed image array ready for prediction
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image '{image_path}'.", file=sys.stderr)
        print("Make sure it's a valid image file.", file=sys.stderr)
        sys.exit(1)
    
    # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img = cv2.resize(img, img_size)
    
    # Convert to float and normalize to [0, 1] (same as training)
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension (model expects batch of images)
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_disease(model, image_path, class_mapping, top_k=3):
    """
    Predict disease from an image.
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image to predict
        class_mapping: Dictionary mapping indices to class names
        top_k: Number of top predictions to return
        
    Returns:
        List of tuples (class_name, confidence)
    """
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    
    # Get probabilities for each class
    probabilities = predictions[0]
    
    # Get top k predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        class_name = class_mapping[idx]
        confidence = probabilities[idx] * 100
        results.append((class_name, confidence))
    
    return results


def format_class_name(class_name):
    """
    Format class name for better display.
    Example: 'Apple_Black_rot' -> 'Apple - Black Rot'
    """
    parts = class_name.split('_')
    if len(parts) >= 2:
        plant = parts[0]
        disease = ' '.join(parts[1:]).replace('_', ' ').title()
        return f"{plant} - {disease}"
    return class_name


def display_prediction_result(image_path, predicted_class, confidence):
    """
    Display prediction result with matplotlib.
    Shows original image on the left and background-removed image on the right.
    
    Args:
        image_path: Path to the original image
        predicted_class: Predicted class name
        confidence: Prediction confidence (0-100)
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not load image for display: {image_path}")
        return
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Get background-removed image
    try:
        bg_removed_img = remove_bg(image_path)
        if bg_removed_img is not None:
            bg_removed_img_rgb = cv2.cvtColor(bg_removed_img, cv2.COLOR_BGR2RGB)
        else:
            # If remove_bg fails, use original image
            bg_removed_img_rgb = original_img_rgb.copy()
    except Exception as e:
        print(f"Warning: Could not remove background: {e}")
        bg_removed_img_rgb = original_img_rgb.copy()
    
    fig = plt.figure(figsize=(10, 6), facecolor='black')
    
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.15, wspace=0.1,
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img_rgb)
    ax1.axis('off')
    
    # Background removed
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bg_removed_img_rgb)
    ax2.axis('off')
    
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')
    
    # Format the predicted class
    formatted_class = format_class_name(predicted_class)
    
    ax_text.text(0.5, 0.7, 'DL Classification', 
                 fontsize=30, fontweight='bold', color='white',
                 ha='center', va='center')
    
    label_text = 'Class predicted: '
    class_text = f'{formatted_class} ({confidence:.2f}%)'
    full_text = label_text + class_text
    
    total_width = len(full_text)
    
    offset = (len(class_text) - len(label_text)) / (total_width * 2)
    
    ax_text.text(0.52 - offset, 0.28, label_text, 
                 fontsize=16, color='white',
                 ha='right', va='center')
    
    ax_text.text(0.52 - offset, 0.28, class_text, 
                 fontsize=16, color='lime',
                 ha='left', va='center')
    
    ax_text.set_facecolor('black')
    
    plt.show()


def main():
    """Main prediction function."""

    if len(sys.argv) < 2:
        print("Error! Usage is: ./predict.py <image_path> [model_path] [class_indices_path]")
        print("\nExample:")
        print("  ./predict.py path/to/leaf_image.jpg")
        print("  ./predict.py path/to/leaf_image.jpg my_model.h5 my_classes.json")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'leaf_disease_model.h5'
    class_indices_path = sys.argv[3] if len(sys.argv) > 3 else 'class_indices.json'
    
    print("\n-- Leaffliction Disease Prediction --\n")
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    file_ext = os.path.splitext(image_path.lower())[1]
    if file_ext not in valid_extensions:
        print(f"Error: '{image_path}' is not a valid image file.", file=sys.stderr)
        print(f"Supported formats: {', '.join(valid_extensions)}", file=sys.stderr)
        sys.exit(1)
    
    # Load the trained model
    print(f"[1/3] Loading model from '{model_path}'...")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        print("Make sure you have trained the model first using train.py", file=sys.stderr)
        sys.exit(1)
    
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load class indices
    print(f"\n[2/3] Loading class mapping from '{class_indices_path}'...")
    class_mapping = load_class_indices(class_indices_path)
    print(f"✓ Found {len(class_mapping)} classes:")
    for idx, name in sorted(class_mapping.items()):
        print(f"    {idx}: {format_class_name(name)}")
    
    # Make prediction
    print(f"\n[3/3] Analyzing image '{image_path}'...")
    try:
        predictions = predict_disease(model, image_path, class_mapping, top_k=3)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    top_class, top_confidence = predictions[0]
    
    print(f"\n🔍 Primary Diagnosis: {format_class_name(top_class)}")
    print(f"   Confidence: {top_confidence:.2f}%")
    
    if len(predictions) > 1:
        print("\nAlternative Diagnoses:")
        for i, (class_name, confidence) in enumerate(predictions[1:], 2):
            print(f"   {i}. {format_class_name(class_name)}: {confidence:.2f}%")
    
    print("\n" + "="*60)
    
    # Interpretation
    if top_confidence >= 90:
        print("\nHigh confidence prediction - Very likely accurate")
    elif top_confidence >= 70:
        print("\nModerate confidence - Result is likely correct")
    else:
        print("\nLow confidence - Consider retaking the image or consulting an expert\n")
    
    # Display visualization
    print("\nDisplaying prediction visualization...")
    display_prediction_result(image_path, top_class, top_confidence)
    
    print("\n-- Prediction Complete --\n")


if __name__ == "__main__":
    main()
