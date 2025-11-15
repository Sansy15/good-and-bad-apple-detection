import numpy as np
import cv2
from PIL import Image
import os

def heuristic_rotten_mask(img_rgb):
    """
    Returns an RGB mask showing likely rotten/brown/dark regions as white on black background.
    img_rgb: HxWx3 uint8
    """
    img = img_rgb.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Brown-ish: hue roughly 5-30, saturation moderate-to-high, value not very high
    lower_brown = np.array([5, 50, 30])
    upper_brown = np.array([30, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark regions
    mask_dark = (v < 80).astype('uint8') * 255

    # Combine
    mask_combined = cv2.bitwise_or(mask_brown, mask_dark)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    # convert to 3-channel for display
    mask_rgb = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2RGB)
    return mask_rgb

def heuristic_rotten_score(img_rgb, brown_weight=0.7, dark_weight=0.3):
    """
    Compute a simple score in [0,1] indicating fraction of apple area that looks rotten.
    Uses brownish color and dark pixel ratios.
    """
    img = img_rgb.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # simple apple area mask: assume object is not background (use simple brightness+color heuristic)
    # This is naive — for production use a proper segmentation model.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, obj_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

    total_pixels = max(1, np.count_nonzero(obj_mask))

    # brownish pixels
    lower_brown = np.array([5, 50, 30])
    upper_brown = np.array([30, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_pixels = np.count_nonzero(cv2.bitwise_and(mask_brown, mask_brown, mask=obj_mask))

    # dark pixels
    mask_dark = (v < 80).astype('uint8') * 255
    dark_pixels = np.count_nonzero(cv2.bitwise_and(mask_dark, mask_dark, mask=obj_mask))

    brown_ratio = brown_pixels / total_pixels
    dark_ratio = dark_pixels / total_pixels

    score = brown_weight * brown_ratio + dark_weight * dark_ratio
    # clamp
    score = max(0.0, min(1.0, score))
    return score

def preprocess_image_for_model(img_rgb, target_size=(224,224)):
    """
    Resize, scale to [0,1], and return as float32 array suitable for Keras models.
    """
    img = Image.fromarray(img_rgb).resize(target_size)
    arr = np.asarray(img).astype('float32') / 255.0
    return arr

def load_model_if_exists(path):
    """
    Returns a loaded Keras model if the file exists, otherwise None.
    """
    try:
        if not os.path.exists(path):
            return None
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model
    except Exception:
        return None