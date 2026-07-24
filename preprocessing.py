"""
preprocessing.py
-----------------
Converts a raw image (from the drawing canvas or the webcam) into the
28x28, centered, MNIST-style format the model expects.

This is the piece that makes or breaks real-world accuracy: MNIST
digits are always centered and normalized in a very specific way, so
we replicate that here rather than just doing a plain resize.
"""

import numpy as np
import cv2


def _center_by_mass(img28):
    """Shift the digit so its center of mass sits at the center of the
    28x28 canvas, exactly like the original MNIST preprocessing does."""
    total = img28.sum()
    if total == 0:
        return img28

    h, w = img28.shape
    ys, xs = np.indices((h, w))
    cy = (ys * img28).sum() / total
    cx = (xs * img28).sum() / total

    shift_y = int(round(h / 2.0 - cy))
    shift_x = int(round(w / 2.0 - cx))

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img28, M, (w, h), borderValue=0)


def preprocess_to_mnist(gray_uint8, already_white_on_black=True, threshold_val=None):
    """
    Parameters
    ----------
    gray_uint8 : np.ndarray (H, W), dtype uint8
        Grayscale image, single channel.
    already_white_on_black : bool
        True if the digit is already bright/white on a dark background
        (like the drawing canvas). False if it needs inverting
        (like a normal webcam photo of pen-on-paper).
    threshold_val : int or None
        Manual binarization threshold. If None, Otsu's method picks it
        automatically.

    Returns
    -------
    model_input : np.ndarray, shape (1, 28, 28, 1), float32, values in [0,1]
    display_img : np.ndarray, shape (28, 28), uint8 -- for showing what
                  the model actually "sees"
    """
    img = gray_uint8.copy()

    if not already_white_on_black:
        img = 255 - img

    # Binarize (Otsu picks a good threshold automatically for varying
    # lighting conditions, which matters a lot for webcam shots)
    if threshold_val is None:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)

    # Find the bounding box of the digit strokes so we can crop tightly
    coords = cv2.findNonZero(binary)
    if coords is None:
        # Nothing drawn
        blank = np.zeros((28, 28), dtype=np.uint8)
        return blank.astype('float32').reshape(1, 28, 28, 1), blank

    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y:y + h, x:x + w]

    # Resize the digit so its largest side is 20px (MNIST convention
    # leaves a small border), preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste onto a 28x28 canvas, centered
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit_resized

    # Fine center-of-mass alignment, like the real MNIST pipeline
    canvas = _center_by_mass(canvas)

    normalized = canvas.astype('float32') / 255.0
    model_input = normalized.reshape(1, 28, 28, 1)

    return model_input, canvas
