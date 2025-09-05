# **Digit Extraction and 28×28 Image Generation**

## **1. Introduction**

Digit recognition is a common task in computer vision and machine learning, often used in handwritten digit classification systems such as MNIST. The goal of this project is to:

1. Extract individual digits from a scanned or photographed sheet of digits.
2. Resize each digit to **28×28 pixels** (compatible with MNIST format).
3. Save the extracted digits as **JPEG images** and optionally bundle them into a ZIP file.

This preprocessing is crucial for feeding images into machine learning models for classification.

---

## **2. Theory**

### **2.1 Grayscale Conversion**

Images are first converted to grayscale because color information is unnecessary for digit recognition. Grayscale reduces computational complexity while preserving structural information of digits.

```python
img = cv2.imread("digits_sheet.JPG", cv2.IMREAD_GRAYSCALE)
```

### **2.2 Thresholding**

To isolate the digits from the background, **binary thresholding** is used. This converts all pixel values above a threshold to white (background) and below the threshold to black (digits). In this project, we invert the image so digits become white on a black background:

```python
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
```

### **2.3 Contour Detection**

Contours represent the boundaries of objects (digits). Using OpenCV, we detect all external contours:

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### **2.4 Sorting Contours**

To maintain reading order, contours are sorted **row-wise and left-to-right**:

```python
contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1]//50, cv2.boundingRect(c)[0]))
```

* `cv2.boundingRect(c)` gives `(x, y, w, h)` for each contour.
* Sorting ensures digits are saved in natural reading order.

### **2.5 Resizing and Padding**

For ML models, digits are standardized to **28×28 pixels**:

1. Resize to **20×20** while preserving aspect ratio.
2. Add **4-pixel padding** on all sides to reach 28×28.

```python
digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
padded = np.pad(digit_resized, ((4,4),(4,4)), "constant", constant_values=0)
```

### **2.6 Saving and Archiving**

Digits are saved as JPEG files for compatibility and smaller file size. Finally, all images are zipped for easy distribution.

---

## **3. Full Python Code**

```python
import cv2
import numpy as np
from PIL import Image
import os
import zipfile

# ========= CONFIG =========
INPUT_IMAGE = "digits_sheet.JPG"   # Input image file
OUTPUT_DIR = "digits_28x28"       # Directory to save extracted digits
ZIP_NAME = "digits_28x28.zip"     # Name of output ZIP file
# ==========================

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image in grayscale
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

# Threshold (black digits on white background)
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours (digits)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours row by row, left to right
contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1]//50, cv2.boundingRect(c)[0]))

idx = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    
    # Skip tiny noise
    if w < 5 or h < 5:
        continue
    
    # Crop digit from thresholded image
    digit = thresh[y:y+h, x:x+w]

    # Resize digit to 20x20
    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Pad to 28x28
    padded = np.pad(digit_resized, ((4,4),(4,4)), "constant", constant_values=0)
    
    # Save each digit as JPEG
    out_path = os.path.join(OUTPUT_DIR, f"digit_{idx:03d}.jpg")
    Image.fromarray(padded).save(out_path, format='JPEG')
    idx += 1

# Create ZIP file
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for file in os.listdir(OUTPUT_DIR):
        zipf.write(os.path.join(OUTPUT_DIR, file), file)

print(f"✅ Done! Extracted {idx} digits into {OUTPUT_DIR} and zipped into {ZIP_NAME}")
```

---

## **4. Output**

1. Directory `digits_28x28/` containing JPEG images:

```
digit_000.jpg
digit_001.jpg
digit_002.jpg
...
```

2. ZIP file `digits_28x28.zip` containing all digits.

Each image is **28×28 pixels**, white digit on black background, suitable for ML training.

---

## **5. Applications**

* Preprocessing for handwritten digit recognition (e.g., MNIST classifier).
* Data augmentation or dataset creation for OCR systems.
* Any ML pipeline that requires uniform image input.
