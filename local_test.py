import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_PATH = "segmentation_unet.h5"
INPUT_IMAGE = "test_finger2.jpg" 

# ==============================================================================
# 1. DEFINE BRAIN STRUCTURE (To fix version mismatch)
# ==============================================================================
def build_unet(input_shape):
    """
    We rebuild the exact same model structure locally.
    This prevents the 'Unrecognized keyword' error.
    """
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return models.Model(inputs=[inputs], outputs=[outputs])

# ==============================================================================
# 2. LOAD & RUN
# ==============================================================================
print("🏗️ Building Brain Structure...")
model = build_unet((256, 256, 3))

print(f"⏳ Loading Knowledge (Weights) from {MODEL_PATH}...")
try:
    # We load ONLY the weights, ignoring the version-specific config
    model.load_weights(MODEL_PATH)
    print("✅ Brain Ready!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit()

def process_fingerprint(image_path):
    # --- A. Load Image ---
    original = cv2.imread(image_path)
    if original is None:
        print(f"❌ Error: Could not read {image_path}")
        return
    
    h, w = original.shape[:2]
    print(f"📸 Image Loaded: {w}x{h} pixels")

    # --- B. AI Segmentation (The Scalpel) ---
    print("🔪 Running AI Scalpel...")
    
    img_resized = cv2.resize(original, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.astype('float32') / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    mask_pred = model.predict(input_tensor, verbose=0)[0] 
    mask_binary = (mask_pred > 0.5).astype('uint8')

    mask_full = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_3ch = np.stack([mask_full]*3, axis=-1)
    segmented = cv2.multiply(original, mask_3ch)

    # --- C. V3 Enhancement (The Static Killer) ---
    print("✨ Enhancing Ridges...")
    
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur (Static Remover)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Threshold (Ink Stamp)
    # Tweak '15' to '21' if ridges are too thick
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        15, 2 
    )
    
    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    final_output = cv2.bitwise_and(cleaned, cleaned, mask=mask_full)

    cv2.imwrite("result_1_segmented.jpg", segmented)
    cv2.imwrite("result_2_final.jpg", final_output)
    print("🎉 Done! Check 'result_2_final.jpg'")

process_fingerprint(INPUT_IMAGE)