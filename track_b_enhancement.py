#!/usr/bin/env python3
"""
TRACK B - FINGER IMAGE ENHANCEMENT & TEMPLATE READINESS (UPDATED)
YellowSense Technologies - UIDAI SITAA Challenge

Updates:
- Added MediaPipe Segmentation Mask to remove background noise.
- Added Re-masking in enhancement step to prevent static.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from pathlib import Path
import urllib.request
import os

# ============================================================================
# STEP 1: FINGER REGION ISOLATION (AI-BASED WITH SEGMENTATION)
# ============================================================================

def isolate_finger_mediapipe(image):
    """
    Extract, ROTATE, and MASK finger region using MediaPipe Tasks API.
    Uses AI segmentation to force background to black.
    """
    model_path = 'hand_landmarker.task'
    
    # --- AUTO-DOWNLOAD MODEL IF MISSING ---
    if not os.path.exists(model_path):
        print(f"⬇️ Downloading {model_path} from Google...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✅ Download complete!")
        except Exception as e:
            return None, f"❌ Failed to download model: {str(e)}"

    try:
        # Create HandLandmarker options WITH SEGMENTATION
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=1,
            output_segmentation_masks=True  # <--- NEW: Enable Segmentation
        )
        detector = vision.HandLandmarker.create_from_options(options)
        
        # Convert image for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect
        results = detector.detect(mp_image)
        
        if not results.hand_landmarks:
            return None, "No hand detected (MediaPipe)"
        
        # --- 1. GET SEGMENTATION MASK ---
        # Convert mask from float (0.0-1.0) to uint8 (0-255)
        mask_float = results.segmentation_masks[0].numpy_view()
        mask_uint8 = (mask_float * 255).astype(np.uint8)
        
        # Resize mask to match image if needed (MediaPipe output can sometimes vary slightly)
        if mask_uint8.shape != image.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]))

        # --- 2. CALCULATE ROTATION ---
        hand_landmarks = results.hand_landmarks[0]
        h, w = image.shape[:2]
        
        # Index 5 (MCP) to Index 8 (TIP)
        idx_base = hand_landmarks[5] 
        idx_tip = hand_landmarks[8]
        
        px_base = (int(idx_base.x * w), int(idx_base.y * h))
        px_tip = (int(idx_tip.x * w), int(idx_tip.y * h))
        
        # Calculate angle
        angle_rad = math.atan2(px_tip[1] - px_base[1], px_tip[0] - px_base[0])
        angle_deg = math.degrees(angle_rad)
        rotation_angle = angle_deg + 90
        
        # --- 3. ROTATE IMAGE AND MASK ---
        center = ((px_base[0] + px_tip[0]) // 2, (px_base[1] + px_tip[1]) // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Adjust bounding box size
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
        # Use NEAREST neighbor for mask to keep edges sharp
        rotated_mask = cv2.warpAffine(mask_uint8, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
        
        # --- 4. APPLY MASK (REMOVE BACKGROUND) ---
        # Everything outside the mask becomes Black (0,0,0)
        # We use a threshold to ensure the mask is binary
        _, binary_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)
        rotated_image = cv2.bitwise_and(rotated_image, rotated_image, mask=binary_mask)

        # --- 5. CROP ---
        crop_w, crop_h = 300, 400
        new_center_x = int(new_w / 2)
        new_center_y = int(new_h / 2)
        
        x_start = max(0, new_center_x - crop_w // 2)
        y_start = max(0, new_center_y - crop_h // 2)
        
        finger_region = rotated_image[y_start:y_start+crop_h, x_start:x_start+crop_w]
        
        # Safety resize
        if finger_region.shape[:2] != (400, 300):
             finger_region = cv2.resize(finger_region, (300, 400))
             
        return finger_region, "✅ Finger isolated & MASKED (AI)"

    except Exception as e:
        return None, f"AI Failed ({str(e)})"

def isolate_finger_classical(image):
    """
    Fallback: Classical skin color detection
    """
    try:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask immediately to remove background
        masked_image = cv2.bitwise_and(image, image, mask=skin_mask)
        
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, "❌ No skin region detected"
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop from the masked image, not the original
        finger_region = masked_image[y:y+h, x:x+w]
        
        return finger_region, "✅ Finger isolated (Classical Masking)"
    
    except Exception as e:
        return None, f"❌ Classical segmentation failed: {str(e)}"


# ============================================================================
# STEP 2: NOISE REDUCTION & CONTRAST NORMALIZATION
# ============================================================================

def preprocess_image(image):
    """
    Noise reduction and contrast enhancement
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Bilateral filter (edge-preserving denoising)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(denoised)
    
    return normalized


# ============================================================================
# STEP 3: RIDGE ENHANCEMENT (ADAPTIVE WITH RE-MASKING)
# ============================================================================

def enhance_ridges_adaptive(image):
    """
    Enhance ridges using CLAHE and Adaptive Thresholding.
    Includes re-masking to prevent background noise.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- 1. CREATE VALIDITY MASK ---
    # Since we set the background to 0 in Step 1, we can create a mask 
    # of where the actual finger is (pixels > 0).
    # We will use this at the end to wipe out any noise the thresholder creates.
    _, valid_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # --- 2. ENHANCEMENT ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # --- 3. ADAPTIVE THRESHOLDING ---
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        11, 
        2
    )

    # --- 4. RE-APPLY MASK (CRITICAL FIX) ---
    # Adaptive thresholding often creates noise in large black areas.
    # We force those areas back to black using the validity mask.
    final = cv2.bitwise_and(binary, binary, mask=valid_mask)

    return final

# ============================================================================
# STEP 4: POST-PROCESSING (MORPHOLOGY & SKELETON)
# ============================================================================

def post_process_and_export(image):
    """
    Clean up the binary image and invert it for the standard look.
    """
    # Morphological Open to remove small white dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Invert (Standard: Black Ridges, White Valleys)
    final_output = cv2.bitwise_not(cleaned)
    
    # NOTE: Since we inverted, the background is now White (255).
    # If you prefer a transparent/white background for the final card, this is correct.

    # Resize to ISO Standard (300x400)
    iso_format = cv2.resize(final_output, (300, 400), interpolation=cv2.INTER_NEAREST)

    return iso_format

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def track_b_enhancement_pipeline(input_image):
    results = {
        'step1_original': input_image.copy(),
        'step2_isolated': None,
        'step3_preprocessed': None,
        'step4_enhanced': None,
        'step5_iso_format': None,
        'status_messages': []
    }
    
    # STEP 1: Finger region isolation
    finger_region, status1 = isolate_finger_mediapipe(input_image)
    results['status_messages'].append(status1)
    
    if finger_region is None:
        finger_region, status1_fallback = isolate_finger_classical(input_image)
        results['status_messages'].append(status1_fallback)
        if finger_region is None:
            return results, "❌ Failed to isolate finger region"
    
    results['step2_isolated'] = finger_region
    
    # STEP 2: Preprocessing
    preprocessed = preprocess_image(finger_region)
    results['step3_preprocessed'] = preprocessed
    results['status_messages'].append("✅ Preprocessing complete")
    
    # STEP 3: Ridge enhancement (Updated with re-masking)
    enhanced = enhance_ridges_adaptive(preprocessed)
    results['step4_enhanced'] = enhanced
    results['status_messages'].append("✅ Ridge enhancement complete")
    
    # STEP 4: Post-processing
    iso_format = post_process_and_export(enhanced)
    results['step5_iso_format'] = iso_format
    results['status_messages'].append("✅ ISO-like format export complete")
    
    final_status = "\n".join(results['status_messages'])
    return results, final_status

# ============================================================================
# DEMO
# ============================================================================

def demo_single_image(image_path):
    print("=" * 80)
    print("TRACK B - IMAGE ENHANCEMENT DEMO (SEGMENTATION ENABLED)")
    print("=" * 80)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"\n📁 Processing: {image_path}")
    results, status = track_b_enhancement_pipeline(image)
    print("\n" + status)
    
    if results['step5_iso_format'] is not None:
        output_dir = Path("track_b_results")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "step1_original.png"), results['step1_original'])
        cv2.imwrite(str(output_dir / "step2_isolated.png"), results['step2_isolated'])
        cv2.imwrite(str(output_dir / "step3_preprocessed.png"), results['step3_preprocessed'])
        cv2.imwrite(str(output_dir / "step4_enhanced.png"), results['step4_enhanced'])
        cv2.imwrite(str(output_dir / "step5_iso_format.png"), results['step5_iso_format'])
        print(f"\n✅ Results saved to: {output_dir}/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python track_b_enhancement.py <image_path>")
    else:
        demo_single_image(sys.argv[1])