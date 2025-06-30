#!/usr/bin/env python3
import cv2
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

# IMAGE = "tulips.png"
IMAGE = "jet.jpg"

LOWPASS = 0
HIGHPASS = 1

def normalize_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def greyscale_to_freq(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = 20 * np.log1p(magnitude)  # log scale for visibility
    mag_display = normalize_image(magnitude_spectrum)
    return mag_display, dft_shift

def make_frequency_mask(img, high_low=LOWPASS, radius=30):
    rows, cols = img.shape
    ccrow, ccol = rows // 2, cols // 2

    radius = int(max(ccol, ccrow) * (radius / 100))

    if high_low == LOWPASS:
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, ccrow), radius, (1, 1), thickness=-1)
        return mask
    else:
        mask = np.ones((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, ccrow), radius, (0, 0), thickness=-1)
        return mask

def make_butterworth_mask(img, high_low=LOWPASS, cutoff=30, order=2):
    rows, cols = img.shape
    ccrow, ccol = rows // 2, cols // 2
    
    # Create distance matrix from center
    u = np.arange(rows).reshape(-1, 1) - ccrow
    v = np.arange(cols) - ccol
    D = np.sqrt(u**2 + v**2)
    
    # Cutoff frequency
    D0 = max(ccol, ccrow) * (cutoff / 100)
    
    # Create mask based on type
    if high_low == LOWPASS:
        # Butterworth lowpass filter
        H = 1 / (1 + (D / D0)**(2 * order))
    else:
        # Butterworth highpass filter
        H = 1 / (1 + (D0 / (D + 1e-8))**(2 * order))
    
    # Convert to complex mask
    mask = np.zeros((rows, cols, 2), dtype=np.float32)
    mask[:, :, 0] = H
    mask[:, :, 1] = H
    
    return mask

def apply_freq_mask(freq_img, mask):
    masked_img = freq_img * mask
    back_shift = np.fft.ifftshift(masked_img)
    spatial = cv2.idft(back_shift)
    spatial_magnitude = cv2.magnitude(spatial[:, :, 0], spatial[:, :, 1])
    spatial_img = normalize_image(spatial_magnitude)
    
    mag = cv2.magnitude(masked_img[:, :, 0], masked_img[:, :, 1])
    mag_log = 20 * np.log1p(mag)
    mag_norm = normalize_image(mag_log)
    
    return spatial_img, mag_norm

def display_filter_response(mask):
    """Display the filter response"""
    response = mask[:, :, 0]  # Take one channel
    response_display = normalize_image(response * 255)
    return response_display

# Load and prepare image
img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
# img = ut.resize_img(img, 400)
img_display = normalize_image(img)

# Get frequency domain representation
mag_display, dft_shift = greyscale_to_freq(img)

images = [img, mag_display]
labels = ["original", "mag_display"]

for x in [LOWPASS, HIGHPASS]:
    for i in [30, 50, 80]:
        mask = make_frequency_mask(img, x, i)
        applied_filter, magnitude_masked = apply_freq_mask(dft_shift, mask)

        images += [display_filter_response(mask), magnitude_masked, applied_filter]
        form = f"lowpass_{i}" if x == LOWPASS else f"highpass_{i}"
        labels += [ f"{form}_mask", f"{form}_magnitude", f"{form}_filtered_image" ]

ut.save_images(images, labels, "saved/zadatak5/ideal")

images = [img, mag_display]
labels = ["original", "mag_display"]

for x in [LOWPASS, HIGHPASS]:
    for i in [2, 3, 5, 8]:
        mask = make_butterworth_mask(img, x, 20, i)
        applied_filter, magnitude_masked = apply_freq_mask(dft_shift, mask)

        images += [display_filter_response(mask), magnitude_masked, applied_filter]
        form = f"lowpass_{i}" if x == LOWPASS else f"highpass_{i}"
        labels += [ f"{form}_mask", f"{form}_magnitude", f"{form}_filtered_image" ]

ut.save_images(images, labels, "saved/zadatak5/butterworth")

images = [img, mag_display]
labels = ["original", "mag_display"]

low_mask = make_frequency_mask(img, LOWPASS, 80)
high_mask = make_frequency_mask(img, HIGHPASS, 30)
bandpass_mask = cv2.bitwise_and(low_mask, high_mask)

applied_filter, magnitude_masked = apply_freq_mask(dft_shift, bandpass_mask)
images += [display_filter_response(bandpass_mask), magnitude_masked, applied_filter]
labels += [ "bandpass_mask", "bandpass_magnitude", "bandpass_filtered_image" ]

ut.save_images(images, labels, "saved/zadatak5/bandpass")
