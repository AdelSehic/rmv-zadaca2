#!/usr/bin/env python3

import cv2
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

IMAGE = "tulips.png"

def normalize_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def greyscale_to_freq(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = 20 * np.log1p(magnitude)  # log scale for visibility
    mag_display = normalize_image(magnitude_spectrum)
    return mag_display, dft_shift

def get_frequency_masks(img, radius = 30):
    rows, cols = img.shape
    ccrow, ccol = rows // 2, cols // 2

    lowpass_mask = np.zeros((rows, cols, 2), np.uint8)
    highpass_mask = np.ones((rows, cols, 2), np.uint8)

    radius = int(max(ccol, ccrow) * ( radius / 100 ))

    cv2.circle(lowpass_mask, (ccol, ccrow), radius, (1, 1), thickness=-1)
    cv2.circle(highpass_mask, (ccol, ccrow), radius, (0, 0), thickness=-1)

    return lowpass_mask, highpass_mask

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

img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
img = ut.resize_img(img, 350)
img_display = normalize_image(img)

mag_display, dft_shift = greyscale_to_freq(img)
low, high = get_frequency_masks(img)

low_display = normalize_image(low)[:, :, 0]
high_display = normalize_image(high)[:, :, 0]

applied_low, filter_low = apply_freq_mask(dft_shift, low)
applied_high, filter_high = apply_freq_mask(dft_shift, high)

joined = np.vstack([np.hstack([img, mag_display, high_display, filter_high, applied_high]), np.hstack([img, mag_display, low_display, filter_low, applied_low])])
cv2.imshow("low", joined)
ut.waitEsc()
cv2.destroyAllWindows()
