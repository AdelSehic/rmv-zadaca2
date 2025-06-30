#!/usr/bin/env python3

import cv2
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

IMAGE = "lena.png"

titles = [
    "Original",
    "Gaussian Noise",
    "Salt and Pepper Noise",
    "Gaussian Blur (Original)",
    "Gaussian Blur (Gaussian Noise)",
    "Gaussian Blur (Salt and Pepper)",
    "Averaging Filter (Original)",
    "Averaging Filter (Gaussian Noise)",
    "Averaging Filter (Salt and Pepper)",
    "Median Filter (Original)",
    "Median Filter (Gaussian Noise)",
    "Median Filter (Salt and Pepper)"
]

def add_gaussian_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noised_img = image.astype(np.float32) + gauss
    noised_img = np.clip(noised_img, 0, 255).astype(np.uint8)
    return noised_img

# amount of image to noise, ratio 0-100 of salt and pepper
def add_salt_and_pepper_noise(image, amount=0.01, ratio=50):
    noised_img = image.copy()
    h, w = image.shape[:2]
    noise_count = int(amount * h * w)

    for i in range(noise_count):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        noised_img[y, x] = 255 if np.random.randint(0, 100) < ratio else 0

    return noised_img

def add_gauss_filter(images):
    filtered = []
    for img in images:
        filtered.append(cv2.GaussianBlur(img, (3, 3), 0))
    return filtered

def add_median_filter(images):
    filtered = []
    for img in images:
        filtered.append(cv2.medianBlur(img, 3))
    return filtered

def add_averaging_filter(images):
    filtered = []
    for img in images:
        filtered.append(cv2.blur(img, (3, 3)))
    return filtered

img = cv2.imread(IMAGE)
images = [img]

gaussian_noise = add_gaussian_noise(img)
snp_noise = add_salt_and_pepper_noise(img)

images.extend([gaussian_noise, snp_noise])
images.extend(add_gauss_filter([img, gaussian_noise, snp_noise]))
images.extend(add_averaging_filter([img, gaussian_noise, snp_noise]))
images.extend(add_median_filter([img, gaussian_noise, snp_noise]))

images = ut.transposeArray(images, 3, 4)
titles = ut.transposeArray(titles, 3, 4)

ut.save_images(images, titles, "saved/zadatak2/", 400)

ut.addImageTitles(images, titles)
ut.imageScroller(images)

ut.createGrid(3, 4, images, titles)
