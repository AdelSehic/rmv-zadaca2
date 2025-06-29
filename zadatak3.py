#!/usr/bin/env python3

import cv2
import utils as ut
import numpy as np
from InquirerPy import inquirer

IMAGE = "lizard.jpg"

def sobel_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sob_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sob_x_display = cv2.convertScaleAbs(sob_x)
    sob_y_display = cv2.convertScaleAbs(sob_y)

    combined = cv2.addWeighted(sob_x_display, 0.5, sob_y_display, 0.5, 0)

    return ut.gray_to_bgr([image, gray, sob_x, sob_y, combined])

def canny_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 1.4)

    sob_x = cv2.convertScaleAbs(cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=3))
    sob_y = cv2.convertScaleAbs(cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=3))
    sobel_step = cv2.addWeighted(sob_x, 0.5, sob_y, 0.5, 0)

    canny = cv2.Canny(gray, 150, 175)
    return ut.gray_to_bgr([image, gray, gauss, sobel_step, canny])

def laplacian_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gauss, cv2.CV_64F)
    laplacian_display = cv2.convertScaleAbs(laplacian)

    # Apply Laplacian to original gray image (without blur)
    laplacian_raw = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_raw_display = cv2.convertScaleAbs(laplacian_raw)

    return ut.gray_to_bgr([image, gray, gauss, laplacian_raw_display, laplacian_display])

img = cv2.imread(IMAGE)

while True:
    choice = inquirer.select(
        message="Select:",
        choices=["All", "Canney", "Sobel", "Laplacian", "Exit"]
    ).execute()

    if choice == "All":
        sobel_images = sobel_process(img)
        canny_images = canny_process(img)
        laplacian_images = laplacian_process(img)
        
        all_images = sobel_images + canny_images + laplacian_images
        ut.createGridNoTitles(5, 3, all_images)
    if choice == "Sobel":
        ut.createGridNoTitles(5, 1, sobel_process(img))
    if choice == "Canney":
        ut.createGridNoTitles(5, 1, canny_process(img))
    if choice == "Laplacian":
        ut.createGridNoTitles(5, 1, laplacian_process(img))
    if choice == "Exit":
        break

cv2.destroyAllWindows()
