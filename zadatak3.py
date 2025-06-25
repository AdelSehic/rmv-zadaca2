#!/usr/bin/env python3

import cv2
import utils as ut
import numpy as np
from InquirerPy import inquirer
import canneyutils as cut

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

img = cv2.imread(IMAGE)

while True:
    choice = inquirer.select(
        message="Select:",
        choices=["All", "Canney", "Sobel", "Laplacian", "Exit"]
    ).execute()

    if choice == "Sobel":
        ut.createGridNoTitles(5, 1, sobel_process(img))
    if choice == "Canney":
        ut.createGridNoTitles(5, 1, canny_process(img))
    if choice == "Exit":
        break

cv2.destroyAllWindows()
