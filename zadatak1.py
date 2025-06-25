#!/usr/bin/env python3

import cv2
import utils as ut

IMAGE = "tulips.png"

ROTATION_STEP = 1

def rotate_image(image, angle):
    # dimenzije i centar slike
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # matrica rotacije slike
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # nove dimenzije rotirane slike
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # prilagodjavanje matrice rotacije novim dimenzijama
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated

img = cv2.imread(IMAGE)

rotation_steps_n = int(360 / ROTATION_STEP)
rotated_images = []

for i in range(rotation_steps_n):
    angle = i*ROTATION_STEP
    rotated = rotate_image(img, angle)
    rotated_images.append(rotated)

ut.imageScroller(rotated_images, -1)
cv2.destroyAllWindows()
