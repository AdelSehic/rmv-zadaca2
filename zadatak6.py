#!/usr/bin/env python3
import utils as ut
import cv2
import numpy as np

titles = ['Original', 'Erosion', 'Dilation', 'Opening', 'Closing']

# Učitavanje slike u grayscale
img = cv2.imread('coins.jpg', cv2.IMREAD_GRAYSCALE)

# Binarizacija (za morfološke operacije je obično potrebna binarna slika)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Kreiranje strukturnog elementa (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Erozija (smanjuje objekte)
erosion = cv2.erode(binary, kernel, iterations=1)

# Dilacija (proširuje objekte)
dilation = cv2.dilate(binary, kernel, iterations=1)

# Otvaranje (erodira, pa dilatira) - uklanja šum
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Zatvaranje (dilatira, pa erodira) - zatvara male rupe
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

images = [binary, erosion, dilation, opening, closing]

ut.addImageTitles(images, titles)
ut.imageScroller(images)
