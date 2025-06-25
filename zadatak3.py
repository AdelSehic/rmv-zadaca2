#!/usr/bin/env python3

import cv2
import utils as ut

IMAGE = "jet.jpg"

img = cv2.imread(IMAGE)

ut.imageScroller([img])
