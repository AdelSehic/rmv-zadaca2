def waitEsc():
    import cv2
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return 0


def imageScroller(image_array):
    import cv2
    index = 0
    while True:
        cv2.imshow("Image scroller", image_array[index])
        key = cv2.waitKey(0) & 0xFF
        if key == 81:
            index = (index + 1) % len(image_array)
        if key == 83:
            index = (index - 1) % len(image_array)
        if key == 27:
            return 0
