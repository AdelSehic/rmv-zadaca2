def waitEsc():
    import cv2
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return 0


def imageScroller(image_array, increment=1):
    import cv2
    index = 0
    while True:
        cv2.imshow("Image scroller", image_array[index])
        key = cv2.waitKey(0) & 0xFF
        if key == 81:
            index = (index - increment) % len(image_array)
        if key == 83:
            index = (index + increment) % len(image_array)
        if key == 27:
            return 0

def createGridNoTitles(grid_x, grid_y, images):
    import matplotlib.pyplot as plt
    import cv2

    fig, axs = plt.subplots(grid_y, grid_x)

    for idx, img in enumerate(images):
        row = idx // grid_x
        col = idx % grid_x
        ax = axs[row, col] if grid_y > 1 else axs[col]  # handle 1-row case
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def createGrid(grid_x, grid_y, images, titles):
    import matplotlib.pyplot as plt
    import cv2

    fig, axs = plt.subplots(grid_y, grid_x)

    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // grid_x
        col = idx % grid_x
        ax = axs[row, col] if grid_y > 1 else axs[col]  # handle 1-row case
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def addImageTitles(images, titles):
    import cv2
    for idx, (img, title) in enumerate(zip(images, titles)):
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

def transposeArray(array, x, y):
    transposed = []
    for col in range(x):
        for row in range(y):
            transposed.append(array[row * x + col])
    return transposed
