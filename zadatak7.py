#!/usr/bin/env python3

import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
import utils as ut

def make_probability_graph_img(probs, categories):
    scores = probs.numpy()
    labels = categories

    # Crtanje vertikalnog bar grafa (x=klase, y=vjerovatnoće)
    fig = plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color='skyblue')
    plt.ylabel("Confidence")
    plt.xlabel("Class")
    plt.xticks(rotation=30, ha='right')
    plt.title("Top-5 Predictions")
    plt.tight_layout()

    # Render to canvas
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Pretvori u NumPy sliku
    graph_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # Pretvori RGBA u BGR
    return cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)

def open_images(dir):
    images = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        img = Image.open(file_path).convert("RGB")
        images.append(img)
    return images

def classify_image(image, transform, vgg):
    input_tensor = transform(image).unsqueeze(0)
    output = vgg(input_tensor)

    with torch.no_grad():
        probs  = torch.nn.functional.softmax(output[0], dim=0)

    from torchvision.models import vgg16, VGG16_Weights

    weights = VGG16_Weights.DEFAULT
    categories = weights.meta["categories"]

    top5 = torch.topk(probs, 5)
    top5_labels = [categories[idx] for idx in top5.indices.tolist()]

    return top5.values, top5_labels


def main():
    images = open_images("neural_network_sets/vgg16")

    vgg16 = models.vgg16(pretrained = True)
    vgg16.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        )
    ])

    top5 = []
    top5_labels = []
    for img in images:
        top, label = classify_image(img, transform, vgg16)
        top5.append(top)
        top5_labels.append(label)

    graphs = []
    for values, labels, img in zip(top5, top5_labels, images):
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        graph_img = make_probability_graph_img(values, labels)
        ut.save_images([img_np, graph_img], ["image", "graph"], f"saved/zadatak7/{labels[0]}", 400)
        # graphs.append( np.hstack([ ut.resize_img(img_np, 400), make_probability_graph_img(values, labels) ]) )

    # ut.imageScroller(graphs)

if __name__ == "__main__":
    main()
