#!/usr/bin/env python3

import cv2
import numpy as np
import os

YOLO_FS = os.path.join( os.path.curdir, "neural_network_sets", "yolo")

def yolo_path(file):
    return os.path.join(YOLO_FS, file)

# Load YOLO
net = cv2.dnn.readNetFromDarknet( yolo_path("yolov4.cfg"), yolo_path("yolov4.weights") )
classes = []
with open( yolo_path("coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Load video
cap = cv2.VideoCapture(yolo_path("yolo_clip.mp4"))  # or 0 for webcam

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("saved/zadatak8/output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    # Parse outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key to break
        break

cap.release()
out.release()
cv2.destroyAllWindows()
