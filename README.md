# Object-Detection-using-YOLOv3
YOLO v3 (You Only Look Once Version 3) is a real-time object detection model that improves on YOLO v1 and v2. It uses a multi-scale detection approach, making it effective for objects of various sizes. With 53 convolutional layers (Darknet-53), YOLO v3 offers enhanced accuracy and speed, suitable for applications requiring real-time performance.

# Pre Trained Model
YOLOv3 Weights: The code downloads the pretrained YOLOv3 model weights from pjreddie.com. These weights are trained on the COCO dataset.

# Dataset
COCO Dataset: The YOLOv3 model in your code uses the COCO (Common Objects in Context) dataset, which is a large-scale object detection, segmentation, and captioning dataset. The COCO dataset contains 80 object categories with over 330,000 images.

# Features
1. Real-Time Object Detection: YOLOv3 is capable of detecting objects in real-time, making it ideal for applications that require immediate feedback.
2. Multi-Scale Detection: YOLOv3 can detect objects of different sizes by using multiple layers of the network for prediction, enhancing accuracy across various scales.
3. High-Speed Performance: Despite its deep architecture, YOLOv3 is optimized for speed, allowing for quick processing of images and videos.
4. Accuracy: YOLOv3 uses Darknet-53, a 53-layer convolutional network, to improve the precision of object detection compared to previous YOLO versions.
5. Pretrained Weights: The model utilizes pretrained weights, which allow for immediate use without the need for training from scratch.

# Requirements
System Requirements:
Operating System: Linux/Ubuntu, Windows, or macOS

Processor: Intel i5/i7 or AMD equivalent

GPU: NVIDIA GPU with CUDA support (optional but recommended for faster processing)

RAM: 8GB or more

Storage: At least 10GB free space (for dependencies, datasets, and outputs)

# Required Libraries and Tools
Git: To clone the Darknet repository.

Darknet Framework: The deep learning framework specifically for YOLO (cloned from GitHub).

GCC (GNU Compiler Collection): Required for compiling the Darknet framework.

OpenCV (optional): For processing and visualizing images (integrated within Darknet).

CUDA (optional): If using an NVIDIA GPU, CUDA is required for GPU acceleration.

wget: To download the YOLOv3 pretrained weights.

# Explanation
This project involves setting up and running YOLOv3 for object detection on a sample image using the Darknet framework. YOLOv3 is an advanced deep learning model that excels in detecting multiple objects within an image in real-time.

The project begins by cloning the Darknet repository, which contains the necessary files and scripts for running YOLO models. After compiling Darknet, the pretrained YOLOv3 weights are downloaded, which have been trained on the COCO dataset—a large-scale dataset containing 80 object categories.

Using these pretrained weights, the project performs object detection on a sample image (dog.jpg). The model processes the image, identifies objects (such as a dog, bicycle, etc.), and provides bounding boxes with labels around them.
