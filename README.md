
# YOLOv4 Object Detection Project

## Overview
This project implements a custom YOLOv4 object detection model for identifying multiple classes of objects in videos and images. The model is trained to detect:
- Persons
- Cars
- Bicycles
- Motorbikes
- Buses
- Trucks
- Vans
- Mopeds
- Trailers
- Emergency vehicles

## Project Structure
```
yolov4-object-detection/
│
├── src/
│   ├── train.py
│   ├── testing.py
│   └── auto_labelling_v2.py
│
├── config/
│   ├── custom.names
│   └── yolov4-aug22.cfg
│
├── weights/
│   └── yolov4-aug22.weights (not included in repository)
│
└── README.md
```

## Prerequisites
- Python 3.6 or later
- CUDA-capable GPU (8GB+ VRAM recommended)
- CUDA and cuDNN installed
- OpenCV
- Darknet framework

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/yolov4-object-detection.git
    cd yolov4-object-detection
    ```

2. **Install required Python packages:**
    ```bash
    pip install opencv-python numpy
    ```

3. **Download pre-trained weights (not included in repository):**
    ```bash
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    ```

## Data Annotation Process
We used LabelImg for annotating our dataset. Here's our annotation workflow:

1. **Tool Setup**: Used LabelImg, a graphical image annotation tool
2. **Annotation Process**:
   - Labeled 1000+ images across all classes
   - Used bounding boxes to mark objects
   - Ensured consistent labeling across similar objects
3. **Data Format**: Annotations saved in YOLO format
4. **Quality Control**: Double-checked annotations for accuracy

### Auto-Labeling Script
We developed an auto-labeling script to speed up the annotation process:

```python
import cv2
import numpy as np
import os

# Paths to your model files
weights_path = 'yolov4-custom_final.weights'
cfg_path = 'yolov4-custom.cfg'
names_path = 'custom.names'  # or your custom .names file

# Load YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load class names
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Specify the desired classes
desired_classes = {
    'person': 0,
    'car': 1,
    'bicycle': 2,
    'motorbike': 3,
    'bus': 4,
    'truck': 5,
    'van': 6,
    'moped': 7,
    'trailers': 8,
    'emergency vehicles': 9
}

# Function to perform detection and display the image
def perform_detection(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Process each detection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter by confidence and class
            if confidence > 0.5 and classes[class_id] in desired_classes:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                x = int(centerX - w / 2)
                y = int(centerY - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(desired_classes[classes[class_id]])

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) > 0:
        label_path = image_path.rsplit('.', 1)[0] + '.txt'
        with open(label_path, 'w') as f:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                class_id = class_ids[i]

                # Convert to YOLO format (normalized center coordinates and width/height)
                x_center = (x + w_box / 2) / width
                y_center = (y + h_box / 2) / height
                w_box_norm = w_box / width
                h_box_norm = h_box / height

                # Save in YOLO format
                f.write(f"{class_id} {x_center} {y_center} {w_box_norm} {h_box_norm}\n")

# Directory containing the images
image_dir = 'images'

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        perform_detection(os.path.join(image_dir, filename))
```

## Training Process

### Dataset Preparation
1. Collected diverse dataset of traffic videos
2. Extracted frames and annotated using LabelImg
3. Split data: 80% training, 20% validation

### Training Configuration
- **Learning rate**: 0.0001 (lowered for fine-tuning)
- **Batch size**: 64
- **Subdivisions**: 16
- **Max batches**: 20000
- **Steps**: 16000, 18000

### Training Script
Key components of our training script:

```python
from pathlib import Path
import subprocess

class YOLOv4Trainer:
    def __init__(self):
        self.darknet_path = Path("darknet")
        self.classes = ["person", "car", "bicycle", "motorbike", "bus", 
                        "truck", "van", "moped", "trailers", "emergency vehicles"]
        self.num_classes = len(self.classes)

    def train(self):
        subprocess.run([
            str(self.darknet_path / "darknet"),
            "detector",
            "train",
            str(self.darknet_path / "data/custom.data"),
            str(self.darknet_path / "cfg/yolov4-custom.cfg"),
            str(self.darknet_path / "yolov4.conv.137"),
            "-map"
        ])
```

## Testing Process
We have developed a comprehensive testing script to evaluate the performance of our YOLOv4 model. This script automates the setup, runs the detection on input videos, processes the output, and displays the results.

### Detailed Testing Script
Below is the detailed testing script with explanations:

```python
# -*- coding: utf-8 -*-
"""testing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12sSFc-sb-LbXjK-pQE9Rpv35Yqd4qFFF
"""

# Commented out IPython magic to ensure Python compatibility.

# Clone Darknet repository (if not already done)
!git clone https://github.com/AlexeyAB/darknet.git
# %cd darknet

# Modify Makefile to enable GPU and OpenCV
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

# Compile Darknet
!make

import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

custom_data ="/content/darknet/custom.data"

print("Please upload your custom.data file:")
uploaded = files.upload()
custom_data = next(iter(uploaded))

print("Please upload your yolov4-aug22.cfg file:")
uploaded = files.upload()
cfg_file = next(iter(uploaded))

print("Please upload your yolov4-aug22.weights file:")
uploaded = files.upload()
weights_file = next(iter(uploaded))

print("Please upload your input video (output_video1.avi):")
uploaded = files.upload()
input_video = next(iter(uploaded))

# Run YOLOv4 on the video with custom data
!./darknet detector demo {custom_data} {cfg_file} {weights_file} {input_video} -thresh 0.25 -dont_show -ext_output -out_filename result_video.avi

# Run YOLOv4 and redirect output to a file
!./darknet detector demo {custom_data} {cfg_file} {weights_file} {input_video} -thresh 0.25 -dont_show -ext_output -out_filename result_video.avi > yolo_output.txt

import re

# Read the YOLOv4 output from the file
with open('yolo_output.txt', 'r') as file:
    yolo_output = file.read()

# Regex pattern to match the detection lines
pattern = r"(\w+): (\d+)%"

# Initialize variables
frame_detections = []
current_frame_detections = []
current_frame_confidences = []
total_confidence = 0
total_detections = 0

# Process each line of the output
for line in yolo_output.splitlines():
    matches = re.findall(pattern, line)

    if matches:
        for match in matches:
            class_name = match[0]
            confidence = int(match[1])
            current_frame_detections.append(class_name)
            current_frame_confidences.append(confidence)
            total_confidence += confidence
            total_detections += 1
    elif "FPS" in line:
        if current_frame_detections:
            frame_detections.append({
                'num_detections': len(current_frame_detections),
                'avg_confidence': sum(current_frame_confidences) / len(current_frame_confidences)
            })
            current_frame_detections = []
            current_frame_confidences = []

# Handle the last frame if the output does not end with "FPS"
if current_frame_detections:
    frame_detections.append({
        'num_detections': len(current_frame_detections),
        'avg_confidence': sum(current_frame_confidences) / len(current_frame_confidences)
    })

# Print per-frame results
for i, frame_data in enumerate(frame_detections):
    print(f"Frame {i+1}: {frame_data['num_detections']} detections, Avg confidence: {frame_data['avg_confidence']:.2f}%")

# Calculate overall confidence
if total_detections > 0:
    overall_avg_confidence = total_confidence / total_detections
    print(f"\nOverall: {total_detections} detections, Overall average confidence: {overall_avg_confidence:.2f}%")
else:
    print("\nNo detections were made.")

# Function to play the output video
import cv2
from google.colab.patches import cv2_imshow

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2_imshow(frame)

        # Clear the output to avoid cluttering
        from IPython.display import clear_output
        clear_output(wait=True)

        # Wait for a short time and check if the user wants to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Play the output video
print("Playing the output video:")
play_video('result_video.avi')
```

### Explanation of the Testing Script

1. **Environment Setup:**
   - **Cloning Darknet Repository:** The script starts by cloning the Darknet repository, which is essential for running YOLOv4.
   - **Configuring Makefile:** It modifies the `Makefile` to enable GPU support and OpenCV, which are crucial for performance and image processing capabilities.
   - **Compiling Darknet:** After configuration, it compiles Darknet to prepare it for detection tasks.

2. **Uploading Required Files:**
   - **Custom Data Files:** The script prompts the user to upload necessary configuration and weight files (`custom.data`, `yolov4-aug22.cfg`, `yolov4-aug22.weights`).
   - **Input Video:** It also requires an input video (`output_video1.avi`) on which object detection will be performed.

3. **Running YOLOv4 Detection:**
   - **Detection Command:** The script runs the YOLOv4 detector on the uploaded video with specified thresholds and outputs the result to `result_video.avi`.
   - **Output Redirection:** It also redirects the detection output to a text file `yolo_output.txt` for further processing.

4. **Processing Detection Output:**
   - **Parsing Detections:** Using regular expressions, the script parses the detection results to extract class names and confidence scores.
   - **Frame-wise Analysis:** It aggregates detections per frame, calculating the number of detections and average confidence for each frame.
   - **Overall Metrics:** The script computes the overall number of detections and the average confidence across all frames.

5. **Displaying Results:**
   - **Per-Frame Results:** Prints the number of detections and average confidence for each frame.
   - **Overall Results:** Provides cumulative detection metrics.
   - **Playing Output Video:** Finally, it plays the output video with detections visualized.

## Results
- **Average precision**: 85%
- **Average recall**: 80%
- **FPS on GPU**: 30-35

## Challenges and Solutions
1. **Annotation Time**: Developed auto-labeling script
2. **GPU Memory**: Used batch size of 64 with subdivisions
3. **Class Imbalance**: Augmented minority classes

## Future Improvements
- [ ] Implement TensorRT for faster inference
- [ ] Add more emergency vehicle data
- [ ] Experiment with YOLOv5 and YOLOv7

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Alexey Bochkovskiy](https://github.com/AlexeyAB) for Darknet and YOLOv4
- The LabelImg team for the annotation tool

