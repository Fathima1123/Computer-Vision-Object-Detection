#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import shutil
from pathlib import Path

class YOLOv4Trainer:
    def __init__(self):
        self.darknet_path = Path("darknet")
        self.classes = ["person", "car", "bicycle", "motorbike", "bus", 
                        "truck", "van", "moped", "trailers", "emergency vehicles"]
        self.num_classes = len(self.classes)

    def setup_environment(self):
        """Set up the training environment"""
        if not self.darknet_path.exists():
            print("Cloning darknet repository...")
            subprocess.run(["git", "clone", "https://github.com/AlexeyAB/darknet.git"])
        
        # Modify Makefile
        makefile_path = self.darknet_path / "Makefile"
        with open(makefile_path, 'r') as f:
            makefile_content = f.read()
        
        makefile_content = makefile_content.replace("GPU=0", "GPU=1")
        makefile_content = makefile_content.replace("CUDNN=0", "CUDNN=1")
        makefile_content = makefile_content.replace("OPENCV=0", "OPENCV=1")
        
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        
        # Compile darknet
        subprocess.run(["make", "-C", str(self.darknet_path)])

    def create_config_files(self, data_path):
        """Create necessary configuration files"""
        # Create custom.names
        with open(self.darknet_path / "data/custom.names", 'w') as f:
            f.write("\n".join(self.classes))
        
        # Create custom.data
        with open(self.darknet_path / "data/custom.data", 'w') as f:
            f.write(f"classes = {self.num_classes}\n")
            f.write(f"train = data/train.txt\n")
            f.write(f"valid = data/valid.txt\n")
            f.write(f"names = data/custom.names\n")
            f.write(f"backup = backup/\n")

        # Modify yolov4-custom.cfg
        cfg_path = self.darknet_path / "cfg/yolov4-custom.cfg"
        shutil.copy(self.darknet_path / "cfg/yolov4-custom.cfg", cfg_path)
        
        with open(cfg_path, 'r') as f:
            cfg_content = f.read()
        
        # Modify key parameters
        cfg_content = cfg_content.replace("learning_rate=0.001", "learning_rate=0.0001")
        cfg_content = cfg_content.replace("max_batches = 500500", f"max_batches = {max(6000, self.num_classes * 2000)}")
        cfg_content = cfg_content.replace("steps=400000,450000", f"steps={int(max_batches*0.8)},{int(max_batches*0.9)}")
        
        # Modify filters and classes in YOLO layers
        filters = (self.num_classes + 5) * 3
        cfg_content = self.modify_yolo_layers(cfg_content, filters)
        
        with open(cfg_path, 'w') as f:
            f.write(cfg_content)

    def modify_yolo_layers(self, cfg_content, filters):
        """Modify YOLO layers in config file"""
        lines = cfg_content.split('\n')
        yolo_count = 0
        for i, line in enumerate(lines):
            if '[yolo]' in line:
                yolo_count += 1
                # Find and modify the filters line before this yolo layer
                for j in range(i-1, max(i-20, 0), -1):
                    if 'filters=' in lines[j]:
                        lines[j] = f"filters={filters}"
                        break
                # Modify classes line in this yolo layer
                for j in range(i+1, min(i+20, len(lines))):
                    if 'classes=' in lines[j]:
                        lines[j] = f"classes={self.num_classes}"
                        break
        return '\n'.join(lines)

    def prepare_data(self, data_path):
        """Prepare train.txt and valid.txt"""
        image_files = list(Path(data_path).glob('*.jpg')) + list(Path(data_path).glob('*.png'))
        
        # Shuffle files
        import random
        random.shuffle(image_files)
        
        # Split into train and validation
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        valid_files = image_files[split_idx:]
        
        # Write train.txt
        with open(self.darknet_path / "data/train.txt", 'w') as f:
            f.write("\n".join(str(path) for path in train_files))
        
        # Write valid.txt
        with open(self.darknet_path / "data/valid.txt", 'w') as f:
            f.write("\n".join(str(path) for path in valid_files))

    def download_pretrained_weights(self):
        """Download pre-trained weights"""
        weights_path = self.darknet_path / "yolov4.conv.137"
        if not weights_path.exists():
            print("Downloading pre-trained weights...")
            subprocess.run(["wget", "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137",
                           "-P", str(self.darknet_path)])

    def train(self):
        """Start training"""
        subprocess.run([
            str(self.darknet_path / "darknet"),
            "detector",
            "train",
            str(self.darknet_path / "data/custom.data"),
            str(self.darknet_path / "cfg/yolov4-custom.cfg"),
            str(self.darknet_path / "yolov4.conv.137"),
            "-map"
        ])

def main():
    # Initialize trainer
    trainer = YOLOv4Trainer()
    
    # Setup paths
    data_path = input("Enter the path to your dataset directory: ")
    if not os.path.exists(data_path):
        print(f"Error: Path {data_path} does not exist")
        return
    
    # Setup and start training
    print("Setting up training environment...")
    trainer.setup_environment()
    
    print("Creating configuration files...")
    trainer.create_config_files(data_path)
    
    print("Preparing data...")
    trainer.prepare_data(data_path)
    
    print("Downloading pre-trained weights...")
    trainer.download_pretrained_weights()
    
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()