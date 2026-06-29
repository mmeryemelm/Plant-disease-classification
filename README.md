# 🌱 Plant disease classification

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)


## 📊 Overview

Deep learning project using transfer learning to classify plant diseases from leaf images.

### Models Included
- ResNet50
- ResNet101
- ResNet152
- VGG16
- InceptionV4
- DenseNet121

### Features
- Transfer learning from ImageNet pre-trained models
- Data augmentation for training
- Categorical classification
- 80/20 train-validation split
- Model evaluation and comparison

## 🚀 Quick Start

```bash
pip install tensorflow opencv-python matplotlib numpy
python resnet50.py
```

## Models

Run any of these models to train and evaluate:
- `resnet50.py` - ResNet50 model
- `resnet101.py` - ResNet101 model
- `resnet152.py` - ResNet152 model
- `vgg16.py` - VGG16 model
- `inceptionv4.py` - InceptionV4 model
- `densenet121.py` - DenseNet121 model
