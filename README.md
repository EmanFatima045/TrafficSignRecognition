# 🚦 Traffic Sign Recognition System

A deep learning-powered system that accurately identifies and classifies traffic signs using Convolutional Neural Networks (CNN). This project demonstrates the application of computer vision and machine learning to solve real-world autonomous driving challenges.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This Traffic Sign Recognition System leverages deep learning to automatically detect and classify traffic signs from images. The system processes traffic sign images, extracts meaningful features, and accurately identifies the type of sign—making it invaluable for autonomous vehicle navigation, driver assistance systems, and road safety applications.

**Key Capabilities:**
- Automated traffic sign detection and classification
- High accuracy recognition across multiple sign categories
- Robust preprocessing pipeline for consistent results
- Trained CNN model ready for deployment

---

## ✨ Features

- **Intelligent Image Processing**: Automatic resizing and normalization of input images
- **Label Extraction**: Smart parsing of labels from image filenames
- **Advanced CNN Architecture**: Multi-layer convolutional neural network for feature detection
- **Data Normalization**: Pixel value scaling for optimal model performance
- **Train-Test Split**: Proper dataset division to ensure unbiased evaluation
- **Model Persistence**: Save and load trained models for future predictions
- **High Accuracy**: Achieves strong performance on traffic sign classification tasks

---

## 🏗️ System Architecture

The system follows a comprehensive pipeline from raw images to trained model:

```
Raw Images → Preprocessing → Feature Extraction → CNN Training → Model Evaluation → Saved Model
```

### Pipeline Stages:

1. **Data Loading**: Read traffic sign images from dataset directory
2. **Image Preprocessing**: Resize images to uniform dimensions
3. **Label Extraction**: Parse sign categories from filenames
4. **Normalization**: Convert images to numerical arrays and normalize pixel values
5. **Label Encoding**: Transform labels into categorical format
6. **Dataset Splitting**: Divide data into training and testing sets
7. **Model Training**: Train CNN on features like edges, shapes, and patterns
8. **Evaluation**: Test model accuracy on unseen data
9. **Model Saving**: Persist trained model for future use

---

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

```
tensorflow>=2.0.0
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pandas>=1.2.0
```

---

## 📁 Dataset Structure

The dataset should be organized with images containing labels in their filenames:

```
dataset/
│
├── stop_sign_001.jpg
├── stop_sign_002.jpg
├── yield_sign_001.jpg
├── speed_limit_30_001.jpg
├── speed_limit_30_002.jpg
└── ...
```

**Filename Convention**: `{label}_{identifier}.{extension}`

The system automatically extracts the label from the filename prefix before the first underscore or number.

---

## 🚀 Usage

### Training the Model

```python
# Run the main training script
python train_model.py
```

### Making Predictions

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('traffic_sign_model.h5')

# Load and preprocess an image
image = cv2.imread('test_sign.jpg')
image = cv2.resize(image, (32, 32))  # Adjust size as per your model
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)

# Make prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print(f"Predicted Sign: {predicted_class}")
```

### Evaluating Model Performance

```python
# The training script automatically evaluates on test data
# Check the output for accuracy metrics
```

---

## 🧠 Model Architecture

The CNN architecture is specifically designed to detect and classify traffic signs through hierarchical feature learning:

### Network Layers:

1. **Convolutional Layers**: Extract spatial features like edges, corners, and textures
2. **Activation Functions**: ReLU activation for non-linearity
3. **Pooling Layers**: Reduce spatial dimensions while retaining important features
4. **Dropout Layers**: Prevent overfitting during training
5. **Fully Connected Layers**: Combine features for final classification
6. **Output Layer**: Softmax activation for multi-class probability distribution

### Feature Detection Hierarchy:

- **Layer 1**: Detects basic edges and color gradients
- **Layer 2**: Identifies shapes and simple patterns
- **Layer 3**: Recognizes complex sign structures
- **Final Layers**: Classifies complete traffic sign types

---

## 📊 Results

The model demonstrates strong performance on traffic sign classification:

- **Training Accuracy**: Achieved through iterative learning on training dataset
- **Test Accuracy**: Validated on held-out test set for real-world performance
- **Generalization**: Model successfully recognizes signs it hasn't seen during training

### Performance Metrics:

- Precision, Recall, and F1-Score available per sign category
- Confusion matrix for detailed classification analysis
- Training/validation loss curves for convergence monitoring

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement:

- Add real-time video processing capabilities
- Implement data augmentation for improved accuracy
- Support for additional traffic sign datasets
- Mobile deployment optimization
- Multi-language sign recognition

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Traffic sign datasets from [specify your data source]
- TensorFlow and Keras documentation
- Computer vision research community
- Open-source contributors

---

## 📧 Contact

**Project Maintainer**: Your Name

- Email: your.email@example.com
- GitHub: https://github.com/EmanFatima045/
- LinkedIn: ttps://www.linkedin.com/in/eman-fatima-512a01246/

---

## 🔮 Future Enhancements

- [ ] Real-time traffic sign detection from video streams
- [ ] Mobile app integration (iOS/Android)
- [ ] Multi-country traffic sign support
- [ ] Transfer learning with pre-trained models
- [ ] Edge device deployment (Raspberry Pi, Jetson Nano)
- [ ] API endpoint for cloud-based predictions
- [ ] Enhanced data augmentation techniques

---

**Built with ❤️ for safer roads and autonomous driving**
