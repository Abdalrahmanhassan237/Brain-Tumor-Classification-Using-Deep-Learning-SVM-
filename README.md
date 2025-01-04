# Brain Tumor Classification Using Deep Learning 🧠
![image](https://github.com/user-attachments/assets/2cc8772f-17ef-489b-97f0-2024dfdd0260)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔭 Overview

This project implements a machine learning system for brain tumor classification using MRI images. It includes both basic and advanced implementations, utilizing various image processing techniques and machine learning algorithms to classify brain tumors into four categories:
- No Tumor
- Pituitary Tumor
- Glioma Tumor
- Meningioma Tumor

## ✨ Features

### Basic Model
- Local Binary Pattern (LBP) feature extraction
- Basic morphological feature analysis
- Single SVM classifier
- Suitable for learning and initial implementation

### Advanced Model
- Enhanced feature extraction (HOG + multi-radius LBP)
- Data augmentation techniques
- Ensemble learning with multiple classifiers
- Parallel processing for improved performance
- Advanced preprocessing techniques

## 📁 Project Structure
```
brain_tumor_classification/
├── requirements.txt
├── src/
│   ├── basic_model.py      # Basic implementation
│   ├── advanced_model.py   # Advanced implementation
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── visualization.py
├── Training/               # Training dataset
│   ├── no_tumor/
│   ├── pituitary_tumor/
│   ├── glioma_tumor/
│   └── meningioma_tumor/
└── Testing/               # Testing dataset
    ├── no_tumor/
    ├── pituitary_tumor/
    ├── glioma_tumor/
    └── meningioma_tumor/
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Create and activate virtual environment:

For Windows:
```bash
python -m venv brain_tumor_env
brain_tumor_env\Scripts\activate
```

For Linux/Mac:
```bash
python3 -m venv brain_tumor_env
source brain_tumor_env/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running Basic Model
```bash
python src/basic_model.py
```

### Running Advanced Model
```bash
python src/advanced_model.py
```

### Example Code
```python
from src.advanced_model import process_single_image

# Process a single image
result = process_single_image('path_to_image.jpg')
print(f"Classification Result: {result}")
```

## 🏗️ Model Architecture

### Basic Model
- Feature Extraction: LBP, Basic Morphological Features
- Classifier: Single SVM with RBF Kernel
- Processing: Sequential Processing

### Advanced Model
- Feature Extraction:
  - Multi-radius LBP
  - HOG Features
  - Enhanced Morphological Features
- Classifier: Stacking Classifier
  - SVM with RBF Kernel
  - SVM with Polynomial Kernel
  - Neural Network
- Processing: Parallel Processing with JobLib

## 📊 Results

Performance metrics for both models:

| Model    | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|---------|-----------|
| Basic    | 85%      | 84%       | 85%     | 84%       |
| Advanced | 92%      | 91%       | 92%     | 91%       |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📧 Contact

Abdelrahman Hassan - abdelrahman.hassan237@gmail.com
<br> Youssef Ahmed - youssefahmed8915@gmail.com


