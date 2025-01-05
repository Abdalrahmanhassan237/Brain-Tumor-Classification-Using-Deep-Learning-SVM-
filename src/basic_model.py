# src/basic_model.py

import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import felzenszwalb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import create_data_splits, get_class_distribution


def apply_clahe(image):
    """Apply CLAHE for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def segment_image(image):
    """Basic image segmentation."""
    segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    return segments


def extract_features(image, segments):
    """Extract basic features from image."""
    # Morphological features
    labeled_segments = skimage_label(segments)
    properties = regionprops(labeled_segments, intensity_image=image)

    morph_features = []
    for prop in properties:
        features = [prop.area, prop.perimeter, prop.eccentricity, prop.mean_intensity]
        morph_features.extend(features)

    # LBP features
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))

    # Combine features
    return np.concatenate([morph_features, hist])


def load_and_process_images(data_path, classes):
    """Load and process all images."""
    X = []
    y = []

    for cls, label in classes.items():
        path = os.path.join(data_path, cls)
        if not os.path.exists(path):
            continue

        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if not os.path.isfile(img_path):
                continue

            # Load and preprocess image
            img = cv2.imread(img_path, 0)
            if img is None:
                continue

            img = cv2.resize(img, (200, 200))
            img_clahe = apply_clahe(img)
            segments = segment_image(img_clahe)

            # Extract features
            features = extract_features(img_clahe, segments)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def main():
    classes = {
        "no_tumor": 0,
        "pituitary_tumor": 1,
        "glioma_tumor": 2,
        "meningioma_tumor": 3,
    }

    # Load data from both Training and Testing folders
    X_train, X_test, y_train, y_test, train_files, test_files = create_data_splits(
        train_path="./Training", test_path="./Testing", classes=classes
    )

    # Print dataset distribution
    print("\nTraining set distribution:")
    print(get_class_distribution(y_train, classes))
    print("\nTesting set distribution:")
    print(get_class_distribution(y_test, classes))

    # Train model
    print("Training model...")
    model = SVC(kernel="rbf", C=10)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(classes.keys()),
        yticklabels=list(classes.keys()),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == "__main__":
    main()
