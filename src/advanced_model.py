# src/advanced_model.py

import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import felzenszwalb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import albumentations as A
from utils.data_loader import create_data_splits, get_class_distribution


def create_augmentation_pipeline():
    """Create data augmentation pipeline."""
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                p=0.3,
            ),
        ]
    )


def extract_advanced_features(image, segments):
    """Extract advanced features from image."""
    # Morphological features
    labeled_segments = skimage_label(segments)
    properties = regionprops(labeled_segments, intensity_image=image)

    morph_features = []
    for prop in properties:
        features = [
            prop.area,
            prop.perimeter,
            prop.eccentricity,
            prop.mean_intensity,
            prop.solidity,
            prop.euler_number,
        ]
        morph_features.extend(features)

    # HOG features
    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )

    # Multi-radius LBP
    lbp_features = []
    for radius in [1, 2, 3]:
        lbp = local_binary_pattern(image, P=8 * radius, R=radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, 8 * radius + 3), range=(0, 8 * radius + 2)
        )
        lbp_features.extend(hist)

    # Combine all features
    return np.concatenate([morph_features, hog_features, lbp_features])


def process_single_image(img_path, augment=False):
    """Process a single image with optional augmentation."""
    img = cv2.imread(img_path, 0)
    if img is None:
        return None

    img = cv2.resize(img, (200, 200))

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Segment image
    segments = felzenszwalb(img_clahe, scale=150, sigma=0.7, min_size=100)

    # Extract features
    features = extract_advanced_features(img_clahe, segments)

    if augment:
        aug = create_augmentation_pipeline()
        augmented = aug(image=img_clahe)["image"]
        aug_segments = felzenszwalb(augmented, scale=150, sigma=0.7, min_size=100)
        aug_features = extract_advanced_features(augmented, aug_segments)
        return [features, aug_features]

    return [features]


def load_and_process_images(data_path, classes, augment=True):
    """Load and process all images with parallel processing."""
    X = []
    y = []

    for cls, label in classes.items():
        path = os.path.join(data_path, cls)
        if not os.path.exists(path):
            continue

        # Process images in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_single_image)(os.path.join(path, filename), augment)
            for filename in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename))
        )

        # Flatten results and add labels
        for result in results:
            if result is not None:
                for features in result:
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)


def create_stacking_classifier():
    """Create stacking classifier with multiple models."""
    estimators = [
        ("svm_rbf", SVC(kernel="rbf", probability=True)),
        ("svm_poly", SVC(kernel="poly", probability=True)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100, 50))),
    ]

    return StackingClassifier(
        estimators=estimators, final_estimator=SVC(kernel="rbf", probability=True), cv=5
    )


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

    # train model
    print("Training model...")
    model = create_stacking_classifier()
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
