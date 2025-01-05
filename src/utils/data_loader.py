# src/utils/data_loader.py

import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(data_path, classes, is_training=True):
    """
    Load dataset from specified path

    Args:
        data_path: Path to data directory (Training or Testing)
        classes: Dictionary of class names and their labels
        is_training: Boolean to indicate if this is training data
    """
    X = []
    y = []
    file_paths = []  # To keep track of processed files

    for cls, label in classes.items():
        class_path = os.path.join(data_path, cls)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} does not exist")
            continue

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            if os.path.isfile(img_path):
                file_paths.append(img_path)
                if is_training:
                    X.append(process_image(img_path))
                    y.append(label)
                else:
                    X.append(process_image(img_path))
                    y.append(label)

    return np.array(X), np.array(y), file_paths


def create_data_splits(train_path, test_path, classes):
    """
    Create training and testing datasets

    Args:
        train_path: Path to training data
        test_path: Path to testing data
        classes: Dictionary of class names and their labels
    """
    # Load training data
    X_train, y_train, train_files = load_dataset(train_path, classes, is_training=True)

    # Load testing data if available
    if os.path.exists(test_path):
        X_test, y_test, test_files = load_dataset(test_path, classes, is_training=False)
    else:
        print(
            "No separate test directory found. Using train_test_split on training data."
        )
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        train_files, test_files = train_test_split(
            train_files, test_size=0.2, random_state=42
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, train_files, test_files


def get_class_distribution(y, classes):
    """
    Get distribution of classes in dataset

    Args:
        y: Array of labels
        classes: Dictionary of class names and their labels
    """
    distribution = {}
    reverse_classes = {v: k for k, v in classes.items()}

    for label in np.unique(y):
        class_name = reverse_classes[label]
        count = np.sum(y == label)
        distribution[class_name] = count

    return distribution
