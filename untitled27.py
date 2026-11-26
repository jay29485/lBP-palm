# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:48:19 2026

@author: karun
"""


"""
Tamil Character Recognition: BiLSTM vs CNN vs MobileNetV2
=========================================================

Dataset structure:
DATASET_ROOT/
    class_01/
        image1.png
        image2.jpg
    class_02/
        ...
    ...

Install:
pip install tensorflow scikit-learn pandas numpy matplotlib pillow

Features:
1. Three deep-learning models: BiLSTM, CNN, MobileNetV2
2. Class-wise sample distribution
3. Stratified 70/15/15 split
4. Exact duplicate audit
5. Near-duplicate audit using dHash
6. Training-only augmentation
7. Accuracy, precision, recall, F1-score
8. Per-class evaluation
9. Confusion matrices
10. Learning curves
11. Training and inference time
12. Model complexity and file size
13. Classification reports
14. Saved models and experiment configuration

IMPORTANT:
- Verify EXPECTED_CLASSES before running.
- All three models use identical train/validation/test indices.
- Test data is not used for training or checkpoint selection.
- Results are actual measured results, not guaranteed accuracy.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import time
import hashlib
import random
import gc
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"C:/Users/karun/Downloads/brii")

OUTPUT_DIR = Path("C:/Users/karun/Downloads/Tsr")

# Your original code specified 25 classes.
# Change to 30 only if your dataset actually contains 30 folders.
EXPECTED_CLASSES = 25

IMAGE_SIZE = 64

RANDOM_SEED = 42

BATCH_SIZE = 32

MAX_EPOCHS = 50

LEARNING_RATE = 5e-4

EARLY_STOPPING_PATIENCE = 50

LR_PATIENCE = 3

MIN_LEARNING_RATE = 1e-6

DROPOUT_RATE = 0.35

L2_COEFFICIENT = 1e-4

# Near-duplicate threshold:
# Smaller distance means more visually similar.
DHASH_THRESHOLD = 5

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

# True for quick pipeline testing.
# Do not use smoke-test results in the research paper.
SMOKE_TEST = False

if SMOKE_TEST:
    MAX_EPOCHS = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seeds(seed=RANDOM_SEED):

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


set_seeds()


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset(root):

    if not root.exists() or not root.is_dir():

        raise FileNotFoundError(
            f"Dataset folder not found: {root}"
        )

    class_names = sorted(
        p.name for p in root.iterdir()
        if p.is_dir()
    )

    if EXPECTED_CLASSES is not None:

        if len(class_names) != EXPECTED_CLASSES:

            raise ValueError(
                f"Expected {EXPECTED_CLASSES} classes, "
                f"but found {len(class_names)}.\n"
                f"Available folders: {class_names}\n"
                "Correct EXPECTED_CLASSES or check your dataset."
            )

    if len(class_names) < 2:
        raise ValueError("At least two classes are required.")

    images = []
    labels = []
    file_paths = []

    distribution_rows = []
    unreadable_rows = []

    for class_index, class_name in enumerate(class_names):

        class_dir = root / class_name

        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
        )

        loaded_count = 0

        for file_path in files:

            try:

                with Image.open(file_path) as im:

                    im = im.convert("L")

                    im = im.resize(
                        (IMAGE_SIZE, IMAGE_SIZE),
                        Image.Resampling.LANCZOS
                    )

                    arr = np.asarray(
                        im,
                        dtype=np.uint8
                    )

                images.append(
                    arr[..., np.newaxis]
                )

                labels.append(class_index)

                file_paths.append(str(file_path))

                loaded_count += 1

            except Exception as exc:

                unreadable_rows.append({
                    "class": class_name,
                    "file": str(file_path),
                    "error": str(exc)
                })

        distribution_rows.append({

            "class_index": class_index,

            "class_name": class_name,

            "image_count": loaded_count

        })

    if not images:
        raise ValueError("No readable images found.")

    X = np.asarray(images, dtype=np.uint8)

    y = np.asarray(labels, dtype=np.int32)

    counts = np.bincount(
        y,
        minlength=len(class_names)
    )

    if np.any(counts < 2):

        low_classes = [
            class_names[i]
            for i, count in enumerate(counts)
            if count < 2
        ]

        raise ValueError(
            "Each class needs at least 3 readable images "
            "for the requested stratified 70/15/15 split. "
            f"Problematic classes: {low_classes}"
        )

    distribution = pd.DataFrame(distribution_rows)

    distribution.to_csv(
        OUTPUT_DIR / "class_wise_distribution.csv",
        index=False
    )

    if unreadable_rows:

        pd.DataFrame(unreadable_rows).to_csv(
            OUTPUT_DIR / "unreadable_images.csv",
            index=False
        )

    return (
        X,
        y,
        class_names,
        file_paths,
        distribution
    )


print("\nLoading dataset...")

X, y, class_names, file_paths, class_distribution = (
    load_dataset(DATASET_ROOT)
)

print("\nCLASS-WISE DISTRIBUTION")

print(class_distribution.to_string(index=False))

print("\nNumber of classes:", len(class_names))

print("Total readable images:", len(X))

print("Image shape:", X.shape)


# ============================================================
# 4. EXACT DUPLICATE AUDIT
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


def exact_duplicate_audit(file_paths, labels):

    print("\nChecking exact duplicate images...")

    hash_groups = defaultdict(list)

    for idx, path in enumerate(file_paths):

        try:

            image_hash = sha256_file(path)

            hash_groups[image_hash].append(idx)

        except OSError as exc:

            print("Hash error:", path, exc)

    rows = []

    group_number = 0

    for indices in hash_groups.values():

        if len(indices) > 1:

            group_number += 1

            for idx in indices:

                rows.append({

                    "duplicate_group": group_number,

                    "dataset_index": idx,

                    "class_name": class_names[
                        int(labels[idx])
                    ],

                    "file_path": file_paths[idx]

                })

    df = pd.DataFrame(rows)

    if not df.empty:

        df.to_csv(
            OUTPUT_DIR / "exact_duplicate_audit.csv",
            index=False
        )

    print("Exact duplicate groups:", group_number)

    return df


exact_duplicates = exact_duplicate_audit(
    file_paths,
    y
)


# ============================================================
# 5. NEAR-DUPLICATE IMAGE AUDIT
# ============================================================

def calculate_dhash(path, hash_size=8):

    with Image.open(path) as im:

        im = im.convert("L")

        im = im.resize(
            (hash_size + 1, hash_size),
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(im)

    differences = (
        pixels[:, 1:] > pixels[:, :-1]
    )

    hash_value = 0

    for bit in differences.flatten():

        hash_value = (
            (hash_value << 1) | int(bit)
        )

    return hash_value


def hamming_distance(hash1, hash2):

    return bin(hash1 ^ hash2).count("1")


def audit_near_duplicates(
    file_paths,
    labels,
    train_idx,
    val_idx,
    test_idx
):

    print("\nChecking near-duplicate images...")

    split_map = {}

    for split_name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx)
    ]:

        for idx in indices:

            split_map[int(idx)] = split_name

    class_hashes = defaultdict(list)

    for idx, path in enumerate(file_paths):

        try:

            dhash = calculate_dhash(path)

            class_hashes[int(labels[idx])].append(
                (idx, dhash)
            )

        except Exception as exc:

            print("Near-duplicate hash error:", path, exc)

    rows = []

    # Compare images within the same class.
    # This is a candidate similarity audit, not proof
    # that two samples are duplicates.

    for class_index, items in class_hashes.items():

        for item1, item2 in combinations(items, 2):

            idx1, hash1 = item1

            idx2, hash2 = item2

            distance = hamming_distance(
                hash1,
                hash2
            )

            if distance <= DHASH_THRESHOLD:

                split1 = split_map[idx1]

                split2 = split_map[idx2]

                rows.append({

                    "index_1": idx1,

                    "index_2": idx2,

                    "class_name": class_names[class_index],

                    "split_1": split1,

                    "split_2": split2,

                    "hamming_distance": distance,

                    "cross_split": split1 != split2,

                    "file_1": file_paths[idx1],

                    "file_2": file_paths[idx2]

                })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_DIR / "near_duplicate_audit.csv",
        index=False
    )

    cross_split_count = (
        int(df["cross_split"].sum())
        if not df.empty else 0
    )

    print("Near-duplicate pairs:", len(df))

    print("Cross-split pairs:", cross_split_count)

    print(
        "Manually inspect flagged pairs before deciding "
        "whether any samples should be removed."
    )

    return df


# ============================================================
# 6. STRATIFIED 70/15/15 DATA SPLIT
# ============================================================

all_indices = np.arange(len(y))

train_val_idx, test_idx = train_test_split(

    all_indices,

    test_size=0.15,

    random_state=RANDOM_SEED,

    stratify=y

)

val_fraction = 0.15 / 0.85

train_idx, val_idx = train_test_split(

    train_val_idx,

    test_size=val_fraction,

    random_state=RANDOM_SEED,

    stratify=y[train_val_idx]

)

X_train, y_train = X[train_idx], y[train_idx]

X_val, y_val = X[val_idx], y[val_idx]

X_test, y_test = X[test_idx], y[test_idx]


print("\nDATASET PARTITION")

print(
    f"Training: {len(X_train)} "
    f"({len(X_train)/len(X)*100:.2f}%)"
)

print(
    f"Validation: {len(X_val)} "
    f"({len(X_val)/len(X)*100:.2f}%)"
)

print(
    f"Testing: {len(X_test)} "
    f"({len(X_test)/len(X)*100:.2f}%)"
)


# ============================================================
# 7. SAVE SPLIT MANIFEST
# ============================================================

split_rows = []

for split_name, indices in [

    ("train", train_idx),

    ("validation", val_idx),

    ("test", test_idx)

]:

    for idx in indices:

        split_rows.append({

            "dataset_index": int(idx),

            "split": split_name,

            "class_index": int(y[idx]),

            "class_name": class_names[int(y[idx])],

            "file_path": file_paths[idx]

        })


pd.DataFrame(split_rows).to_csv(

    OUTPUT_DIR / "data_split_manifest.csv",

    index=False

)


# Save class distribution within each partition.

split_distribution = []

for split_name, labels in [

    ("train", y_train),

    ("validation", y_val),

    ("test", y_test)

]:

    counts = np.bincount(
        labels,
        minlength=len(class_names)
    )

    for class_index, class_name in enumerate(class_names):

        split_distribution.append({

            "split": split_name,

            "class_index": class_index,

            "class_name": class_name,

            "image_count": int(counts[class_index])

        })


pd.DataFrame(split_distribution).to_csv(

    OUTPUT_DIR / "split_class_distribution.csv",

    index=False

)


# Now perform the near-duplicate audit.

near_duplicate_results = audit_near_duplicates(

    file_paths,

    y,

    train_idx,

    val_idx,

    test_idx

)


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

class_ids = np.unique(y_train)

weights = compute_class_weight(

    class_weight="balanced",

    classes=class_ids,

    y=y_train

)

class_weight = {

    int(k): float(v)

    for k, v in zip(class_ids, weights)

}


# ============================================================
# 9. SHARED DATA AUGMENTATION
# ============================================================

def make_augmentation():

    return tf.keras.Sequential([

        layers.RandomRotation(

            factor=0.02,

            fill_mode="nearest",

            seed=RANDOM_SEED

        ),

        layers.RandomTranslation(

            height_factor=0.04,

            width_factor=0.04,

            fill_mode="nearest",

            seed=RANDOM_SEED + 1

        ),

        layers.RandomZoom(

            height_factor=0.05,

            width_factor=0.05,

            fill_mode="nearest",

            seed=RANDOM_SEED + 2

        )

    ], name="training_augmentation")


# ============================================================
# 10. MODEL A: ROW-SEQUENCE BiLSTM
# ============================================================

def build_bilstm(num_classes):

    inputs = layers.Input(

        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),

        name="grayscale_character_image"

    )

    x = layers.Rescaling(1.0 / 255.0)(inputs)

    x = make_augmentation()(x)

    # Each row represents one time step.

    x = layers.Reshape(

        (IMAGE_SIZE, IMAGE_SIZE)

    )(x)

    x = layers.Bidirectional(

        layers.LSTM(

            64,

            return_sequences=False,

            dropout=0.20,

            recurrent_dropout=0.0

        ),

        name="bidirectional_lstm"

    )(x)

    x = layers.Dense(

        128,

        activation="relu",

        kernel_regularizer=regularizers.l2(
            L2_COEFFICIENT
        )

    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(

        num_classes,

        activation="softmax",

        name="class_probabilities"

    )(x)

    model = models.Model(

        inputs,

        outputs,

        name="Tamil_Row_BiLSTM"

    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ============================================================
# 11. MODEL B: CUSTOM 2D CNN
# ============================================================

def conv_block(x, filters, block_name):

    x = layers.Conv2D(

        filters,

        kernel_size=3,

        padding="same",

        use_bias=False,

        kernel_regularizer=regularizers.l2(
            L2_COEFFICIENT
        ),

        name=f"{block_name}_conv1"

    )(x)

    x = layers.BatchNormalization(

        name=f"{block_name}_bn1"

    )(x)

    x = layers.Activation(

        "relu",

        name=f"{block_name}_relu1"

    )(x)

    x = layers.Conv2D(

        filters,

        kernel_size=3,

        padding="same",

        use_bias=False,

        kernel_regularizer=regularizers.l2(
            L2_COEFFICIENT
        ),

        name=f"{block_name}_conv2"

    )(x)

    x = layers.BatchNormalization(

        name=f"{block_name}_bn2"

    )(x)

    x = layers.Activation(

        "relu",

        name=f"{block_name}_relu2"

    )(x)

    x = layers.MaxPooling2D(

        pool_size=2,

        name=f"{block_name}_pool"

    )(x)

    return x


def build_cnn(num_classes):

    inputs = layers.Input(

        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),

        name="grayscale_character_image"

    )

    x = layers.Rescaling(1.0 / 255.0)(inputs)

    x = make_augmentation()(x)

    x = conv_block(x, 32, "block1")

    x = conv_block(x, 64, "block2")

    x = conv_block(x, 128, "block3")

    x = layers.Conv2D(

        256,

        3,

        padding="same",

        use_bias=False,

        kernel_regularizer=regularizers.l2(
            L2_COEFFICIENT
        ),

        name="block4_conv"

    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(

        128,

        activation="relu",

        kernel_regularizer=regularizers.l2(
            L2_COEFFICIENT
        )

    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(

        num_classes,

        activation="softmax",

        name="class_probabilities"

    )(x)

    model = models.Model(

        inputs,

        outputs,

        name="Tamil_Custom_CNN"

    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ============================================================
# 12. MODEL C: MobileNetV2
# ============================================================

def build_mobilenetv2(num_classes):

    inputs = layers.Input(

        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),

        name="grayscale_character_image"

    )

    # Convert input values from [0,255] to [-1,1].

    x = layers.Rescaling(

        scale=1.0 / 127.5,

        offset=-1.0

    )(inputs)

    x = make_augmentation()(x)

    # Convert grayscale to three channels.

    x = layers.Concatenate(axis=-1)([x, x, x])

    # Train MobileNetV2 from scratch.

    base_model = tf.keras.applications.MobileNetV2(

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),

        include_top=False,

        weights=None,

        alpha=1.0

    )

    base_model.trainable = True

    # Keep batch-normalization behavior controlled by Keras.

    x = base_model(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(

        128,

        activation="relu",

        kernel_regularizer=regularizers.l2(
            L2_COEFFICIENT
        )

    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(

        num_classes,

        activation="softmax",

        name="class_probabilities"

    )(x)

    model = models.Model(

        inputs,

        outputs,

        name="Tamil_MobileNetV2"

    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ============================================================
# 13. TRAINING CALLBACKS
# ============================================================

def get_callbacks(model_name):

    return [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

               patience=50,

            restore_best_weights=True,

            verbose=1

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=LR_PATIENCE,

            min_lr=MIN_LEARNING_RATE,

            verbose=1

        ),

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                OUTPUT_DIR / f"{model_name}_best.keras"
            ),

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        ),

        tf.keras.callbacks.CSVLogger(

            str(
                OUTPUT_DIR / f"{model_name}_training_log.csv"
            )

        )

    ]


# ============================================================
# 14. PER-CLASS PERFORMANCE ANALYSIS
# ============================================================

def save_per_class_metrics(

    y_true,

    y_pred,

    class_names,

    model_name

):

    rows = []

    for class_index, class_name in enumerate(class_names):

        mask = (y_true == class_index)

        total_samples = int(np.sum(mask))

        correct = int(

            np.sum(

                (y_true == class_index) &

                (y_pred == class_index)

            )

        )

        incorrect = total_samples - correct

        class_accuracy = (

            correct / total_samples

            if total_samples > 0 else 0.0

        )

        precision = precision_score(

            y_true,

            y_pred,

            labels=[class_index],

            average="macro",

            zero_division=0

        )

        recall = recall_score(

            y_true,

            y_pred,

            labels=[class_index],

            average="macro",

            zero_division=0

        )

        f1 = f1_score(

            y_true,

            y_pred,

            labels=[class_index],

            average="macro",

            zero_division=0

        )

        rows.append({

            "class_index": class_index,

            "character": class_name,

            "test_samples": total_samples,

            "correct_predictions": correct,

            "incorrect_predictions": incorrect,

            "class_accuracy": class_accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1

        })

    df = pd.DataFrame(rows)

    df.to_csv(

        OUTPUT_DIR / f"{model_name}_per_class_metrics.csv",

        index=False

    )

    print(f"\nPER-CLASS RESULTS: {model_name}")

    print(df.to_string(index=False))

    return df


# ============================================================
# 15. MODEL COMPLEXITY
# ============================================================

def get_model_complexity(model, model_name):

    total_parameters = model.count_params()

    trainable_parameters = sum(

        int(np.prod(variable.shape))

        for variable in model.trainable_weights

    )

    non_trainable_parameters = (

        total_parameters - trainable_parameters

    )

    model_path = OUTPUT_DIR / f"{model_name}_final.keras"

    model_size_mb = (

        os.path.getsize(model_path) / (1024 * 1024)

        if model_path.exists() else None

    )

    return {

        "model": model_name,

        "total_parameters": int(total_parameters),

        "trainable_parameters": int(
            trainable_parameters
        ),

        "non_trainable_parameters": int(
            non_trainable_parameters
        ),

        "model_size_MB": model_size_mb

    }


# ============================================================
# 16. EVALUATION AND REPORT GENERATION
# ============================================================

def evaluate_and_save(

    model,

    model_name,

    history,

    training_seconds

):

    print(f"\nEvaluating {model_name}...")

    # Warm-up inference.

    warmup_n = min(16, len(X_test))

    _ = model.predict(

        X_test[:warmup_n],

        batch_size=BATCH_SIZE,

        verbose=0

    )

    # Measure inference time.

    start = time.perf_counter()

    probabilities = model.predict(

        X_test,

        batch_size=BATCH_SIZE,

        verbose=0

    )

    inference_seconds = (
        time.perf_counter() - start
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # Best validation-loss epoch.

    best_epoch_index = int(

        np.argmin(
            history.history["val_loss"]
        )

    )

    train_acc = float(

        history.history["accuracy"][best_epoch_index]

    )

    val_acc = float(

        history.history["val_accuracy"][best_epoch_index]

    )

    best_val_accuracy = float(

        max(history.history["val_accuracy"])

    )

    # Overall metrics.

    metrics = {

        "model": model_name,

        "test_accuracy": accuracy_score(
            y_test, predictions
        ),

        "macro_precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "weighted_precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "weighted_recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "training_accuracy_at_best_val_loss": train_acc,

        "validation_accuracy_at_best_val_loss": val_acc,

        "best_validation_accuracy": best_val_accuracy,

        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),

        "generalization_gap": train_acc - val_acc,

        "training_time_seconds": training_seconds,

        "test_inference_time_seconds": inference_seconds,

        "average_inference_ms_per_image": (
            inference_seconds / len(X_test)
        ) * 1000.0,

        "test_samples": len(X_test),

        "epoch_of_best_validation_loss": (
            best_epoch_index + 1
        ),

        "epochs_run": len(history.history["loss"]),

        "parameter_count": model.count_params()

    }

    # Save overall metrics.

    pd.DataFrame([metrics]).to_csv(

        OUTPUT_DIR / f"{model_name}_metrics.csv",

        index=False

    )

    # Save detailed per-class results.

    save_per_class_metrics(

        y_test,

        predictions,

        class_names,

        model_name

    )

    # Classification report.

    report = classification_report(

        y_test,

        predictions,

        labels=np.arange(len(class_names)),

        target_names=class_names,

        output_dict=True,

        zero_division=0

    )

    pd.DataFrame(report).transpose().to_csv(

        OUTPUT_DIR /
        f"{model_name}_classification_report.csv"

    )

    # Confusion matrix.

    cm = confusion_matrix(

        y_test,

        predictions,

        labels=np.arange(len(class_names))

    )

    pd.DataFrame(

        cm,

        index=class_names,

        columns=class_names

    ).to_csv(

        OUTPUT_DIR /
        f"{model_name}_confusion_matrix.csv"

    )

    # Confusion matrix visualization.

    plt.figure(figsize=(15, 13))

    plt.imshow(cm, interpolation="nearest")

    plt.title(
        f"{model_name} - Test Confusion Matrix"
    )

    plt.colorbar()

    ticks = np.arange(len(class_names))

    plt.xticks(
        ticks,
        class_names,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        ticks,
        class_names,
        fontsize=7
    )

    plt.xlabel("Predicted class")

    plt.ylabel("True class")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR /
        f"{model_name}_confusion_matrix.png",

        dpi=300

    )

    plt.close()

    # Accuracy curve.

    plt.figure(figsize=(8, 5))

    plt.plot(

        history.history["accuracy"],

        label="Training accuracy"

    )

    plt.plot(

        history.history["val_accuracy"],

        label="Validation accuracy"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        f"{model_name} - Accuracy Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR /
        f"{model_name}_accuracy_curve.png",

        dpi=300

    )

    plt.close()

    # Loss curve.

    plt.figure(figsize=(8, 5))

    plt.plot(

        history.history["loss"],

        label="Training loss"

    )

    plt.plot(

        history.history["val_loss"],

        label="Validation loss"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        f"{model_name} - Loss Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR /
        f"{model_name}_loss_curve.png",

        dpi=300

    )

    plt.close()

    return metrics


# ============================================================
# 17. TRAIN ALL THREE MODELS
# ============================================================

all_results = []

complexity_results = []

experiments = [

    ("BiLSTM", build_bilstm),

    ("CNN", build_cnn),

    ("MobileNetV2", build_mobilenetv2)

]


for model_name, builder in experiments:

    print("\n" + "=" * 70)

    print(f"TRAINING MODEL: {model_name}")

    print("=" * 70)

    tf.keras.backend.clear_session()

    gc.collect()

    set_seeds(RANDOM_SEED)

    # Build model.

    model = builder(len(class_names))

    model.summary()

    # Save model architecture summary using UTF-8.

    with open(

        OUTPUT_DIR /
        f"{model_name}_model_summary.txt",

        "w",

        encoding="utf-8"

    ) as f:

        model.summary(

            print_fn=lambda line: f.write(
                line + "\n"
            )

        )

    # Training.

    start_train = time.perf_counter()

    history = model.fit(

        X_train,

        y_train,

        validation_data=(X_val, y_val),

        epochs=50,

        batch_size=BATCH_SIZE,

        class_weight=class_weight,

        callbacks=get_callbacks(model_name),

        shuffle=True,

        verbose=1

    )

    training_seconds = (
        time.perf_counter() - start_train
    )

    # Save final model.

    model.save(

        OUTPUT_DIR /
        f"{model_name}_final.keras"

    )

    # Calculate model complexity after saving.

    complexity = get_model_complexity(

        model,

        model_name

    )

    complexity_results.append(complexity)

    # Evaluate.

    metrics = evaluate_and_save(

        model,

        model_name,

        history,

        training_seconds

    )

    all_results.append(metrics)

    print("\nFINAL TEST RESULTS")

    for key, value in metrics.items():

        print(f"{key}: {value}")

    print("\nMODEL COMPLEXITY")

    print(complexity)

    del model

    del history

    tf.keras.backend.clear_session()

    gc.collect()


# ============================================================
# 18. FINAL MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(

    OUTPUT_DIR /
    "model_comparison_results.csv",

    index=False

)

complexity_df = pd.DataFrame(
    complexity_results
)

complexity_df.to_csv(

    OUTPUT_DIR /
    "model_complexity_comparison.csv",

    index=False

)


display_columns = [

    "model",

    "test_accuracy",

    "macro_precision",

    "macro_recall",

    "macro_f1",

    "training_time_seconds",

    "average_inference_ms_per_image",

    "parameter_count"

]

print("\n" + "=" * 80)

print("FINAL MODEL COMPARISON")

print("=" * 80)

print(

    results_df[display_columns].to_string(
        index=False
    )

)


# ============================================================
# 19. COMBINED COMPARISON GRAPH
# ============================================================

metric_names = [

    "test_accuracy",

    "macro_precision",

    "macro_recall",

    "macro_f1"

]

plot_df = results_df.set_index("model")[
    metric_names
]

ax = plot_df.plot(

    kind="bar",

    figsize=(11, 6)

)

ax.set_title(
    "BiLSTM vs CNN vs MobileNetV2"
)

ax.set_ylabel("Performance score")

ax.set_xlabel("Model")

ax.set_ylim(0, 1)

ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=0)

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR /
    "combined_model_comparison.png",

    dpi=300

)

plt.close()


# ============================================================
# 20. EXPERIMENT CONFIGURATION
# ============================================================

config = {

    "dataset_root": str(DATASET_ROOT),

    "expected_classes": EXPECTED_CLASSES,

    "actual_classes": len(class_names),

    "class_names": class_names,

    "total_readable_images": int(len(X)),

    "image_size": IMAGE_SIZE,

    "channels": 1,

    "split": {

        "train": int(len(X_train)),

        "validation": int(len(X_val)),

        "test": int(len(X_test)),

        "seed": RANDOM_SEED,

        "stratified": True

    },

    "training": {

        "batch_size": BATCH_SIZE,

        "maximum_epochs": MAX_EPOCHS,

        "optimizer": "Adam",

        "learning_rate": LEARNING_RATE,

        "dropout": DROPOUT_RATE,

        "internal_lstm_dropout": 0.20,

        "l2_coefficient": L2_COEFFICIENT,

        "early_stopping_monitor": "val_loss",

        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),

        "restore_best_weights": True,

        "reduce_lr_factor": 0.5,

        "reduce_lr_patience": LR_PATIENCE,

        "minimum_learning_rate": MIN_LEARNING_RATE,

        "augmentation": [

            "RandomRotation(0.02)",

            "RandomTranslation(0.04, 0.04)",

            "RandomZoom(0.05, 0.05)"

        ]

    },

    "models": [

        "row-sequence BiLSTM",

        "custom 2D CNN",

        "MobileNetV2 trained from scratch"

    ],

    "near_duplicate_threshold": DHASH_THRESHOLD,

    "tensorflow_version": tf.__version__,

    "numpy_version": np.__version__,

    "pandas_version": pd.__version__,

    "smoke_test": SMOKE_TEST

}


with open(

    OUTPUT_DIR /
    "experiment_configuration.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        config,

        f,

        indent=4

    )


# ============================================================
# 21. COMPLETION
# ============================================================

print("\n" + "=" * 80)

print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")

print("=" * 80)

print("\nModels evaluated:")

print("1. BiLSTM")

print("2. Custom CNN")

print("3. MobileNetV2")

print("\nAll results saved in:")

print(OUTPUT_DIR.resolve())

print("\nImportant:")

print("- Use actual test results in your research paper.")

print("- Inspect near-duplicate audit results.")

print("- Verify class labels and annotation quality.")

print("- Do not claim writer-independent generalization")

print("  unless writer-independent splitting was performed.")

print("- One random seed does not establish statistical stability.")

print("\nExecution completed.")
