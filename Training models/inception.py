import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Define dataset path
dataset_path = "/Users/vishnu/Desktop/model/Brain_Data_Organised"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Parameters
batch_size = 32
img_size = (128, 128)  # Updated image size
input_shape = img_size + (3,)
epochs = 50

# Dynamically determine the number of classes
categories = [cat for cat in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cat))]
num_classes = len(categories)
print(f"Number of classes detected: {num_classes}")

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load dataset into arrays
data, labels = [], []
for category_idx, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)

    for img_file in os.listdir(category_path):
        img_path = os.path.join(category_path, img_file)

        # Ensure it's an image file
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        data.append(img_array)
        labels.append(category_idx)

# Convert data and labels to NumPy arrays
data = np.array(data, dtype='float32') / 255.0  # Normalize images
labels = np.array(labels)  # Ensure labels are an integer array
labels = to_categorical(labels, num_classes=num_classes)  # One-hot encode labels

# KFold Cross-Validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store Metrics and Histories
fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
histories = []

# SMOTE and KFold
for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    print(f"Training on fold {fold + 1}...")

    # Split data
    X_train, X_val = data[train_idx], data[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Reshape for SMOTE
    y_train_labels = np.argmax(y_train, axis=1)  # Convert one-hot to labels for SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train_labels)

    # Reshape back to image format
    X_train_resampled = X_train_resampled.reshape(-1, img_size[0], img_size[1], 3)
    y_train_resampled = to_categorical(y_train_resampled, num_classes=num_classes)

    # Load Pretrained InceptionV3
    inception_base = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)
    inception_base.trainable = False  # Freeze base layers

    # Build Model
    x = inception_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    final_output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

    model = Model(inputs=inception_base.input, outputs=final_output)

    # Callbacks
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
    checkpoint = ModelCheckpoint(
        filepath=f'best_model_fold_{fold + 1}.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model
    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_schedule, checkpoint],
        verbose=1
    )

    histories.append(history)

    # Evaluate Model
    model.load_weights(f'best_model_fold_{fold + 1}.weights.h5')
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true_classes = np.argmax(y_val, axis=1)

    accuracy = accuracy_score(y_val_true_classes, y_val_pred_classes)
    precision = precision_score(y_val_true_classes, y_val_pred_classes, average='weighted')
    recall = recall_score(y_val_true_classes, y_val_pred_classes, average='weighted')
    f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='weighted')

    print(f"Fold {fold + 1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1s.append(f1)

# Save Final Model with Best Weights
final_model_path = "/Users/vishnu/Desktop/model/final_model_inceptionv3.h5"
model.save(final_model_path)
print(f"Final trained model saved at: {final_model_path}")

# Summarize Metrics
print("\nCross-Validation Metrics:")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Average Precision: {np.mean(fold_precisions):.4f}")
print(f"Average Recall: {np.mean(fold_recalls):.4f}")
print(f"Average F1 Score: {np.mean(fold_f1s):.4f}")

# Plot training and validation accuracy and loss for each fold
for fold, history in enumerate(histories):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
