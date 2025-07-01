import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Define dataset path
dataset_path = "/Users/vishnu/Desktop/model/alziemers/AugmentedAlzheimerDataset"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Parameters
batch_size = 32
img_size = (256, 256)
input_shape = img_size + (3,)
epochs = 35
num_classes = 4  # Four classes

# Class Labels
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

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

# Load dataset
train_data, train_labels = [], []
for idx, category in enumerate(class_names):
    category_path = os.path.join(dataset_path, category)
    img_files = os.listdir(category_path)
    for img_file in img_files:
        img_path = os.path.join(category_path, img_file)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        train_data.append(img_array)
        train_labels.append(idx)

# Convert data to numpy arrays
train_data = np.array(train_data, dtype='float32') / 255.0  # Normalize images
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels, num_classes=num_classes)

# KFold Cross-Validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Train and Evaluate MobileNetV2
print("Training MobileNetV2...")
histories = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
    print(f"Fold {fold + 1}...")
    X_train, X_val = train_data[train_idx], train_data[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    y_train_labels = np.argmax(y_train, axis=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train_labels)
    X_train_resampled = X_train_resampled.reshape(-1, img_size[0], img_size[1], 3)
    y_train_resampled = to_categorical(y_train_resampled, num_classes=num_classes)

    # Load Pretrained MobileNetV2
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    # Custom Layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    final_output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
    model = Model(inputs=base_model.input, outputs=final_output)

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model
    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    histories.append(history)

# Save final MobileNetV2 model
final_model = Model(inputs=base_model.input, outputs=final_output)
final_model.save("final_model_alzeimer_MobileNetV2.h5")
print("Final MobileNetV2 model saved as final_model_alziemers_MobileNetV2.h5")

# Plot training history
for fold, history in enumerate(histories):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'MobileNetV2 - Fold {fold + 1} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'MobileNetV2 - Fold {fold + 1} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
