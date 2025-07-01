import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define dataset path
dataset_path = "/Users/vishnu/Desktop/model/Brain tumor/Training"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Parameters
batch_size = 32
img_size = (256, 256)
input_shape = img_size + (3,)
epochs = 50
n_splits = 5

# Data Augmentation
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load dataset into arrays
train_data, train_labels = [], []
categories = ["glioma", "meningioma", "notumor", "pituitary"]

for idx, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    img_files = os.listdir(category_path)
    
    for img_file in img_files:
        img_path = os.path.join(category_path, img_file)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        train_data.append(img_array)
        train_labels.append(idx)

# Convert to NumPy arrays
train_data = np.array(train_data, dtype='float32') / 255.0
train_labels = to_categorical(np.array(train_labels), num_classes=4)

# KFold Cross-Validation
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
    print(f"Training on fold {fold + 1}...")
    
    X_train, X_val = train_data[train_idx], train_data[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    y_train_labels = np.argmax(y_train, axis=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train_labels)
    X_train_resampled = X_train_resampled.reshape(-1, img_size[0], img_size[1], 3)
    y_train_resampled = to_categorical(y_train_resampled, num_classes=4)
    
    # Load Pretrained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    # Build Model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    final_output = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=final_output)
    
    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
    
    # Train Model
    model.fit(X_train_resampled, y_train_resampled, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=[lr_schedule], verbose=1)
    
    all_models.append(model)

# Save all models into a single .h5 file
final_model = all_models[0]
for i in range(1, len(all_models)):
    final_model.set_weights([(w1 + w2) / 2 for w1, w2 in zip(final_model.get_weights(), all_models[i].get_weights())])

final_model.save("brain_tumor_mobilenet.h5")
print("Final model saved as brain_tumor_mobilenet.h5")
