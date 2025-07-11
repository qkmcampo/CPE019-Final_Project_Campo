# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Set base directory (modify if needed) ---
base_dir = "/content/drive/MyDrive/Emtech2/Fruits"

# --- Safety Check: Does folder exist? ---
if not os.path.exists(base_dir):
    print("❌ Dataset folder not found!")
    print("➡ Please make sure this path exists in your Google Drive:")
    print(base_dir)
    print("\n📂 The folder structure should look like this:")
    print("Fruits/")
    print("├── Apple/")
    print("├── Banana/")
    print("├── Mango/")
    print("└── ...")
    raise SystemExit("⚠️ Stopping: Dataset path not found.")

# --- Image Parameters ---
img_height, img_width = 150, 150
batch_size = 32

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% validation
)

# --- Load Training and Validation Data ---
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- Build Transfer Learning Model ---
def create_transfer_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Model Training ---
model = create_transfer_model(num_classes=train_generator.num_classes)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop]
)

# --- Plot Training History ---
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# --- Save Model ---
model_save_path = '/content/drive/MyDrive/Emtech2/fruits_model.h5'
model.save(model_save_path, save_format='h5')
print(f"✅ Model saved at: {model_save_path}")
