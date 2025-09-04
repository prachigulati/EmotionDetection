# train_emotion_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30,
                                width_shift_range=0.2, height_shift_range=0.2,
                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load FER-2013 dataset (download from Kaggle and place in ./fer2013/)
train_set = train_datagen.flow_from_directory('fer2013/train', 
                                            target_size=(48,48), color_mode='grayscale',
                                            batch_size=64, class_mode='categorical')

test_set = test_datagen.flow_from_directory('fer2013/test', 
                                            target_size=(48,48), color_mode='grayscale',
                                            batch_size=64, class_mode='categorical')

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_set, validation_data=test_set, epochs=25)

# Save model
model.save("emotion_model.h5")
print("✅ Model saved as emotion_model.h5")
