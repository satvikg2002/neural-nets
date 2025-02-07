import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))

# Freeze the layers in the base model so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = Sequential([
    base_model,
    Flatten(),
    Dense(3, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('train',
                                                    target_size=(224, 224),
                                                    batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory('test',
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary')

# Train the model
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=5,
          validation_data=test_generator,
          validation_steps=len(test_generator))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator,
                                     steps=len(test_generator))
print("Test accuracy:", test_acc)
