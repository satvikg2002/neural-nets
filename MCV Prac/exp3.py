import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def create_model(activation='relu', optimizer='adam'):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


activations = ['relu', 'tanh', 'sigmoid']
optimizers = ['sgd', 'adam']

for activation in activations:
    for optimizer in optimizers:
        model = create_model(activation=activation, optimizer=optimizer)
        print(
            f"\nTraining model with {activation} activation and {optimizer} optimizer...")
        model.fit(train_images, train_labels,
                  epochs=5, batch_size=128, verbose=0)
        _, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"Test accuracy: {accuracy:.4f}")
