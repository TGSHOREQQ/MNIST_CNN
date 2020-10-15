# Recreating CNN self, with description and explanations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Import Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# dividing by 255 rescale values (between 0 & 1)
x_train, x_test = x_train / 255.0, x_test / 255.0
# Expand dims -> insert new axis at -1 position for both
#
x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)

# CNN Model
# images are 28x28 and 1 number representing greyscale
# Sequential model for simple stack of layers -> each layer here has 1 i/p & o/p tensor
model = keras.Sequential(
    [
        # Convolutional base - common pattern of Conv2D and MaxPooling2D layers
        # Conv2D - creates convolution kernal. matrix for performing image processing task
        # 28 filters -
        # Kernal Size - 3x3 size of 2d conv window
        layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(56, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(56, (3, 3), activation='relu'),

        layers.Flatten(),
        layers.Dense(56, activation='relu'),
        layers.Dense(10)
    ]
)

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(test_acc)
