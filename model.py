import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

gener = ImageDataGenerator(rescale = 1.0/255)
train_data = gener.flow_from_directory('augmented-data/train', color_mode = 'grayscale', target_size = (256, 256), class_mode = 'categorical', batch_size = 3)
test_data = gener.flow_from_directory('augmented-data/test', color_mode = 'grayscale', target_size = (256, 256), class_mode = 'categorical', batch_size = 3)

model = Sequential()

model.add(keras.layers.Input(shape = (256, 256,1)))
model.add(keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'valid', strides = 1))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),   strides=(3, 3), padding='valid'))
model.add(keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'valid', strides = 1))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),   strides=(3, 3), padding='valid'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation = 'softmax'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

history = model.fit(train_data, steps_per_epoch = len(train_data) / 3, epochs = 40, validation_data = test_data, validation_steps = len(test_data) / 3)


# Do Matplotlib extension below
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()
 
# use this savefig call at the end of your graph instead of using plt.show()
plt.savefig('static/images/my_plots.png')
