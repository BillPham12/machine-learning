import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt
import get_data_for_cnn
from datetime import datetime
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
data,labels = get_data_for_cnn.get_data()

data, labels = shuffle(data,labels,random_state = 1)

model = Sequential()

model.add(Conv1D(filters=100, kernel_size=10, activation='relu', input_shape=(400,100)))
model.add(Conv1D(100, 11, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(100, 11, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(625, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(data, labels, batch_size=128, epochs=20, verbose=1,validation_split = 0.2)

print(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('The result of the third strategy (GloVe word embedding)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
