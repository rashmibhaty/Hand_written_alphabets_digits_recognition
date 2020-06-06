# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:11:02 2020

@author: rashmibh
"""
import matplotlib.pyplot as plt
from keras.utils import to_categorical

MODEL_FILE="model.alphabets"
NO_EPOCHS=6
NO_CLASSES=62
BATCH_SIZE=32
IMAGE_DIM1=28
IMAGE_DIM2=28

from mnist import MNIST
emnist_data = MNIST(path='gzip\\', return_type='numpy')
emnist_data.select_emnist('byclass')

(x_train, y_train) = emnist_data.load_training()
(x_test,y_test)=emnist_data.load_testing()
    

x_train = x_train.reshape((x_train.shape[0], IMAGE_DIM1, IMAGE_DIM2, 1))
x_test = x_test.reshape((x_test.shape[0], IMAGE_DIM1, IMAGE_DIM2, 1))


# one hot encode target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#for i in range(0,20):
#  plt.imshow(x_train[i], cmap = plt.cm.binary)
#  plt.show()
  
# convert from integers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize to range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0
    

print("Training Data Shape is {}".format(x_train.shape))
print("Training Labels Shape is {}".format(y_train.shape))
print("Testing Data Shape is {}".format(x_test.shape))
print("Testing Labels Shape is {}".format(y_test.shape))


  

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout,BatchNormalization


model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(IMAGE_DIM1, IMAGE_DIM2, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(NO_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist=model.fit(x_train,y_train,epochs=NO_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

model.save(MODEL_FILE)

 # Get training and validation loss histories
train_loss = hist.history['loss']
validation_loss = hist.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(train_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'g--')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

 # Get training and validation loss histories
train_accu = hist.history['accuracy']
validation_accu= hist.history['val_accuracy']

# Visualize loss history
plt.plot(epoch_count, train_accu, 'r--')
plt.plot(epoch_count, validation_accu, 'g--')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();
