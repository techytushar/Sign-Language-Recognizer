"""
Trains the model on the dataset
"""

import numpy as np
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import cv2


def load_data(path):
    #X is the training set, Y is the train lables
    X = np.zeros((1,100,100,1), dtype=float)
    Y = np.zeros((1,10),dtype=float)

    # reading images, converting to grayscale and adding them to array
    for i in os.listdir(path):
        print(f"Reading images for number {i}")
        for j in os.listdir(f'{path}/{i}'):
            img = cv2.imread(f'{path}/{i}/{j}')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                assert np.prod(gray.shape) == 10000
            except AssertionError as e:
                gray = cv2.resize(gray, (100,100))
            gray = gray/255
            gray = gray.reshape(1,100,100,1)
            X = np.vstack((X,gray))
            y = np.zeros((1,10))
            y[0,int(i)] = 1
            Y = np.vstack((Y,y))
    return X,Y

def preprocess_data(X,Y):    
    # removing the zeros array and shuffling the data
    X = X[1:,:,:,:]
    Y = Y[1:,:]
    X,Y = shuffle(X,Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    return X_train, X_test, Y_train, Y_test

#visualizing the images
def visualize_data(X, Y, image_number=10):
    plt.imshow(X[image_number,:,:].reshape(100,100))
    plt.show()
    print(Y[image_number])

#defining the model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding="same",  input_shape=(100, 100, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation="softmax"))
    return model

#training the model
def train_model(model):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=47, batch_size=128)
    return model

if __name__ == "__main__":
    path = "./Sign-Language-Recognizer/Dataset"
    X,Y = load_data(path)
    X_train, X_test, Y_train, Y_test = preprocess_data(X,Y)
    image_number = 10
    visualize_data(X, Y, image_number)
    model = define_model()
    model = train_model(model)
    #saving the model
    model.save("model.h5")
