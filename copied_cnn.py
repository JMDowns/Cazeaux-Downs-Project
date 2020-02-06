from os import system
from sys import stdout
import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D


def life_step(X):
    """
    'Game of Life' logic, from: 
    https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    """
    live_neighbors = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (live_neighbors == 3) | (X & (live_neighbors == 2)).astype(int)

def generate_frames(num_frames, board_shape=(100,100), prob_alive=0.15):
    """
    Generates `num_frames` random game boards with a particular shape and a predefined 
    probability of each cell being 'alive'.
    """
    
    return np.array([
        np.random.choice([False, True], size=board_shape, p=[1-prob_alive, prob_alive])
        for _ in range(num_frames)
    ]).astype(int)

def render_frame(frame1):
    system('clear')
    for row in frame1:
            for element in row:
                if (element):
                    stdout.write("X")
                else:
                    stdout.write(".")
            stdout.write("\n")
    time.sleep(.1)

def reshape_input(X):
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

def generate_dataset(num_frames, board_shape, prob_alive):
    X = generate_frames(num_frames, board_shape=board_shape, prob_alive=prob_alive)
    X = reshape_input(X)
    y = np.array([
        life_step(frame) 
        for frame in X
    ])
    return X, y

def train(model, X_train, y_train, X_val, y_val, batch_size=50, epochs=2, filename_suffix=''):
    model.fit(
        X_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_data=(X_val, y_val)
    )
    
    with open('cgol_cnn{}.json'.format(filename_suffix), 'w') as file:
        file.write(model.to_json())
    model.save_weights('cgol_cnn{}.h5'.format(filename_suffix))




train_size = 70000
val_size   = 10000
test_size  = 20000
    
board_shape = (20, 20)
board_size = board_shape[0] * board_shape[1]
probability_alive = 0.15

frames = generate_frames(10, board_shape=board_shape, prob_alive=probability_alive)
frames.shape # (num_frames, board_w, board_h)

print("Sample frame vs. next step:")
render_frame(frames[0])

print("Training Set:")
X_train, y_train = generate_dataset(train_size, board_shape, probability_alive)
print(X_train.shape)
print(y_train.shape)

print("Validation Set:")
X_val, y_val = generate_dataset(val_size, board_shape, probability_alive)
print(X_val.shape)
print(y_val.shape)

print("Test Set:")
X_test, y_test = generate_dataset(test_size, board_shape, probability_alive)
print(X_test.shape)
print(y_test.shape)


def pad_input(X):
    return reshape_input(np.array([
        np.pad(x.reshape(board_shape), (1,1), mode='wrap')
        for x in X
    ]))

X_train_padded = pad_input(X_train)
X_val_padded = pad_input(X_val)
X_test_padded = pad_input(X_test)

print(X_train_padded.shape)
print(X_val_padded.shape)
print(X_test_padded.shape)



# CNN Properties
filters = 50
kernel_size = (3, 3) # look at all 8 neighboring cells, plus itself
strides = 1
hidden_dims = 100

model_padded = Sequential()
model_padded.add(Conv2D(
    filters, 
    kernel_size,
    padding='valid',
    activation='relu',
    strides=strides,
    input_shape=(board_shape[0] + 2, board_shape[1] + 2, 1)
))
model_padded.add(Dense(hidden_dims))
model_padded.add(Dense(1))
model_padded.add(Activation('sigmoid'))

model_padded.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_padded.summary()

train(
    model_padded, 
    X_train_padded, y_train, X_val_padded, y_val, 
    filename_suffix='_padded'
)
