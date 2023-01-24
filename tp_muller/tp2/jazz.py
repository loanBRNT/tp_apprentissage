# TP4: Recurrent neural network for jazz improvisation

from __future__ import print_function
from music21 import *
import numpy as np
from data_utils import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU

# Load necessary data files
# X and y is our training data, the remaining files are used for music
# generation according to the predictions of the network
(X, y, n_values, chords,
 abstract_grammars, corpus, tones, 
 tones_indices, indices_tones) = load_music_data()

n_train = X.shape[0]
n_timesteps = X.shape[1]

print('number of training examples:', n_train)
print('Tx (length of sequence):', n_timesteps)
print('total # of unique values:', n_values)

# Our data consists of 58 training examples. Each training example in
# X is represented by a 20x78 matrix. This matrix represents a
# sequence of 20 notes, and each of the 20 notes can be one of 78
# possibilities (represented as a one-hot vector)
print('shape of X:', X.shape)

# Each label in y is a 78-valued one hot vector, representing the note
# that follows the 20 notes represented in X. The goal of the network
# is to predict the correct note in y given the 20 notes that precede
# it
print('Shape of Y:', y.shape)

# We will predict the correct note using a recurrent neural
# network. The RNN processes the sequence of 20 notes, which results
# in a hidden representation of the sequence. The hidden
# representation is then propagated to a dense layer with softmax
# activation, resulting in a probability distribution over the 78
# possibilities for the next note
model = Sequential()
model.add(LSTM(32, input_shape=(n_timesteps, n_values)))
model.add(Dense(78, activation = 'softmax'))

# Once our model is defined, we compile it and fit it to our data
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(X, y, batch_size=10, epochs=100)

# Once our model is fitted, we can use it to predict sequences of
# notes. These sequences are then used by the function below in order
# to generate some jazz improvisation. The function below equally
# applies a number of post-processing steps, which are beyond the
# scope of this practical session.
out_stream = generate_music(model, chords, abstract_grammars,
                            corpus, tones, tones_indices,
                            indices_tones, X)
