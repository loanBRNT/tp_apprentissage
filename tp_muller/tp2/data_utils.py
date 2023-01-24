import numpy as np

from preprocess import *
from tensorflow import keras
from grammar import *
from qa import *

def load_music_data():
    chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    X, y, N_tones = data_processing(corpus, tones_indices, 20)   
    return (X, y, N_tones, chords,
            abstract_grammars, corpus, tones,
            tones_indices, indices_tones) 
    
def data_processing(corpus, values_indices, max_len):
    # cut the corpus into semi-redundant sequences of max_len values
    step = 3 
    N_values = len(set(corpus))
    sentences = []
    next_values = []
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])

    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, values_indices[val]] = 1
        y[i, values_indices[next_values[i]]] = 1
    return X, y, N_values


def generate_music(inference_model, chords, abstract_grammars, corpus,
                   tones, tones_indices, indices_tones, X):
    
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0                  # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)  # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a
    # sequence of tones and use the current chords to convert it into
    # actual sounds
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate
            # offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model, X)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
        
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common
        # choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds (chord %s)' \
              % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to file
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    return out_stream

def predict_and_sample(model, X):
    allInds = []
    allResults = []
    x = np.zeros((1, X.shape[1], X.shape[2]))
    x[0] = X[5]
    for i in range(50):
        pred = model.predict(x)[0]
        nextind = np.argmax(pred)
        result = keras.utils.to_categorical(nextind, num_classes = X.shape[2])
        allInds.append(nextind)
        allResults.append(result)
        x = np.concatenate((x[:,1:,:],
                            result[np.newaxis, np.newaxis, :]), axis = 1)
    allInds = np.array([np.array(i) for i in allInds])
    allResults = np.array(allResults)
    return allResults, allInds
