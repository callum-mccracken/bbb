import uproot as ur
import uproot_methods as urm
import numpy as np
import awkward
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical   
from sklearn.metrics import roc_curve

import tools

class FourJetNetwork:
    def __init__(self, events):
        """
        Takes events with 4 jets where the first 3 are correctly tagged b-jets,
        and creates a FourJetNetwork using those"""
        self.events = events
        self.ideal_model_output = events.truth[:,3]
        self.s_in = tools.scale_nn_input(events)

        # split into subsets
        train, val, test = tools.splitTVT(events.truth, trainfrac=0.7, testfrac=0.2)
        self.train = train
        self.val = val
        self.test = test
        
        # create network
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=12, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(lr=5e-5)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
    def learn(self, plot=True, epochs=400):
        """Train the neural network, make a plot of accuracy if desired"""
        self.history = self.model.fit(
            self.s_in[self.train], self.ideal_model_output[self.train],
            validation_data=(self.s_in[self.val],self.ideal_model_output[self.val]),
            epochs = epochs, batch_size = 200, verbose = 1)
        if plot:
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
        
    def evaluate(self, events=None, savename=None):
        """
        Given some (scaled) input data,
        get model's predictions of what the result should be,
        and evaluate those predictions.
        """
        if events is None:
            print("using data given when this model was created")
            truth = self.events.truth[self.test]
            tag = self.events.tag[self.test]
            data = self.s_in[self.test]
        else:
            print("using different data than when this model was created")
            truth = events.truth
            tag = events.tag
            data = tools.scale_nn_input(events)

        select = self.model.predict_classes(data)
        
        # put this in a better format
        selections = np.zeros((len(data), 4), dtype=int)
        for i, s in enumerate(select):
            selections[i][3] = s

        # and actually evaluate
        tools.evaluate_model(truth, tag, selections, savename=savename)