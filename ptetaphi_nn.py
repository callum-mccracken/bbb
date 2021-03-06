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
from keras.models import load_model
from keras.models import model_from_json
import json
import random
import tools

class PtEtaPhiNN:
    def __init__(self, events, print_summary=False, model=None, load=None, njets=10, save_csv=None, chop=0, dropout=0.1, fold=None):
        """
        A class for neural networks that take raw pt, eta, phi values
        Note we assuume events is rectangular (all same # of jets)

        events:
            uproot table, contains event data
        print_summary:
            bool, whether or not to print a summary of the model's structure,
            default: False
        model:
            keras model, so you can create a PtEtaPhiNN with your own model,
            default: None, usually models are created in this class.
        load:
            tuple of strings, (json_path, h5_path), path to files required to
            create a neural network (json for architecture, h5 for weights)
        njets:
            int, number of jets to include in each event, default: 10
        save_csv:
            string, if supplied, save csv with scaling parameters to [save_csv].csv,
            default: None
        chop:
            int, number of jets to cut off from the start, e.g. set to 3 if you
            have 3 jets where you want to ignore the information from those three jets,
            default: 0
        dropout:
            float < 1, fraction of dropout to apply between intermediate layers of the network,
            default: 0.1
        """
        self.chop = chop
        self.events = events
        self.njets = njets
        # 1 = should've been tagged, wasn't. 0 = correctly tagged (/ not tagged)
        untagged = np.logical_xor(events.truth, events.tag).astype(int)

        # where does the untagged jet occur?
        missed_jet_events, missed_jet_index = np.where(untagged==1)

        # njets = no 4th jet, the rest means set that index's jet = 4th jet
        missed_jet = np.zeros(len(untagged), dtype=int)
        missed_jet += njets  # don't pick a jet
        missed_jet[missed_jet_events] = missed_jet_index  # unless you should

        missed_jet_arr = np.zeros((len(events), njets-3+1), dtype=int)

        for i, m in enumerate(missed_jet):
            missed_jet_arr[i][m-3] = 1

        self.ideal_model_output = missed_jet_arr

        self.s_in = tools.scale_nn_input(events, chop=chop, save_csv=save_csv)
        
        self.fold = fold

        # split into subsets
        if fold is None:
            # these are lists of indices
            train, val, test = tools.splitTVT(events, trainfrac=0.7, testfrac=0.2)
        else:
            train = []
            val = []
            test = None
            # start at index fold and take every 5th element for val
            print('k-folding: every 5th element starting at', fold)
            
            shuffled_indices = list(range(len(events)))
            random.shuffle(shuffled_indices)
            events = events[shuffled_indices]

            for i in range(len(events)):
                if i % 5 == fold:
                    val.append(i)
                else:
                    train.append(i)

        self.train = train
        self.val = val
        self.test = test

        # create network
        if load:
            jsonfile, h5file = load
            print("Loading model... \nUsing architecture:",jsonfile,
                  "\nand weights:", h5file)
            with open(jsonfile,'rb') as f:
                model_json = f.read()
            self.model = model_from_json(model_json)
            self.model.load_weights(h5file)
        elif model is None:
            print("creating default model")
            if dropout != 0:
                self.model = Sequential([
                    Dense(3*(njets-chop), input_dim=3*(njets-chop),
                          kernel_initializer='normal', activation='relu'),
                    Dense(700, activation='relu'),
                    Dropout(dropout),
                    Dense(500, activation='relu'),
                    Dropout(dropout),
                    Dense(300, activation='relu'),
                    Dropout(dropout),
                    Dense(100, activation='relu'),
                    Dropout(dropout),
                    Dense(50, activation='relu'),
                    Dense(njets-3+1,  # - 3 correctly tagged + 1 for no jet
                          kernel_initializer='normal', activation='softmax')
                ])
            else:
                self.model = Sequential([
                    Dense(3*(njets-chop), input_dim=3*(njets-chop),
                          kernel_initializer='normal', activation='relu'),
                    Dense(700, activation='relu'),
                    Dense(500, activation='relu'),
                    Dense(300, activation='relu'),
                    Dense(100, activation='relu'),
                    Dense(50, activation='relu'),
                    Dense(njets-3+1,  # - 3 correctly tagged + 1 for no jet
                          kernel_initializer='normal', activation='softmax')
                ])

            optimizer = Adam(lr=5e-5)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer, metrics=['acc'])
        else:
            print("using input model")
            self.model = model
        if print_summary:
            self.model.summary()

    def learn(self, plot=True, epochs=200):
        """
        Train the neural network, make a plot of accuracy if desired
        
        plot:
            boolean, whether to make a plot at the end of accuracy,
            default: True
        epochs:
            int, number of epochs for which to train
        """
        self.history = self.model.fit(
            self.s_in[self.train], self.ideal_model_output[self.train],
            validation_data=(
                self.s_in[self.val],self.ideal_model_output[self.val]),
            epochs = epochs, batch_size = 200, verbose = 1)
        if plot:
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

    def evaluate(self, events=None, weights=None, output="pretty", savename=None, save_csv=None):
        """
        Given some (scaled) input data,
        get model's predictions of what the result should be,
        and evaluate those predictions.

        events:
            uproot table, contains event data, if you want to evaluate
            on a whole different dataset than the one used to create
            this network. Default: None (use data provided to make network)
        output:
            string, what kind of output to print after evaluating,
            options: 
                "pretty" for a nice graphical table (default)
                "ascii" for an ascii-style table
                None for no output
        savename:
            string, if output="pretty", save the table as table_{savename}.png,
            default: None, no table output
        save_csv:
            string, save scaling parameters to [save_csv].csv,
            default None --> no csv saved 
        """
        if events is None:
            #print("using data given when this model was created")
            if self.test is not None:
                truth = self.events.truth[self.test]
                tag = self.events.tag[self.test]
                data = self.s_in[self.test]
            else:
                truth = self.events.truth[self.val]
                tag = self.events.tag[self.val]
                data = self.s_in[self.val]
        else:
            #print("using different data than when this model was created")
            truth = events.truth
            tag = events.tag
            data = tools.scale_nn_input(events, chop=self.chop, save_csv=save_csv)
            #print('scaled', self.s_in[3,:])

        nn_score = self.model.predict(data)

        if output == "nn_score":
            return nn_score

        #print(type(nn_score), nn_score.shape)
        if len(nn_score) == 1:
            print("model scores")
            for i, s in enumerate(nn_score[0,:]):
                print(i, f"{s:.4f}")
        select=np.argmax(nn_score,axis=-1)
        # put this in a better format
        selections = np.zeros((len(data), len(truth[0])+1), dtype=int)
        for i, s in enumerate(select):
            selections[i][s+3] = 1
        # chop off the last "no selection" jet
        selections = selections[:,:-1]

        # and actually evaluate
        tools.evaluate_model(truth, tag, selections, weights=weights, output=output, savename=savename)
        return selections
