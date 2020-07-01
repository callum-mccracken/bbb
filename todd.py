# Load the Keras NN from the h5 and json config files
# Open the file with uproot
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import uproot
import csv

architecture_filepath = "models/architecture_same_filters.json"
weights_filepath = "models/weights_same_filters.h5"
scale_filepath = "models/scaling_parameters_same_filters_3tags_only.csv"
data_filepath = "user.jagrundy.20736236._000001.MiniNTuple.root" 

# First, read the csv to get the offsets and scales for each input
offset = {"pt":[],"eta":[],"phi":[]}
scale  = {"pt":[],"eta":[],"phi":[]}
with open(scale_filepath) as csvfile:
    scale_reader = csv.reader(csvfile,delimiter=",")
    row_count=0
    for row in scale_reader:
        if row_count < 4:
            row_count+=1
            continue
        offset["pt"].append(-float(row[0])) 
        offset["eta"].append(-float(row[2])) 
        offset["phi"].append(-float(row[4])) 
        scale["pt"].append(1/np.sqrt(float(row[1])))
        scale["eta"].append(1/np.sqrt(float(row[3])))
        scale["phi"].append(1/np.sqrt(float(row[5])))

# Horribly inefficient, but whatever, it's meant to be a quick check
events = uproot.open(data_filepath)["XhhMiniNtuple"]
nn_inputs = []
event_count=0
for pts,etas,phis,tags in zip(events.array("resolvedJets_pt"),  \
                              events.array("resolvedJets_eta"), \
                              events.array("resolvedJets_phi"), \
                              events.array("resolvedJets_is_DL1r_FixedCutBEff_77")):
    # Only consider 3-tag events
    if sum(tags) != 3:
        continue
    # Loop over all the untagged jets in the event, up to 7, and save the scaled pt,eta,phi
    event_pts,event_etas,event_phis = [],[],[]
    notag_i = 0
    for i in range(len(tags)):
        if not tags[i]:
            event_pts.append(  (pts[i] +offset["pt"][notag_i]) *scale["pt"][notag_i] )
            event_etas.append( (etas[i]+offset["eta"][notag_i])*scale["eta"][notag_i] )
            event_phis.append( (phis[i]+offset["phi"][notag_i])*scale["phi"][notag_i] )
            notag_i += 1
        if notag_i == 7: 
            break
    # If we ended up with less than 7 jets, do zero-padding
    while notag_i < 7:
        event_pts.append(offset["pt"][notag_i]*scale["pt"][notag_i])
        event_etas.append(offset["eta"][notag_i]*scale["eta"][notag_i])
        event_phis.append(offset["phi"][notag_i]*scale["phi"][notag_i])
        notag_i += 1
    nn_inputs.append(np.array(event_pts+event_etas+event_phis))
    # Only do the first few events for testing
    if event_count == 3:
        break
    event_count+=1
nn_inputs = np.array(nn_inputs) # Has to be numpy array, not list

# Load and run the model
model = model_from_json(open(architecture_filepath,'r').read())
model.load_weights(weights_filepath)
scores = model.predict(nn_inputs)

for i in range(len(nn_inputs)):
    print("Event",i)
    print("Inputs",nn_inputs[i])
    print("Outputs")
    for j in range(len(scores[i])):
        print("  ",j,round(scores[i][j],4))
    print("")
