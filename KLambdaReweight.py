# Derives new mc scale factors assuming an input file with a mhh function
import numpy as np
import pandas
import uproot

tree = uproot.open("SM_HH_Mar20.2_MC16d_nominal.root")["signal"] # Open signal tree of NTuple
smnr = tree.pandas.df() # Convert to pandas dataframe

# Open the lambda weights file
lambdaFile="LambdaWeightFile.root"
f = uproot.open(lambdaFile)
for ttree in f.keys():
    tree = uproot.open(lambdaFile)[ttree]
    idx = np.digitize(smnr['m_hh'],tree.edges) 
    l = ttree.decode()[13:][:-2]
    smnr['w_lambda{}'.format(int(l))] = tree.allvalues[idx] * smnr['mc_sf']


