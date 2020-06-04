"""A place to put some functions that a bunch of notebooks might need"""

import uproot as ur
import uproot_methods as urm
import awkward
import numpy as np
from tqdm import tqdm

def open_file(filepath):
    """
        returns `X`, `y`, numpy arrays, in a format required by energyflow

        filepath:
            string, path to the .root file you wish to open

        - `X` : a three-dimensional numpy array of the jets with shape 
        `(num_data,max_num_particles,4)`.
        - `y` : a numpy array of quark/gluon jet labels (quark=`1` and gluon=`0`).

        The jets are padded with zero-particles in order to make a contiguous array.
        The particles are given as `(pt,y,phi,pid)` values, where `pid` is the
        particle's [PDG id](http://pdg.lbl.gov/2018/reviews/rpp2018-rev-monte
        -carlo-numbering.pdf). Quark jets either include or exclude $c$ and $b$
        quarks depending on the `with_bc` argument.
    """
    # open the file
    sm_file = ur.open(filepath)

    # get all data in tree format
    sm_tree = sm_file['XhhMiniNtuple']

    # get branches we care about
    branches = sm_tree.arrays(branches=[
        'resolvedJets_pt', 'resolvedJets_eta', 'resolvedJets_phi',
        'resolvedJets_E', 'resolvedJets_HadronConeExclTruthLabelID',
        'resolvedJets_is_DL1r_FixedCutBEff_77'], namedecode='utf-8')
    # Meanings of the branches we took
    """
        resolvedJets_pt:
            transverse momentum of each jet
        resolvedJets_eta:
            pseudorapidity of each jet
        resolvedJets_phi:
            azimuth of each jet (angle around beam)
        resolvedJets_E:
            energy of each jet
        resolvedJets_HadronConeExclTruthLabelID:
            see Monte Carlo paper, classification number, e.g. 5=bjet, 15=tau
        resolvedJets_is_DL1r_FixedCutBEff_77:
            whether or not a jet has been b-tagged
    """

    # convert to "pandas-dataframe-style" table
    table = awkward.Table(branches)

    # use pt, eta, phi, E to make LorentzVectors for each jet
    lv = urm.TLorentzVectorArray.from_ptetaphie(table['resolvedJets_pt'],
                                                table['resolvedJets_eta'],
                                                table['resolvedJets_phi'],
                                                table['resolvedJets_E'])
    # add LVs to table
    table['resolved_lv'] = lv

    # add a "truth" category to the table,
    # based on whether or not a jet is a b-jet or not
    table['truth'] = (table['resolvedJets_HadronConeExclTruthLabelID'] == 5)
    table['truth'] = table['truth'].astype(np.int32)

    # and for convenience rename resolvedJets_is_DL1r_FixedCutBEff_77 to "tag"
    table['tag'] = table['resolvedJets_is_DL1r_FixedCutBEff_77']

    # a few easier names to use later
    # number of jets overall
    table['njets'] = awkward.AwkwardArray.count(table.resolved_lv.pt)
    # number of b jets in each event
    table['nbjets'] = awkward.AwkwardArray.count_nonzero(table['truth'])
    # number of b tags in each event
    table['nbtags'] = awkward.AwkwardArray.count_nonzero(table['tag'])

    # sort table by tag
    indices = table["tag"].argsort()
    s_table = awkward.Table()
    for label in ['resolved_lv', 'truth', 'tag']:
        s_table[label] = table[label][indices]

    max_njets = max(table.njets)

    print("padding arrays")
    pt_arr = np.array([np.concatenate((pt, [0]*(max_njets-len(pt)))) for pt in s_table.resolved_lv.pt])
    eta_arr = np.array([np.concatenate((eta, [0]*(max_njets-len(eta)))) for eta in s_table.resolved_lv.eta])
    phi_arr = np.array([np.concatenate((phi, [0]*(max_njets-len(phi)))) for phi in s_table.resolved_lv.phi])
    tag_arr = np.array([np.concatenate((tag, [0]*(max_njets-len(tag)))) for tag in s_table.tag])
    truth_arr = np.array([np.concatenate((tru, [0]*(max_njets-len(tru)))) for tru in s_table.truth])
    print('done padding')

    # ensure the first 3 jets are tagged correctly
    events = np.ones((len(pt_arr)), dtype=bool)
    print(np.count_nonzero(events), 'events total')
    events[tag_arr[:,0] == 0] = 0
    events[tag_arr[:,1] == 0] = 0
    events[tag_arr[:,2] == 0] = 0
    events[truth_arr[:,0] == 0] = 0
    events[truth_arr[:,1] == 0] = 0
    events[truth_arr[:,2] == 0] = 0
    print(np.count_nonzero(events), 'events after ensuring first 3 are correctly tagged')

    # and ensure no other jets are tagged
    events[np.count_nonzero(tag_arr[:,4:], axis=1) > 0] = 0
    print(np.count_nonzero(events), 'events after ensuring we only have 3 tags')

    # 1 = should've been tagged, wasn't. 0 = correctly tagged (/ not tagged)
    untagged = np.logical_xor(truth_arr, tag_arr).astype(int)
    n_untagged = np.count_nonzero(untagged, axis=1)

    # ensure we only have one untagged jet
    events[n_untagged > 1] = 0
    print(np.count_nonzero(events), 'events after ensuring there is at most 1 untagged jet')

    # create our data with only the filtered events
    pt_arr = pt_arr[events]
    eta_arr = eta_arr[events]
    phi_arr = phi_arr[events]
    tag_arr = tag_arr[events]
    truth_arr = truth_arr[events]

    X = np.zeros((len(pt_arr), max_njets, 4), dtype=float)
    X[:,:,0] = pt_arr
    X[:,:,1] = eta_arr
    X[:,:,2] = phi_arr
    X[:,:,3] = tag_arr  # should be all zeros at this point

    # where does the untagged jet occur?
    missed_jet_events, missed_jet_index = np.where(untagged[events]==1)

    # njets = no 4th jet, the rest means set that index's jet = 4th jet
    missed_jet = np.zeros(len(pt_arr), dtype=int)
    missed_jet += max_njets  # don't pick a jet
    missed_jet[missed_jet_events] = missed_jet_index  # unless you should

    missed_jet_arr = np.zeros((len(pt_arr), max_njets-3+1), dtype=int)

    for i, m in enumerate(missed_jet):
        missed_jet_arr[i][m-3] = 1

    y = missed_jet_arr
    return X, y

if __name__ == "__main__":
    open_file("/home/callum/Documents/bbb/user.jagrundy.20736236._000001.MiniNTuple.root")
