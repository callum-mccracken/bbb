"""A place to put some functions that a bunch of notebooks might need"""

import uproot as ur
import uproot_methods as urm
import awkward
import numpy as np
from sklearn.model_selection import ShuffleSplit


def open_file(filepath, sort_by=None):
    """
        open_file returns an awkward table, sorted by the parameter sort_by

        filepath:
            string, path to the .root file you wish to open

        sort_by:
            string, parameter with which to sort the table,
            currently only "tag", "phi", "eta", "pt" are options
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

    # sort table if needed
    if sort_by is not None:
        print("sorting data by", sort_by)
        # indices of table, sorted by whatever variable is provided
        if sort_by == "pt":
            indices = table['resolved_lv'].pt.argsort()
        elif sort_by == "eta":
            indices = table['resolved_lv'].eta.argsort()
        elif sort_by == "phi":
            indices = table['resolved_lv'].phi.argsort()
        elif sort_by == "tag":
            indices = table["tag"].argsort()
        else:
            raise ValueError(f"sort_by={sort_by} is not yet supported")

        # make new sorted table with attributes we need, properly sorted
        s_table = awkward.Table()
        for label in ['resolved_lv', 'truth', 'tag']:
            s_table[label] = table[label][indices]
    else:
        print("not sorting data")
        s_table = table

    # a few easier names to use later

    # number of jets overall
    s_table['njets'] = awkward.AwkwardArray.count(s_table.resolved_lv.pt)
    # number of b jets in each event
    s_table['nbjets'] = awkward.AwkwardArray.count_nonzero(s_table['truth'])
    # number of b tags in each event
    s_table['nbtags'] = awkward.AwkwardArray.count_nonzero(s_table['tag'])
 
    return s_table

def splitTVT(input, trainfrac = 0.8, testfrac = 0.2):
    """
        splits data into training, validation, and test subsets

    """
    # by default no validation, be sure to have validation later!
    valfrac = 1.0 - trainfrac - testfrac
    
    train_split = ShuffleSplit(n_splits=1, test_size=testfrac + valfrac, random_state=0)
    # advance the generator once with the next function
    train_index, testval_index = next(train_split.split(input))  

    if valfrac > 0:
        testval_split = ShuffleSplit(
            n_splits=1, test_size=valfrac / (valfrac+testfrac), random_state=0)
        test_index, val_index = next(testval_split.split(testval_index)) 
    else:
        test_index = testval_index
        val_index = []

    return train_index, val_index, test_index