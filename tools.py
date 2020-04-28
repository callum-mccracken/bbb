"""A place to put some functions that a bunch of notebooks might need"""

import uproot as ur
import uproot_methods as urm
import awkward
import numpy as np
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

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


def evaluate_model(truths, tags, selections):
    """
        Given a list of events and a list of selections, how often did we...
        1. make the correct selection of jet?
        2. pick an incorrect jet?
        3. pick a jet when we shouldn't have?
        4. give up?

        Each input should be an array of n events
        truths = np.array([[1, 1, 1, 1, 0]])# was each jet truly a b jet?
        tags = np.array([[1, 1, 1, 0, 0]])  # was each jet tagged as a b jet?
        selections = np.array([[0, 0, 0, 1, 0]])  # which jet is the 4th?
    """

    # counters for when conditions 1 2 3 4 are satisfied
    count_1 = count_2 = count_3 = count_4 = 0

    # counters for situations when 1 2 3 4 apply
    count_1_total = count_2_total = count_3_total = count_4_total = 0
    
    n_events = len(truths)
    for i in tqdm(range(n_events)):
        selection = selections[i]
        truth = truths[i]
        tag = tags[i]
        
        selection_index = np.where(selection == 1)[0]
        n_selections = len(selection_index)
        #print(n_selections, "seclections at indices", selection_index)

        untagged = np.logical_xor(truth, tag).astype(int)
        n_untagged = np.count_nonzero(untagged)
        #print(n_untagged, "true jets that were untagged")

        if n_untagged == 0:  # should not have made any selection
            count_3_total += 1
            if n_selections != 0:  # made a selection, should not have
                count_3 += 1
            else:  # good call, no selection
                pass

        elif n_untagged == 1:  # if we should have made 1 selection
            count_1_total += 1
            count_2_total += 1

            if n_selections == 0:  # no selection
                # failing to pick a jet counts as picking the wrong jet I guess?
                count_2 += 1

            elif n_selections == 1:  # 1 selection
                # was it the right one?
                right_selection = bool(truth[selection_index] == 1)
                if right_selection:
                    count_1 += 1  # correct selection
                else:
                    count_2 += 1  # wrong selection when there was a right answer

            else:
                raise ValueError("Why did you select more than 1 jet?")

        else:  # there were 2 or more untagged jets
            count_4_total += 1  # should have given up
            count_4 += 1  # did
    print('\r')  # remove progress bar

    percent_1 = count_1/count_1_total * 100
    percent_2 = count_2/count_2_total * 100
    percent_3 = count_3/count_3_total * 100
    percent_4 = count_4/count_4_total * 100

    fixed = count_1 / n_events * 100
    broken = count_2 / n_events * 100
    unchanged = (count_3 + count_4) / n_events * 100

    output_str = f"""
    There were {count_1_total} situations where we should have picked a jet,
    and {count_1} of those jets were picked correctly ({percent_1:.2f}%).

    We picked an incorrect jet {count_2} times ({percent_2:.2f}%).

    There were {count_3_total} times we should not have picked a jet,
    and {count_3} of those were handled correctly ({percent_3:.2f}%).

    There were {count_4} situations where we weren't sure what to do.

    In terms of overall percentages of events "fixed"
    (i.e. where we tagged a 4th jet), we have the following:

    Events correctly fixed = {fixed:.2f}%

    Events incorrectly fixed = {broken:.2f}%

    Events where nothing was done (no tag/gave up) = {unchanged:.2f}%
    """
    print(output_str)
    #print(count_1, count_2, count_3, count_4)
    #print(count_1_total, count_2_total, count_3_total, count_4_total)

