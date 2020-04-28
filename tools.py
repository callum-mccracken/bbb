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

    - detect the right jet
    - pick a wrong jet
    - not pick a jet when we should have
    - correctly do nothing to a 3-jet event
    - incorrectly add a 4th jet to a 3-jet event

    truths:
        numpy array, 1 for real b-jet, 0 for not
    tags:
        numpy array, 1 for tagged, 0 for not
    selections:
        numpy array, 1 for selected, 0 for not

    Example for one event::

        truths = [1, 1, 1, 1, 0]
        tags = [1, 1, 1, 0, 0]
        selections = [0, 0, 0, 1, 0]
    """

    # counters for when conditions are satisfied
    
    # if 4 jets exist, use these for picking right, picking wrong, or ignoring
    right_pick = wrong_pick_4 = wrong_ignore = 0
    # if 3 jets exist, use these
    wrong_pick_3 = right_ignore = 0
    # for other events
    give_up = 0

    # total number of events
    n_events = len(truths)
    for i in tqdm(range(n_events)):
        selection = selections[i]
        truth = truths[i]
        tag = tags[i]

        selection_index = np.where(selection == 1)[0]
        n_selections = len(selection_index)

        untagged = np.logical_xor(truth, tag).astype(int)
        n_untagged = np.count_nonzero(untagged)

        if n_untagged == 0:  # should not have made any selection
            if n_selections != 0:  # made a selection, should not have
                wrong_pick_3 += 1
            else:  # good call, no selection
                right_ignore += 1
        elif n_untagged == 1:  # if we should have made 1 selection
            if n_selections == 0:  # no selection, bad
                wrong_ignore += 1
            elif n_selections == 1:  # 1 selection, good
                # was it the right jet though?
                right_selection = bool(truth[selection_index] == 1)
                if right_selection:
                    right_pick += 1
                else:
                    wrong_pick_4 += 1
            else:
                raise ValueError("Why did you select more than 1 jet?")
        else:  # there were 2 or more untagged jets
            give_up += 1
    print('\r')  # remove progress bar

    # ignore "give up" events in the stats at the end
    n = n_events - give_up

    # pc = percent
    right_pick_pc = right_pick / n * 100
    wrong_pick_3_pc = wrong_pick_3 / n * 100
    wrong_pick_4_pc = wrong_pick_4 / n * 100
    right_ignore_pc = right_ignore / n * 100
    wrong_ignore_pc = wrong_ignore / n * 100
    give_up_pc = give_up / n_events * 100

    # ensure percentages add up

    five_percent_sum = right_pick_pc + wrong_pick_3_pc + wrong_pick_4_pc + right_ignore_pc + wrong_ignore_pc
    if abs(five_percent_sum - 100) > 1e-6:
        print(five_percent_sum)
        raise ValueError("percentages should add up to 100!")

    output_str = f"""
    Total number of events: {n_events}
    Minus events ignored: {give_up}, ({give_up_pc:.2f}%)

    N for percentages: {n}

    4th b-jet really exists:
        Correct 4th jet picked:         {right_pick_pc:.2f}%
        Incorrect 4th jet picked:       {wrong_pick_4_pc:.2f}%
        Event incorrectly ignored:      {wrong_ignore_pc:.2f}%

    No 4th b-jet really exists:
        Correctly ignored event:        {right_ignore_pc:.2f}%
        Incorrectly picked a 4th jet:   {wrong_pick_3_pc:.2f}%

    Or formatted in table form:
                    ____________________
                   |Truth-Matching      |
                   |____________________|
                   |4th exists  |No 4th |
     ______________|____________|_______|
    |4th |4th found|corr. {right_pick_pc:05.1f}%| {wrong_pick_3_pc:05.1f}%|
    |Jet |         |inco. {wrong_pick_4_pc:05.1f}%|       |
    |Reco|_________|____________|_______|
    |    |no 4th   |      {wrong_ignore_pc:05.1f}%| {right_ignore_pc:05.1f}%|
    |____|_________|____________|_______|
    
    (sum = {five_percent_sum:.2f})
    """
    print(output_str)
    #print(count_1, count_2, count_3, count_4)
    #print(count_1_total, count_2_total, count_3_total, count_4_total)

