"""A place to put some functions that a bunch of notebooks might need"""

import uproot as ur
import uproot_methods as urm
import awkward
import numpy as np
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing.sequence import pad_sequences 
from sklearn.preprocessing import StandardScaler


def open_file(filepath, sort_by=None, pt_cut=None, eta_cut=None):
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
        'resolvedJets_is_DL1r_FixedCutBEff_77', 'mcEventWeight'], namedecode='utf-8')
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

    # get important data
    pt = table['resolvedJets_pt']
    eta = table['resolvedJets_eta']
    phi = table['resolvedJets_phi']
    E = table['resolvedJets_E']
    truth = (table['resolvedJets_HadronConeExclTruthLabelID'] == 5).astype(np.int32)
    tag = table['resolvedJets_is_DL1r_FixedCutBEff_77']

    # apply cuts if needed
    if pt_cut != None:
        cut = pt > pt_cut
        pt = pt[cut]
        eta = eta[cut]
        phi = phi[cut]
        E = E[cut]
        truth = truth[cut]
        tag = tag[cut]
    if eta_cut != None:
        cut = abs(eta) < eta_cut
        pt = pt[cut]
        eta = eta[cut]
        phi = phi[cut]
        E = E[cut]
        truth = truth[cut]
        tag = tag[cut]

    # use pt, eta, phi, E to make LorentzVectors for each jet
    lv = urm.TLorentzVectorArray.from_ptetaphie(pt, eta, phi, E)
    # add LVs to table
    table['resolved_lv'] = lv

    # add a "truth" category to the table,
    # based on whether or not a jet is a b-jet or not
    table['truth'] = truth
    
    # and for convenience rename resolvedJets_is_DL1r_FixedCutBEff_77 to "tag"
    table['tag'] = tag

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
        s_table['mcEventWeight'] = table['mcEventWeight']
    else:
        print("not sorting data")
        s_table = table

    # a few easier names to use later

    # number of jets overall
    s_table['njets'] = awkward.AwkwardArray.count(pt)
    # number of b jets in each event
    s_table['nbjets'] = awkward.AwkwardArray.count_nonzero(truth)
    # number of b tags in each event
    s_table['nbtags'] = awkward.AwkwardArray.count_nonzero(tag)
 
    return s_table


def splitTVT(input_item, trainfrac = 0.8, testfrac = 0.2):
    """
    Splits data into training, validation, and test subsets.
    Ensure trainfrac + testfrac = 1.

    input_item:
        array of data to be split
    trainfrac:
        float < 1, fraction of data to be used in the training dataset, default 0.8
    testfrac:
        float < 1, fraction of data to be used in the test dataset, default 0.2
    """
    # by default no validation, be sure to have validation later!
    valfrac = 1.0 - trainfrac - testfrac
    
    train_split = ShuffleSplit(n_splits=1, test_size=testfrac + valfrac, random_state=0)
    # advance the generator once with the next function
    train_index, testval_index = next(train_split.split(input_item))  

    if valfrac > 0:
        testval_split = ShuffleSplit(
            n_splits=1, test_size=valfrac / (valfrac+testfrac), random_state=0)
        test_index, val_index = next(testval_split.split(testval_index)) 
    else:
        test_index = testval_index
        val_index = []

    return train_index, val_index, test_index


def evaluate_model(truths, tags, selections, weights=None, output="pretty", savename=None):
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
    output:
        string or None, what kind of output to produce, "pretty" or "ascii" or None
    savename:
        string, if output="pretty", save the table as table_{savename}.png

    Example for one event::

        truths = [[1, 1, 1, 1, 0]]
        tags = [[1, 1, 1, 0, 0]]
        selections = [[0, 0, 0, 1, 0]]
    """
    assert len(truths) == len(tags) == len(selections)

    if weights is None:
        weights = np.ones(len(selections))
    else:
        print('using weights')
        assert len(truths) == len(weights)

    n_correct = 0
    for i in range(len(truths)):
        if all(truths[i][3:] == selections[i][3:]):
            n_correct += 1
    #print(f"accuracy around {n_correct/len(truths)*100:.2f} percent")

    
    #print(selections)
    #print(truths)
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
            
        if n_selections > 1:
            raise ValueError("why are you selecting more than 1?")

        untagged = np.logical_xor(truth, tag).astype(int)

        n_untagged = np.count_nonzero(untagged)
        
        if np.count_nonzero(tag) > 3:
            print('too many tags ye goon')
            print(tag)
        
        if False: #i <10:
            print('i', i)
            print('truth    ', truth)
            print('tag      ', tag)
            print('selection', selection)
            print('selection_index', selection_index)
            print('n_selections', n_selections)
            print('n_untagged', n_untagged)
        
        if n_untagged == 0:  # should not have made any selection
            if n_selections != 0:  # made a selection, should not have
                wrong_pick_3 += weights[i]
            else:  # good call, no selection
                right_ignore += weights[i]
        elif n_untagged == 1:  # if we should have made 1 selection
            if n_selections == 0:  # no selection, bad
                wrong_ignore += weights[i]
            elif n_selections == 1:  # 1 selection, good
                # was it the right jet though?
                right_selection = bool(truth[selection_index] == 1)
                if right_selection:
                    right_pick += weights[i]
                else:
                    wrong_pick_4 += weights[i]
            else:
                raise ValueError("Why did you select more than 1 jet?")
        else:  # there were 2 or more untagged jets
            give_up += 1
    #print('\r')  # remove progress bar

    # ignore "give up" events in the stats at the end
    n = np.sum(weights)
    if np.all(weights == 1):
        n = n - give_up
    if n == 0:
        raise ValueError("something is causing the model evaluator to give up all the time!")

    # pc = percent
    right_pick_pc = right_pick / n * 100
    wrong_pick_3_pc = wrong_pick_3 / n * 100
    wrong_pick_4_pc = wrong_pick_4 / n * 100
    right_ignore_pc = right_ignore / n * 100
    wrong_ignore_pc = wrong_ignore / n * 100
    give_up_pc = give_up / n_events * 100

    print(f"overall accuracy: {(right_pick+right_ignore)/len(truths)*100:.2f} percent")
    
    print(f"ignoring {give_up_pc:.2f} percent ({give_up} events) of {n_events} events")
    # ensure percentages add up

    five_percent_sum = sum([right_pick_pc, wrong_pick_3_pc, wrong_pick_4_pc,
                            right_ignore_pc, wrong_ignore_pc])
    if np.all(weights == 1):
        if abs(five_percent_sum - 100) > 1e-6:
            print(five_percent_sum)
            raise ValueError("percentages should add up to 100!")

    # now normalize columns
    left_col_factor = (right_pick_pc + wrong_pick_4_pc + wrong_ignore_pc) / 100
    right_col_factor = (wrong_pick_3_pc + right_ignore_pc) / 100

    if left_col_factor != 0:
        right_pick_pc /= left_col_factor
        wrong_pick_4_pc /= left_col_factor
        wrong_ignore_pc /= left_col_factor

    if right_col_factor != 0:
        wrong_pick_3_pc /= right_col_factor
        right_ignore_pc /= right_col_factor


    output_str = f"""
    Total number of events: {n_events}
    Minus events ignored: {give_up}, ({give_up_pc:.2f}%)

    4th b-jet really exists:
        Correct 4th jet picked:         {right_pick_pc:.2f}%, {right_pick}
        Incorrect 4th jet picked:       {wrong_pick_4_pc:.2f}%, {wrong_pick_4}
        Event incorrectly ignored:      {wrong_ignore_pc:.2f}%, {wrong_ignore}

    No 4th b-jet really exists:
        Correctly ignored event:        {right_ignore_pc:.2f}%, {right_ignore}
        Incorrectly picked a 4th jet:   {wrong_pick_3_pc:.2f}%, {wrong_pick_3}

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

    (columns add to 100% each)
    """
    if output is None:
        pass
    elif output == "ascii":
        print(output_str)
    elif output == "pretty":
        table_plot(right_pick, wrong_pick_4, wrong_ignore,
                   wrong_pick_3, right_ignore, savename=savename)
    return selections
    #return right_pick_pc, wrong_pick_4_pc, wrong_ignore_pc, wrong_pick_3_pc, right_ignore_pc


def table_plot(true4_found4_corr, true4_found4_incorr, true4_found3,
               true3_found4, true3_found3, savename=None):
    """
    Makes a pretty plot of a model evaluation table,
    as opposed to the ascii version created by evaluate_model.

    true4_found4_corr:
        int, # of events where 4th jet was found correctly
    true4_found4_incorr:
        int, # of events where 4th jet was found incorrectly
    true4_found3:
        int, # of events where a 4th jets should have been picked but wasn't
    true3_found4:
        int, # of events where a 4th jets shouldn't have been picked but was
    true3_found3:
        int, # of events where no 4th jet was picked, and that was correct
    savename:
        str or None, if str, save plot as table_[savename].png in current dir
    """
    
    # Prepare plot on which to place table
    _, ax = plt.subplots()
    plt.xlim(-0.1,5.1)
    plt.ylim(-0.1,3.7)
    ax.axis('off')

    n_events = sum([true4_found4_corr, true4_found4_incorr, true4_found3,
                   true3_found4, true3_found3])

    n_col_1 = sum([true4_found4_corr, true4_found4_incorr, true4_found3])
    n_col_2 = sum([true3_found4, true3_found3])

    n_row_1 = sum([true4_found4_corr, true4_found4_incorr, true3_found4])
    n_row_2 = sum([true4_found3, true3_found3])
    
    if n_col_1 != 0:
        true4_found4_corr_pc = true4_found4_corr / n_col_1 * 100
        true4_found4_incorr_pc = true4_found4_incorr / n_col_1 * 100
        true4_found3_pc = true4_found3 / n_col_1 * 100
    else:
        true4_found4_corr_pc = 0
        true4_found4_incorr_pc = 0
        true4_found3_pc = 0
    if n_col_2 != 0:
        true3_found4_pc = true3_found4 / n_col_2 * 100
        true3_found3_pc = true3_found3 / n_col_2 * 100
    else:
        true3_found4_pc = 0
        true3_found3_pc = 0

    # add a whole bunch of squares and text
    ax.text(0.5,1, "4th Jet\nReco", fontsize=18, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.add_patch(patches.Rectangle((0,0),1,2,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(1.5,1+1/3, "4th jet\nfound", fontsize=13, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(1.5,1, f"({n_row_1:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((1,0),1,1-1/3,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(3-0.05,1/3, f"{true4_found3_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(3-0.05,1/9, f"({true4_found3:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,0),2-0.1,1-1/3,linewidth=1,edgecolor='#262626',facecolor='#ffff66'))

    ax.text(4.45,1/3, f"{true3_found3_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(4.45,1/9, f"({true3_found3:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((4-0.1,0),1+0.1,1-1/3,linewidth=1,edgecolor='#262626',facecolor='#00ff66'))

    ax.text(1.5,0.4, "No 4th jet\nfound", fontsize=13, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(1.5,1/9, f"({n_row_2:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((1,1-1/3),1,1+1/3,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(2.5,1+2/3, "Correct\n4th jet", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,1-1/3),1,0.5*(1+1/3),linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(2.5,1, "Incorrect\n4th jet", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,1-1/3+0.5*(1+1/3)),1,0.5*(1+1/3),linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(3.45,1+2/3, f"{true4_found4_corr_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(3.45,1+2/3-2/9, f"({true4_found4_corr:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((3,1-1/3),1-0.1,0.5*(1+1/3),linewidth=1,edgecolor='k',facecolor='#ff6666'))

    ax.text(3.45,1, f"{true4_found4_incorr_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(3.45,1-2/9, f"({true4_found4_incorr:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((3,1-1/3+0.5*(1+1/3)),1-0.1,0.5*(1+1/3),linewidth=1,edgecolor='#262626',facecolor='#00ff66'))

    ax.text(4.45,1+1/3, f"{true3_found4_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(4.45,1+1/3-2/9, f"({true3_found4:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((4-0.1,1-1/3),1+0.1,1+1/3,linewidth=1,edgecolor='#262626',facecolor='#ff6666'))

    ax.text(3,2.375, "4th tag exists", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(3,2.375-2/9, f"({n_col_1:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,2),2-0.1,0.75,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(4.45,2.375, "No 4th tag", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(4.45,2.375-2/9, f"({n_col_2:.0f})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((4-0.1,2),1+0.1,0.75,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(3.5,3.1, "Truth-Matching", fontsize=18, verticalalignment='center', horizontalalignment='center', fontweight='heavy')

    ax.text(1,2.375, f"(# events={n_events:.0f})", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,2+0.75),3,0.75,linewidth=1,edgecolor='#262626',facecolor='w'))

    # format and show/save
    plt.tight_layout()
    if savename:
        plt.savefig(f"table_{savename}.png", dpi=300)
    plt.show()


def boost_and_rotate(events):
    """
    Given a list of events, boosts into the center of mass frame,
    then rotates jets phi=0 for the vector sum of the first 3 jets in each event.
    """
    # we assume events is rectangular (i.e. all events have same # of jets
    njets = len(events.resolved_lv.pt[0])
    # get vectors
    vectors = events.resolved_lv

    # get sum vectors
    x_sum = np.repeat(np.sum(vectors.x, axis=1).reshape(-1, 1), njets, axis=1)
    y_sum = np.repeat(np.sum(vectors.y, axis=1).reshape(-1, 1), njets, axis=1)
    z_sum = np.repeat(np.sum(vectors.z, axis=1).reshape(-1, 1), njets, axis=1)
    t_sum = np.repeat(np.sum(vectors.t, axis=1).reshape(-1, 1), njets, axis=1)
    v_sum = urm.TLorentzVectorArray(x_sum, y_sum, z_sum, t_sum)

    # b for boosted
    vectors_b = vectors.boost(-v_sum.boostp3)
    v_sum_b = v_sum.boost(-v_sum.boostp3)
    # for filler data where eta = 0, we'll have NaN for eta, replace that
    eta = np.nan_to_num(vectors_b.eta, nan=0.0)
    vectors_b = urm.TLorentzVectorArray.from_ptetaphie(vectors_b.pt, eta, vectors_b.phi, vectors_b.E)

    # now rotate the system based on the first 3 jets
    # get sum of the first 3, similarly to before
    x_sum3 = np.repeat(np.sum(vectors_b.x[:,:3], axis=1).reshape(-1, 1), njets, axis=1)
    y_sum3 = np.repeat(np.sum(vectors_b.y[:,:3], axis=1).reshape(-1, 1), njets, axis=1)
    z_sum3 = np.repeat(np.sum(vectors_b.z[:,:3], axis=1).reshape(-1, 1), njets, axis=1)
    t_sum3 = np.repeat(np.sum(vectors_b.t[:,:3], axis=1).reshape(-1, 1), njets, axis=1)
    v_sum3 = urm.TLorentzVectorArray(x_sum3, y_sum3, z_sum3, t_sum3)

    # rotate about z so that phi=0 for v_sum3
    vectors_r = vectors_b.rotatez(-v_sum3.phi)
    v_sum3 = v_sum3.rotatez(-v_sum3.phi)

    # and again replace filler etas with 0
    print("Don't worry if you see a warning about dividing by zero, fixing that!")
    eta = np.nan_to_num(vectors_r.eta, nan=0.0)
    vectors_final = urm.TLorentzVectorArray.from_ptetaphie(vectors_r.pt, eta, vectors_r.phi, vectors_r.E)

    events.resolved_lv = vectors_final
    return events    
        
    
def pad(events, length=None):
    """
    Given some events of variable length,
    pads each event so they all have a certain length.

    By default, the maximum length of any event is used as the
    padding length, but that can be edited with ``length``.

    events:
        uproot table of event data
    length:
        int, pad events to n_jets = length, default = max n_jets
    """
    events.truth = pad_sequences(events["truth"],padding='post')
    events.tag = pad_sequences(events["tag"], padding='post')
    padded_pt = pad_sequences(events["resolved_lv"].pt, padding='post', 
                              dtype='float32', value = 0)
    padded_eta = pad_sequences(events["resolved_lv"].eta, padding='post', 
                               dtype='float32', value = 0)
    padded_phi = pad_sequences(events["resolved_lv"].phi, padding='post', 
                               dtype='float32', value = 0)
    padded_E = pad_sequences(events["resolved_lv"].E, padding='post', 
                             dtype='float32', value = 0)
    
    if length is not None:
        events.truth = events.truth[:,:length]
        events.tag = events.tag[:,:length]
        padded_pt = padded_pt[:,:length]
        padded_eta = padded_eta[:,:length]
        padded_phi = padded_phi[:,:length]
        padded_E = padded_E[:,:length]
    
    events.resolved_lv = urm.TLorentzVectorArray.from_ptetaphie(
        padded_pt, padded_eta, padded_phi, padded_E)
    
    return events


def scale_nn_input(events, chop=None, save_csv=None):
    """
    Given events, return scaled input for a neural network.

    events:
        uproot table of event data
    chop:
        int, remove the first ``chop'' jets from each events,
        default None --> no jets chopped
    save_csv:
        string, save scaling parameters to [save_csv].csv,
        default None --> no csv saved
    """
    # scale data to be keras-friendly
    scaler_pt = StandardScaler()
    scaler_eta = StandardScaler()
    scaler_phi = StandardScaler()

    s_pt = scaler_pt.fit_transform(events.resolved_lv.pt)
    s_eta = scaler_eta.fit_transform(events.resolved_lv.eta)
    s_phi = scaler_phi.fit_transform(events.resolved_lv.phi)

    if save_csv:
        fname = save_csv+".csv"
        with open(fname, 'w') as f:
            f.write("pt_mean,pt_var,eta_mean,eta_var,phi_mean,phi_var\n")
            for i in range(len(events.resolved_lv.pt[0])):
                string = ",".join(map(str, [
                    scaler_pt.mean_[i],
                    scaler_pt.var_[i],
                    scaler_eta.mean_[i],
                    scaler_eta.var_[i],
                    scaler_phi.mean_[i],
                    scaler_phi.var_[i],
                    ])) + "\n" 
                f.write(string)
    # if chop, "chop off" the first "chop" things from the jets
    if chop:
        s_pt = s_pt[:,chop:]
        s_eta = s_eta[:,chop:]
        s_phi = s_phi[:,chop:]
    
    # stack pt, eta, phi for input into model
    s_in = np.column_stack((s_pt, s_eta, s_phi))
    if len(events.resolved_lv.pt) == 1:
        unscaled_in = np.column_stack((events.resolved_lv.pt, events.resolved_lv.eta, events.resolved_lv.phi))
        print('unscaled', unscaled_in[0])
    return s_in

if __name__ == "__main__":
    table_plot(20,20,20,20,20,20)