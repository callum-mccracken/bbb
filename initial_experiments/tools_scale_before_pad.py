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


def splitTVT(input_item, trainfrac = 0.8, testfrac = 0.2):
    """
        splits data into training, validation, and test subsets

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


def evaluate_model(truths, tags, selections, output="pretty", savename=None):
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
        string or None, what kind of output to produce, "pretty" or "ascii"
    savename:
        string, if output="pretty", save the table as table_{savename}.png

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

    five_percent_sum = sum([right_pick_pc, wrong_pick_3_pc, wrong_pick_4_pc,
                            right_ignore_pc, wrong_ignore_pc])
    if abs(five_percent_sum - 100) > 1e-6:
        print(five_percent_sum)
        raise ValueError("percentages should add up to 100!")

    # now normalize columns
    left_col_factor = (right_pick_pc + wrong_pick_4_pc + wrong_ignore_pc) / 100
    right_col_factor = (wrong_pick_3_pc + right_ignore_pc) / 100

    right_pick_pc /= left_col_factor
    wrong_pick_4_pc /= left_col_factor
    wrong_ignore_pc /= left_col_factor

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
    return right_pick_pc, wrong_pick_4_pc, wrong_ignore_pc, wrong_pick_3_pc, right_ignore_pc


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
    save:
        bool, whether or not to save the plot as table.png in current dir
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

    true4_found4_corr_pc = true4_found4_corr / n_col_1 * 100
    true4_found4_incorr_pc = true4_found4_incorr / n_col_1 * 100
    true4_found3_pc = true4_found3 / n_col_1 * 100

    true3_found4_pc = true3_found4 / n_col_2 * 100
    true3_found3_pc = true3_found3 / n_col_2 * 100

    # add a whole bunch of squares and text
    ax.text(0.5,1, "4th Jet\nReco", fontsize=18, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.add_patch(patches.Rectangle((0,0),1,2,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(1.5,1+1/3, "4th jet\nfound", fontsize=13, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(1.5,1, f"({n_row_1})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((1,0),1,1-1/3,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(3-0.05,1/3, f"{true4_found3_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(3-0.05,1/9, f"({true4_found3})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,0),2-0.1,1-1/3,linewidth=1,edgecolor='#262626',facecolor='#ffff66'))

    ax.text(4.45,1/3, f"{true3_found3_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(4.45,1/9, f"({true3_found3})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((4-0.1,0),1+0.1,1-1/3,linewidth=1,edgecolor='#262626',facecolor='#00ff66'))

    ax.text(1.5,0.4, "No 4th jet\nfound", fontsize=13, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(1.5,1/9, f"({n_row_2})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((1,1-1/3),1,1+1/3,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(2.5,1+2/3, "Correct\n4th jet", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,1-1/3),1,0.5*(1+1/3),linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(2.5,1, "Incorrect\n4th jet", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,1-1/3+0.5*(1+1/3)),1,0.5*(1+1/3),linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(3.45,1+2/3, f"{true4_found4_corr_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(3.45,1+2/3-2/9, f"({true4_found4_corr})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((3,1-1/3),1-0.1,0.5*(1+1/3),linewidth=1,edgecolor='k',facecolor='#ff6666'))

    ax.text(3.45,1, f"{true4_found4_incorr_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(3.45,1-2/9, f"({true4_found4_incorr})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((3,1-1/3+0.5*(1+1/3)),1-0.1,0.5*(1+1/3),linewidth=1,edgecolor='#262626',facecolor='#00ff66'))

    ax.text(4.45,1+1/3, f"{true3_found4_pc:.1f}%", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.text(4.45,1+1/3-2/9, f"({true3_found4})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((4-0.1,1-1/3),1+0.1,1+1/3,linewidth=1,edgecolor='#262626',facecolor='#ff6666'))

    ax.text(3,2.375, "4th tag exists", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(3,2.375-2/9, f"({n_col_1})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,2),2-0.1,0.75,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(4.45,2.375, "No 4th tag", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='heavy')
    ax.text(4.45,2.375-2/9, f"({n_col_2})", fontsize=10, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((4-0.1,2),1+0.1,0.75,linewidth=1,edgecolor='#262626',facecolor='w'))

    ax.text(3.5,3.1, "Truth-Matching", fontsize=18, verticalalignment='center', horizontalalignment='center', fontweight='heavy')

    ax.text(1,2.375, f"(# events={n_events})", fontsize=14, verticalalignment='center', horizontalalignment='center', fontweight='normal')
    ax.add_patch(patches.Rectangle((2,2+0.75),3,0.75,linewidth=1,edgecolor='#262626',facecolor='w'))

    # format and show/save
    plt.tight_layout()
    if savename:
        plt.savefig(f"table_{savename}.png", dpi=300)
    plt.show()


def boost_and_rotate(events):
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
        

fillval = 0

def pad(events, length=None, fill=None):
    """
    pads events out to a certain length (=number of jets).
    Optionally, provide a fill value (numbr)
    """
    global fillval
    if fill:
        fillval=fill
    print('padding with', fillval)
    events.truth = pad_sequences(events["truth"],padding='post')
    events.tag = pad_sequences(events["tag"], padding='post')
    padded_pt = pad_sequences(events["resolved_lv"].pt, padding='post', 
                              dtype='float32', value = fillval)
    padded_eta = pad_sequences(events["resolved_lv"].eta, padding='post', 
                               dtype='float32', value = fillval)
    padded_phi = pad_sequences(events["resolved_lv"].phi, padding='post', 
                               dtype='float32', value = fillval)
    padded_E = pad_sequences(events["resolved_lv"].E, padding='post', 
                             dtype='float32', value = fillval)
    
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


def scale_nn_input(events, chop=None, pad=10):
    global fillval

    print("scaling nn input")
    #print('fill value:', fillval)
    s_pt = np.empty_like(events.resolved_lv.pt)
    s_eta = np.empty_like(events.resolved_lv.eta)
    s_phi = np.empty_like(events.resolved_lv.phi)

    for i in range(pad):
        print("jet index index", i)
        pt = events.resolved_lv.pt[:,i]
        print(np.mean(pt), np.var(pt))
        pt_nonfill = pt[pt != fillval]
        pt_mean = np.mean(pt_nonfill)
        pt_var = np.var(pt_nonfill)
        print("pt mean", pt_mean, 'var', pt_var)
        eta = events.resolved_lv.eta[:,i]
        print(np.mean(eta), np.var(eta))
        eta_nonfill = eta[eta != fillval]
        eta_mean = np.mean(eta_nonfill)
        eta_var = np.var(eta_nonfill)
        print("eta mean", eta_mean, 'var', eta_var)
        phi = events.resolved_lv.phi[:,i]
        print(np.mean(phi), np.var(phi))
        phi_nonfill = phi[phi != fillval]
        phi_mean = np.mean(phi_nonfill)
        phi_var = np.var(phi_nonfill)
        print("phi mean", phi_mean, 'var', phi_var)
        s_pt[:,i] = (pt - pt_mean)/pt_var
        s_eta[:,i] = (eta - eta_mean)/eta_var
        s_phi[:,i] = (phi - phi_mean)/phi_var
        

    # if chop, "chop off" the first "chop" things from the jets
    if chop:
        s_pt = s_pt[:,chop:]
        s_eta = s_eta[:,chop:]
        s_phi = s_phi[:,chop:]
    
    # stack pt, eta, phi for input into model
    s_in = np.column_stack((s_pt, s_eta, s_phi))
    return s_in

if __name__ == "__main__":
    print("opening file")
    s_table = open_file("user.jagrundy.20736236._000001.MiniNTuple.root", sort_by="tag")

    # filter by realistic situation where we have 3 tags and 3 or 4 jets.
    # ignore the case where there may be >4 since those are pretty rare
    nb4 = (s_table.nbjets == 3) | (s_table.nbjets == 4) # 3 or 4 b-jets exist
    nt3 = s_table.nbtags==3  # 3 b tags
    nb4nt3 = nb4 & nt3
    events = s_table[nb4nt3]
    print(len(events))

    # and ensure that the 3 tags are actually correct
    # this results in very little event loss
    events = events[events.truth[:,0] == 1]
    events = events[events.truth[:,1] == 1]
    events = events[events.truth[:,2] == 1]
    print(len(events))

    cutoff = 10  # not many events have >10 jets
    events = pad(events, cutoff)

    import ptetaphi_nn
    nn = ptetaphi_nn.PtEtaPhiNN(events)
    # Feed forward NN