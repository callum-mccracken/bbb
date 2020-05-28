import numpy as np
import matplotlib.pyplot as plt
def select(events):
    """
    We assume events have at least 3 jets, and all events have the same length.
    We also assume the first 3 jets are always correctly tagged!
    """
    njets = events.njets[0]
    # separate the 3 first jets from the rest since they're already tagged correctly
    # (given our filtering procedure above)
    px_3, px_rest = events.resolved_lv.p3.x[:, :3], events.resolved_lv.p3.x[:, 3:]
    py_3, py_rest = events.resolved_lv.p3.y[:, :3], events.resolved_lv.p3.y[:, 3:]
    # find the best jet from _rest that fits with _3, in terms of pt sum = 0
    px_3_sums = np.repeat(np.sum(px_3, axis=1).reshape(-1, 1), njets-3, axis=1)
    py_3_sums = np.repeat(np.sum(py_3, axis=1).reshape(-1, 1), njets-3, axis=1)

    px_sums = np.abs(px_rest + px_3_sums)
    py_sums = np.abs(py_rest + py_3_sums)

    magnitudes = np.sqrt(px_sums**2 + py_sums**2)
    lowest_vals = np.min(magnitudes, axis=1)
    lowest_indices = np.argmin(magnitudes, axis=1)

    # lower values = more certainty
    # let's set an arbitrary threshold,
    # and say if lowest_val > thresh, pick no jet
    thresh1_dict = {
        4: 1,
        5: 100,
        6: 100,
        7: 100,
        8: 100
    }
    thresh1 = thresh1_dict[njets]
    thresh2_dict = {
        4: 50,
        5: 200,
        6: 200,
        7: 200,
        8: 200
    }
    thresh2 = thresh2_dict[njets]
    
    lowest_indices[lowest_vals>thresh1] = njets-3

    # also if the event was fine before adding a 4th jet, pick no jet
    # this ends up having a small effect
    px_sums_no_4th = np.abs(px_3_sums[:,0])
    py_sums_no_4th = np.abs(py_3_sums[:,0])
    magnitudes_no_4th = np.sqrt(px_sums_no_4th**2 + py_sums_no_4th**2)
    print(magnitudes_no_4th)
    lowest_indices[magnitudes_no_4th<thresh2] = njets-3
    print(np.count_nonzero(lowest_indices == njets-3))

    # put this in a better format
    selection_index = lowest_indices + 3
    selections = np.zeros((len(events.truth), njets+1), dtype=int)
    for i, s in enumerate(selection_index):
        selections[i][s] = 1
    # chop off last index so selection = [0,...,0] for no selection
    selections = selections[:, :-1]
    plt.cla(); plt.clf()
    plt.hist(magnitudes)
    plt.show()
    return selections