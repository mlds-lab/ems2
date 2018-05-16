
import numpy as np


def create_map(filename):
    """Create a dictionary map from the 32 character ids to 7 character ids.
    input. filename: name of the file that includes the mapping
    output. d: dictionary that include the mapping"""

    a = np.loadtxt(filename, delimiter="\t", dtype='U36')

    d = {}
    for i in range(a.shape[0]):
        d[a[i, 1]] = a[i, 0][6:]

    return d


def perform_map(a, filename):
    """Perform the mapping form 32 character userids to 7 character userids.
    input. a: an array that includes the 32 long character userids
    output. b: an array that include the corresponding 8 long character userids."""

    # filename = 'mperf_ids.txt'
    d = create_map(filename)

    b = np.zeros(a.shape, dtype='<U10')

    for i in range(a.size):
        b[i] = d[a[i]]

    return b
