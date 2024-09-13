import numpy as np

from .features import get_features, aggregate_features
from .matrices import _flat


def _canberra_dist(v1, v2):
    """The canberra distance between two vectors. We need to carefully handle
    the case in which both v1 and v2 are zero in a certain dimension."""
    eps = 10 ** (-15)
    v1, v2 = [_flat(v) for v in [v1, v2]]
    d_can = 0
    for u, w in zip(v1, v2):
        if np.abs(u) < eps and np.abs(w) < eps:
            d_update = 1
        else:
            d_update = np.abs(u - w) / (np.abs(u) + np.abs(w))
        d_can += d_update
    return d_can


def netsimile(A1, A2):
    """NetSimile distance between two graphs.

    Parameters
    ----------
    A1, A2 : SciPy sparse array
        Adjacency matrices of the graphs in question.

    Returns
    -------
    d_can : Float
        The distance between the two graphs.

    Notes
    -----
    NetSimile works on graphs without node correspondence. Graphs to not need to
    be the same size.

    See Also
    --------

    References
    ----------
    """
    feat_A1, feat_A2 = [get_features(A) for A in [A1, A2]]
    agg_A1, agg_A2 = [aggregate_features(feat) for feat in [feat_A1, feat_A2]]
    # calculate Canberra distance between two aggregate vectors
    d_can = _canberra_dist(agg_A1, agg_A2)
    return d_can
