import numpy as np

from .features import get_features, aggregate_features
from .matrices import _flat


def _canberra_dist(v1, v2):
    """The Canberra distance between two vectors, handling NaN values appropriately."""
    v1, v2 = [_flat(v) for v in [v1, v2]]
    eps = 1e-15
    d_can = 0
    for u, w in zip(v1, v2):
        if np.isnan(u) or np.isnan(w):
            d_update = 0  # Ignore dimensions where either value is NaN
        else:
            denom = np.abs(u) + np.abs(w)
            if denom < eps:
                d_update = 0  # Both u and w are zero; contribution is zero
            else:
                d_update = np.abs(u - w) / denom
        d_can += d_update
    return d_can

def netsimile(G1, G2):
    """NetSimile distance between two graphs.

    Parameters
    ----------
    G1, G2 : networkx graph

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
    feat_G1, feat_G2 = [get_features(G) for G in [G1, G2]]
    agg_A1, agg_A2 = [aggregate_features(feat) for feat in [feat_G1, feat_G2]]
    # calculate Canberra distance between two aggregate vectors
    d_can = _canberra_dist(agg_A1, agg_A2)
    return d_can
