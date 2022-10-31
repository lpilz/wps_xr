from pathlib import Path

import numpy as np


def wps_static_filename_to_idx(_file):
    """Extracts tile indices from filename

    This function extracts the indices of a WPS static data tile from its filename

    Examples:
        >>> wps_static_filename_to_idx("00001-01200.01201-02400")
        (array([   1, 1200]), array([1201, 2400]))
        >>> wps_static_filename_to_idx("000001-001200.001201-002400")
        (array([   1, 1200]), array([1201, 2400]))
    """
    idxtmp = []
    _file = Path(_file).name
    for part in _file.split("."):
        idxtmp.append(np.array(list(map(int, part.split("-")))))
    return tuple(idxtmp)
