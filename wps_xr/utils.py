import os

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
    _file = os.path.basename(_file)
    for part in _file.split("."):
        idxtmp.append(np.array(list(map(int, part.split("-")))))
    return tuple(idxtmp)


def filename_to_idx(_file, tile_extent=None, ceil=False):
    if tile_extent is None:
        tile_extent = (1, 1)
    idxtmp = []
    _file = os.path.basename(_file)
    for part, te in zip(_file.split("."), tile_extent):
        idxtmp.append(int(part.split("-")[int(ceil)]) // te)
    return tuple(idxtmp)


def filelist_to_idxlist(filelist, tile_extent=None, ceil=False):
    idxlist = []
    for _file in filelist:
        print(_file)
        idxlist.append(filename_to_idx(_file, tile_extent, ceil))
    return idxlist


def extent_from_filelist(filelist, tile_extent=None):
    idxlist = filelist_to_idxlist(filelist, tile_extent, ceil=True)
    xlist, ylist = zip(*idxlist)
    return (np.max(xlist), np.max(ylist))


def numtiles_from_filelist(filelist, tile_extent=None):
    idxlist = filelist_to_idxlist(filelist, tile_extent)
    xlist, ylist = zip(*idxlist)
    return tuple(map(len, (np.unique(xlist), np.unique(ylist))))
