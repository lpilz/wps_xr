import ast
import copy
import os

from .config import config


def __read_index(filename_or_obj):
    """Reads index from given file, overriding defaults"""
    _dict = copy.deepcopy(config.get("index_defaults"))

    with open(os.path.join(filename_or_obj), "r") as f:
        for line in f:
            key, val = line.split("=")
            key = key.lower()
            val = val.strip("\"'\n")
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
            _dict.update({key: val})
    if "signed" in _dict:
        _dict["signed"] = "yes" if _dict["signed"] else "no"
    return _dict


def __check_index(index):
    """Checks index against the rules laid out in the WRF user's guide (v4.3)"""
    if index["projection"] not in [
        "regular_ll"
    ]:  # , 'lambert', 'polar', 'mercator',  'albers_nad83', 'polar_wgs84']
        raise NotImplementedError("Other projections are not implemented yet")
    assert index["type"] in ["continuous", "categorical"]
    assert index["signed"] in ["yes", "no"], index
    assert index["row_order"] in ["bottom_top", "top_bottom"]
    assert index["endian"] in ["big", "little"]
    assert index["filename_digits"] in [5, 6]

    # the combined parameters should only occur in pairs
    for params in config.get("general.COMBINED_PARAMS"):
        _intersect = set(params).intersection(set(index.keys()))
        if _intersect:
            assert len(_intersect) == 2

    # tile_z_{start,end} and tile_z are mutually exclusive
    if hasattr(index, "tile_z_start"):
        assert not hasattr(index, "tile_z")
    if hasattr(index, "tile_z"):
        assert not hasattr(index, "tile_z_start") and not hasattr(index, "tile_z_end")


def _construct_index(pathname_or_obj):
    """Reads index from dir/file and constructs config index object"""
    if os.path.basename(pathname_or_obj) != "index":
        pathname_or_obj = os.path.join(pathname_or_obj, "index")

    _dict = __read_index(pathname_or_obj)

    __check_index(_dict)

    config.set({"index": _dict})

    return _dict


def _write_index(pathname_or_obj):
    """Writes index file to disk

    This function only writes non-default values.
    """
    if os.path.basename(pathname_or_obj) != "index":
        pathname_or_obj = os.path.join(pathname_or_obj, "index")

    with open(pathname_or_obj, "w") as f:
        for key, val in config.get("index").items():
            if (
                key in config.get("index_defaults")
                and val != config.get("index_defaults")[key]
            ):
                continue
            if key in ["units", "description", "mminlu"]:
                val = f'"{val}"'
            f.write(f"{key.upper()} = {val}\n")
