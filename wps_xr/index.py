import ast
import copy
import os

from .config import config


def __read_index(filename_or_obj):
    """Reads wps_xr.config.get("index") from given file, overriding defaults.

    Args:
        filename_or_obj (str,pathlib.Path): File to read
    """
    _dict = copy.deepcopy(config.get("index_defaults"))

    with open(filename_or_obj, "r") as f:
        for line in f:
            key, val = line.split("=")
            key = key.lower().strip()
            val = val.strip("\"'\n ")
            try:
                if key != "signed":
                    val = ast.literal_eval(val)
                else:
                    pass
            except (ValueError, SyntaxError):
                pass
            _dict.update({key: val})
    return _dict


def __check_index(index):
    """Checks index dict against the rules laid out in the WRF user's guide (v4.3).

    Args:
        index (dict): Index dictionary to check.
    """
    if index["projection"] not in [
        "regular_ll"
    ]:  # , 'lambert', 'polar', 'mercator',  'albers_nad83', 'polar_wgs84']
        raise NotImplementedError("Other projections are not implemented yet")
    assert index["type"] in ["continuous", "categorical"]
    assert index["signed"] in ["yes", "no"]
    assert index["row_order"] in ["bottom_top", "top_bottom"]
    assert index["endian"] in ["big", "little"]
    assert index["filename_digits"] in [5, 6]

    # the combined parameters should only occur in pairs
    for params in config.get("general.COMBINED_PARAMS"):
        _intersect = set(params).intersection(set(index.keys()))
        print(params)
        if _intersect:
            assert len(_intersect) == 2

    # tile_z_{start,end} and tile_z are mutually exclusive
    if "tile_z_start" in index:
        assert "tile_z" not in index
    if "tile_z" in index:
        assert "tile_z_start" not in index
        assert "tile_z_end" not in index


def _construct_index(pathname_or_obj):
    """Reads index from dir/file and constructs wps_xr.config.get("index") object.

    Args:
        pathname_or_obj (str,pathlib.Path): Directory/filename to read index file from.
    """
    if os.path.basename(pathname_or_obj) != "index":
        pathname_or_obj = os.path.join(pathname_or_obj, "index")

    _dict = __read_index(pathname_or_obj)

    __check_index(_dict)

    config.set({"index": _dict})

    return _dict


def _write_index(pathname_or_obj):
    """Writes index file from wps_xr.config.get("index") to disk.

    Note:
        This function only writes non-default values.

    Args:
        pathname_or_obj (str,pathlib.Path): Directory/filename to write index file.
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
