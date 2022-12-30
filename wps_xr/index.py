import ast
import copy
from pathlib import Path

from .config import config


def __extract_key_val_from_line(line):
    key, val = line.split("=")
    return key.lower().strip(), val.strip("\"'\n ")


def __convert_val(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def __read_index(filename_or_obj):
    """Reads wps_xr.config.get("index") from given file, overriding defaults.

    Args:
        filename_or_obj (str,pathlib.Path): File to read
    """
    _dict = copy.deepcopy(config.get("index_defaults"))

    with open(filename_or_obj, "r") as f:
        for line in f:
            key, val = __extract_key_val_from_line(line)
            _dict.update({key: __convert_val(val) if key != "signed" else val})
    return _dict


def __check_switch_options(index):
    switch_options = {
        "type": ["continuous", "categorical"],
        "signed": ["yes", "no"],
        "row_order": ["bottom_top", "top_bottom"],
        "endian": ["big", "little"],
        "filename_digits": [5, 6],
    }
    for option in switch_options:
        assert index[option] in switch_options[option]


def __check_paired_options(index):
    # the combined parameters should only occur in pairs
    for params in config.get("general.COMBINED_PARAMS"):
        _intersect = set(params).intersection(set(index.keys()))
        if _intersect:
            assert len(_intersect) == 2


def __check_exclusive_options(index):
    # tile_z_{start,end} and tile_z are mutually exclusive
    if "tile_z_start" in index:
        assert "tile_z" not in index
    if "tile_z" in index:
        assert "tile_z_start" not in index
        assert "tile_z_end" not in index


def __check_index(index):
    """Checks index dict against the rules laid out in the WRF user's guide (v4.3).

    Args:
        index (dict): Index dictionary to check.
    """
    if index["projection"] not in [
        "regular_ll"
    ]:  # , 'lambert', 'polar', 'mercator',  'albers_nad83', 'polar_wgs84']
        raise NotImplementedError("Other projections are not implemented yet")

    __check_switch_options(index)

    __check_paired_options(index)

    __check_exclusive_options(index)


def _construct_index(pathname_or_obj):
    """Reads index from dir/file and constructs wps_xr.config.get("index") object.

    Args:
        pathname_or_obj (str,pathlib.Path): Directory/filename to read index file from.
    """
    pathname_or_obj = Path(pathname_or_obj)

    if pathname_or_obj.name != "index":
        pathname_or_obj = pathname_or_obj / "index"

    _dict = __read_index(pathname_or_obj)

    __check_index(_dict)

    config.set({"index": _dict})

    return _dict


def _write_index(pathname_or_obj):
    """Writes index file from wps_xr.config.get('index') to disk.

    Note:
        This function only writes non-default values.

    Args:
        pathname_or_obj (str,pathlib.Path): Directory/filename to write index file.
    """
    pathname_or_obj = Path(pathname_or_obj)

    if pathname_or_obj.name != "index":
        pathname_or_obj = pathname_or_obj / "index"

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
