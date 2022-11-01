import math
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from .config import config
from .index import _write_index
from .wps import _add_latlon_coords, _generate_dtype


def _prepare_wps_directory(dirname_or_obj, force=False):
    """Prepares output directory by creating a new one or overriding an old one.

    Args:
        force (bool): Deletes the old directory should it already exist.
    """
    try:
        if force:
            logger.warning("Removing existing directory")
            shutil.rmtree(dirname_or_obj, ignore_errors=True)
        Path(dirname_or_obj).mkdir(parents=True, exist_ok=force)
    except FileExistsError:
        raise FileExistsError(
            "A directory with that name already exists. "
            "If you want to override the content, please use `force=True`."
        )


def _pad_data_if_needed(da, tile_size):
    """Pads the DataArray if necessary.

    This method checks if padding is necessary and performs it if true.

    Note:
        Padding is performed with `index.missing_value` from wps_xr.config object.

    Args:
        da (xarray.DataArray): Data to potentially pad.
        tile_size (tuple of ints): Size of output tiles. (x_size, y_size)

    Raises:
        KeyError: If padding is necessary but `index.missing_value` is not set.
    """
    shape = np.array([da.shape[da.dims.index(d)] for d in ["x", "y"]])
    tile_size = np.array(tile_size)
    padding_needed = (
        np.array([math.ceil(x) for x in shape / tile_size]) * tile_size - shape
    )
    if padding_needed.any():
        try:
            pad_value = config.get("index.missing_value")
        except KeyError:
            raise KeyError(
                "Couldn't pad data since index.missing_value is not set in config."
            )
        _attrs = da.attrs
        da = da.pad(
            pad_width={"x": (0, padding_needed[0]), "y": (0, padding_needed[1])},
            mode="constant",
            constant_values=pad_value,
        )
        da.attrs = _attrs

        for dim, shp, pad in zip(["x", "y"], shape, padding_needed):
            start_pad = da[dim][shp - 1].item() + 1
            newdim = np.append(np.zeros(shp), np.arange(start_pad, start_pad + pad))
            da[dim] = da[dim].fillna(0) + newdim
        da = _add_latlon_coords(da)
    return da


def _infer_var_name(ds, var):
    if var is None:
        if len(ds.data_vars) > 1:
            raise LookupError("Please provide variable name.")
        else:
            return list(ds.data_vars.keys())[0]
    return var


@xr.register_dataset_accessor("wps")
class WPSAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_disk(self, dirname_or_obj, var=None, tile_size=None, force=False):
        """Writes Dataset to disk.

        Args:
            dirname_or_obj (str, pathlib.Path): Name of output directory.
            var (str): Name of variable to write to disk. (default: the only `data_var`)
            tile_size (tuple): Size of individual tiles to write (x,y). If not given,
                `to_disk` tries to use "index.tile_[x,y]" from config or the dask chunks.
            force (bool): Whether to override existing data if some is present.
                (default: False)
        """
        dirname_or_obj = Path(dirname_or_obj)
        var = _infer_var_name(self._obj, var)

        if isinstance(var, Iterable) and not isinstance(var, (str, bytes)):
            raise Exception("Can only output a single variable.")

        if tile_size is None:
            try:
                tile_size = np.array(
                    (config.get("index.tile_x"), config.get("index.tile_y"))
                )
            except KeyError:
                try:
                    tile_size = np.array([self._obj[var].chunks[d] for d in ["x", "y"]])
                except TypeError:
                    raise KeyError(
                        "Couldn't set tile size, as index.tile_[x,y] not set in config."
                    )
        else:
            config.set({"index.tile_x": tile_size[0], "index.tile_y": tile_size[1]})

        if {**self._obj[var].attrs, **self._obj.attrs} != config.get("index"):
            logger.warning(
                "Variable attributes and config['index'] differ, using config['index']."
            )

        _prepare_wps_directory(dirname_or_obj, force)

        padded = _pad_data_if_needed(self._obj[var], tile_size)

        if padded.isnull().any():
            padded = padded.fillna(config.get("index.missing_value"))

        shape = np.array([padded.shape[padded.dims.index(d)] for d in ["x", "y"]])
        tile_nums = shape // tile_size

        def _fmt(x):
            return f"{x:0{config.get('index.filename_digits')}d}"

        dtype = _generate_dtype()

        for x in range(1, tile_nums[0] * tile_size[0], tile_size[0]):
            for y in range(1, tile_nums[1] * tile_size[1], tile_size[1]):
                xstart, xend = x, x + tile_size[0] - 1
                ystart, yend = y, y + tile_size[1] - 1
                filename = f"{_fmt(xstart)}-{_fmt(xend)}.{_fmt(ystart)}-{_fmt(yend)}"
                padded.sel(
                    {"x": slice(xstart, xend), "y": slice(ystart, yend)}
                ).values.T.tofile(dirname_or_obj / filename, format=dtype)

        _write_index(dirname_or_obj)
        return

    def plot(self, var=None):
        """Plot variable sensibly.

        Plots the given variable in the right orientation,
        using a categorical colorbar for categorical data.

        Args:
            var (str): Variable to plot
        """
        var = _infer_var_name(self._obj, var)

        if self._obj[var].attrs["type"] == "categorical":
            levels = list(
                range(
                    self._obj[var].attrs["category_min"] - 1,
                    self._obj[var].attrs["category_max"] + 1,
                )
            )
            add_args = {"levels": levels}
            if len(levels) < 30:
                add_args["cbar_kwargs"] = {"ticks": levels}
        else:
            add_args = {}
        return self._obj[var].plot(x="lon", y="lat", **add_args)
