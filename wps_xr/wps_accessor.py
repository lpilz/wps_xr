import os
import pathlib
import shutil
from collections.abc import Iterable

import numpy as np
import xarray as xr
from loguru import logger

from .config import config
from .index import _write_index
from .wps import _generate_dtype


def _prepare_wps_directory(dirname_or_obj, force):
    try:
        if force:
            logger.warning("Removing existing directory")
            shutil.rmtree(dirname_or_obj, ignore_errors=True)
        pathlib.Path(dirname_or_obj).mkdir(parents=True, exist_ok=force)
    except FileExistsError:
        raise FileExistsError(
            "A directory with that name already exists. "
            "If you want to override the content, please use `force=True`."
        )


def _pad_data_if_needed(da, tile_size):
    shape = np.array([da.shape[da.dims.index(d)] for d in ["x", "y"]])
    tile_size = np.array(tile_size)
    padding_needed = shape - (shape // tile_size) * tile_size
    if padding_needed.any():
        try:
            pad_value = config.get("index.missing_value")
        except KeyError:
            raise KeyError(
                "Couldn't pad data since index.missing_value is not set in config."
            )
        da = da.pad(
            pad_width={"x": (0, padding_needed[0]), "y": (0, padding_needed[1])},
            mode="constant",
            constant_values=pad_value,
        )
    return da


@xr.register_dataset_accessor("wps")
class WPSAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_wps_data(self, dirname_or_obj, var=None, tile_size=None, force=False):
        """Writes Dataset to WPS static data

        Args:
            dirname_or_obj(str, pathlib.Path): dirname or Path object of output directory
            var(str): name of variable to write to disk
            tile_size(tuple): size of individual tiles to write (x,y)
        """
        if var is None:
            if len(self._obj.data_vars) > 1:
                raise Exception("Please provide variable name.")
            else:
                var = list(self._obj.data_vars.keys())[0]

        if isinstance(var, Iterable) and not isinstance(var, (str, bytes)):
            raise Exception("Can only output a single variable.")

        if tile_size is None:
            try:
                tile_size = np.array(
                    (config.get("index.tile_x"), config.get("index.tile_y"))
                )
            except KeyError:
                tile_size = np.array((self._obj[var].chunks[d] for d in ["x", "y"]))
                if tile_size is None:
                    raise KeyError(
                        "Couldn't set tile size, as index.tile_[x,y] not set in config."
                    )
        else:
            config.set({"index.tile_x": tile_size[0], "index.tile_y": tile_size[1]})

        _prepare_wps_directory(dirname_or_obj, force)

        self._obj[var] = _pad_data_if_needed(self._obj[var], tile_size)

        shape = np.array(
            [self._obj[var].shape[self._obj[var].dims.index(d)] for d in ["x", "y"]]
        )
        tile_nums = shape // tile_size

        def _fmt(x):
            return f"{x:0{config.get('index.filename_digits')}d}"

        for x in range(0, tile_nums[0] * tile_size[0], tile_size[0]):
            for y in range(0, tile_nums[1] * tile_size[1], tile_size[1]):
                xstart, xend = x, x + tile_size[0]
                ystart, yend = y, y + tile_size[1]
                filename = (
                    f"{_fmt(xstart+1)}-{_fmt(xend)}.{_fmt(ystart+1)}-{_fmt(yend)}"
                )
                self._obj[var].sel(
                    {"x": slice(xstart, xend), "y": slice(ystart, yend)}
                ).values.T.tofile(
                    os.path.join(dirname_or_obj, filename), format=_generate_dtype()
                )

        _write_index(dirname_or_obj)
        return

    def plot(self, var):
        """Plot variable sensibly"""
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
