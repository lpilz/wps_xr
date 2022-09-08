"""This module was adapted from
https://github.com/aurghs/xarray-backend-tutorial/blob/main/2.Backend_with_Lazy_Loading.ipynb
"""

import dask
import numpy as np
import xarray as xr

from .backend_array import BinaryBackendArray
from .config import config
from .utils import wps_static_filename_to_idx


def generate_shape_and_coordinate_indices(filename_or_obj):
    _idx = wps_static_filename_to_idx(filename_or_obj)
    _shape = [_i[1] - _i[0] + 1 for _i in _idx]

    try:
        tile_z = config.get("index.tile_z")
        if tile_z == 1:
            raise KeyError
        idx = tuple(list(_idx) + [np.array(range(1, tile_z + 1))])
        shape = tuple(_shape + [tile_z])
    except KeyError:
        try:
            tile_z_start, tile_z_end = config.get("index.tile_z_start"), config.get(
                "index.tile_z_end"
            )
            if tile_z_start == tile_z_end:
                raise KeyError
            idx = tuple(list(_idx) + [np.array(range(tile_z_start, tile_z_end + 1))])
            shape = tuple(_shape + [tile_z_end - tile_z_start + 1])
        except KeyError:
            idx = _idx
            shape = _shape

    return shape, idx


class BinaryBackend(xr.backends.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, drop_variables=None, dtype=np.int64):
        shape, idx = generate_shape_and_coordinate_indices(filename_or_obj)

        backend_array = BinaryBackendArray(
            filename_or_obj=filename_or_obj,
            shape=shape,
            dtype=dtype,
            lock=dask.utils.SerializableLock(),
        )
        data = xr.core.indexing.LazilyIndexedArray(backend_array)

        var = xr.Variable(
            dims=["y", "x"] + (["z"] if len(shape) > 2 else []), data=data
        )

        ds = xr.Dataset(
            data_vars={"foo": var},
            coords={
                dim: list(range(istart, istop + 1))
                for dim, (istart, istop) in zip(
                    ["x", "y"] + (["z"] if len(shape) > 2 else []), idx
                )
            },
        )

        if drop_variables is not None:
            ds = ds.drop_vars(drop_variables)

        return ds
