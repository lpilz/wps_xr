"""This module was adapted from
https://github.com/aurghs/xarray-backend-tutorial/blob/main/2.Backend_with_Lazy_Loading.ipynb
"""

import os

import dask
import numpy as np
import xarray as xr

from .backend_array import BinaryBackendArray
from .config import config
from .utils import wps_static_filename_to_idx


class BinaryBackend(xr.backends.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        dtype=np.int64
    ):
        size = np.dtype(dtype).itemsize
        idx = wps_static_filename_to_idx(filename_or_obj)
        shape = tuple([_i[1]-_i[0]+1 for _i in idx])

        backend_array = BinaryBackendArray(
            filename_or_obj=filename_or_obj,
            shape=shape,
            dtype=dtype,
            lock=dask.utils.SerializableLock(),
        )
        data = xr.core.indexing.LazilyIndexedArray(backend_array)

        var = xr.Variable(dims=config.get('general.DIMS_AVAIL')[:len(shape)][::-1], data=data)
        ds = xr.Dataset(data_vars = {
                    "foo": var
                },
                coords = {
                    dim:list(range(istart,istop+1)) for dim,(istart,istop) in zip(config.get('general.DIMS_AVAIL')[:len(shape)],idx)

                }
            )

        return ds
