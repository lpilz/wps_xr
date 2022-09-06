"""This module was adapted from
https://github.com/aurghs/xarray-backend-tutorial/blob/main/2.Backend_with_Lazy_Loading.ipynb
"""

import numpy as np
import xarray as xr

from .config import config

# FIXME? This backend is dependent on config. It cannot be used independently...


class BinaryBackendArray(xr.backends.BackendArray):
    def __init__(
        self,
        filename_or_obj,
        shape,
        dtype,
        lock,
    ):
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock

    def __getitem__(self, key: tuple):
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        if type(key) is list:
            raise NotImplementedError(
                "Advanced indexing is not implemented,"
                "as IndexingSupport should only be BASIC."
            )

        key = (
            tuple([key] + [slice(None)] * (len(self.shape) - 1))
            if type(key) is int
            else key
        )
        size = np.dtype(self.dtype).itemsize
        flip_yax = config.get("index.row_order") == "top_bottom"

        if isinstance(key[0], slice):
            start = key[0].start if key[0].start is not None else 0
            stop = key[0].stop if key[0].stop is not None else self.shape[0]
            if flip_yax:
                start = self.shape[0] - stop
                stop = self.shape[0] - start
            offset = size * np.prod(self.shape[1:]) * start
            count = (stop - start) * np.prod(self.shape[1:])
            modshape = tuple([stop - start] + list(self.shape[1:]))
        else:
            offset = size * np.prod(self.shape[1:]) * key[0]
            count = 1 * np.prod(self.shape[1:])
            modshape = tuple([1] + list(self.shape[1:]))

        with self.lock, open(self.filename_or_obj) as f:
            arr = np.fromfile(f, self.dtype, offset=offset, count=count)

        arr = arr.reshape(modshape, order="C")
        if flip_yax:
            arr = np.flip(arr, 0)

        try:
            return arr[tuple([slice(None, stop - start, key[0].step)] + list(key[1:]))]
        except NameError:
            return arr[tuple([0] + list(key[1:]))]
