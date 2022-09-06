import copy
import itertools

import numpy as np
import pytest
from dask.utils import SerializableLock

from wps_xr.backend_array import BinaryBackendArray
from wps_xr.config import config

np_arr1 = np.array([[1, 2, 3], [4, 5, 6]]).astype("int8").T
arr1_pad = np.pad(np_arr1, ((1, 1), (1, 1)))
np_arr2 = (
    np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[10, 11, 12], [13, 14, 15]],
            [[20, 21, 22], [23, 24, 25]],
        ]
    )
    .astype("int8")
    .T
)
arr2_pad = np.pad(np_arr2, ((1, 1), (1, 1), (0, 0)))
np_arr3 = (
    np.array(
        [
            [
                [[1, 2, 3], [4, 5, 6]],
                [[10, 11, 12], [13, 14, 15]],
                [[20, 21, 22], [23, 24, 25]],
            ],
            [
                [[101, 102, 103], [104, 105, 106]],
                [[111, 112, 113], [114, 115, 116]],
                [[120, 121, 122], [123, 124, 125]],
            ],
            [
                [[31, 32, 33], [34, 35, 36]],
                [[41, 42, 43], [44, 45, 46]],
                [[50, 51, 52], [53, 54, 55]],
            ],
            [
                [[61, 62, 63], [64, 65, 66]],
                [[71, 72, 73], [74, 75, 76]],
                [[70, 71, 72], [73, 74, 75]],
            ],
        ]
    )
    .astype("int8")
    .T
)
arr3_pad = np.pad(np_arr3, ((1, 1), (1, 1), (0, 0), (0, 0)))


@pytest.fixture(scope="session")
def binfile(request, tmp_path_factory):
    fn = tmp_path_factory.mktemp("binaries") / "temp"
    request.param.tofile(fn)
    return fn


# Basic indexing is defined in:
# https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html
# it is supposed to support integers, slice, Ellipsis and np.newaxis objects
# however, the xarray adapter does not support Ellipsis and newaxis objects,
# so we don't test for them
@pytest.mark.parametrize(
    "binfile,arr,bdr",
    [
        [np_arr1, np_arr1, 0],
        [np_arr2, np_arr2, 0],
        [np_arr3, np_arr3, 0],
        [arr1_pad, arr1_pad, 1],
        [arr2_pad, arr2_pad, 1],
        [arr3_pad, arr3_pad, 1],
    ],
    indirect=["binfile"],
)
def test_raw_indexing_method_integers(binfile, arr, bdr):
    config.set({"index.row_order": "bottom_top"})
    config.set({"index.tile_bdr": bdr})
    # in action, shape is inferred from filename (=> is true data shape, non-padded)
    shape = tuple(
        [_shp - 2 * bdr if i < 2 else _shp for i, _shp in enumerate(arr.shape)]
    )
    test_arr = BinaryBackendArray(binfile, shape, arr.dtype, SerializableLock())

    # test element access
    # generate list of all possible indices
    idxlist = list(itertools.product(*map(list, map(range, shape))))
    for idx in idxlist:
        _test = test_arr._raw_indexing_method(tuple(idx))
        _arr = arr if bdr == 0 else arr[bdr:-bdr, bdr:-bdr, ...]
        _comp = _arr[tuple(idx)]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert _test == _comp, f"Access {idx} at {_arr} did not work"
        _idx = copy.copy(list(idx))
        _idx[-1] = -idx[-1]
        assert (
            test_arr._raw_indexing_method(tuple(_idx)) == _arr[tuple(_idx)]
        ), f"Access {idx} at {_arr} did not work"

    # test trailing slice filling
    idxlist = list(range(shape[0]))
    for idx in idxlist:
        _arr = arr if bdr == 0 else arr[bdr:-bdr, bdr:-bdr, ...]
        assert (test_arr._raw_indexing_method(idx) == _arr[idx, :]).all()
        assert (test_arr._raw_indexing_method(tuple([idx])) == _arr[idx, :]).all()


@pytest.mark.parametrize(
    "binfile,arr,bdr",
    [
        [np_arr1, np_arr1, 0],
        [np_arr2, np_arr2, 0],
        [np_arr3, np_arr3, 0],
        [arr1_pad, arr1_pad, 1],
        [arr2_pad, arr2_pad, 1],
        [arr3_pad, arr3_pad, 1],
    ],
    indirect=["binfile"],
)
def test_raw_indexing_method_slices(binfile, arr, bdr):
    config.set({"index.row_order": "bottom_top"})
    config.set({"index.tile_bdr": bdr})
    # in action, shape is inferred from filename (=> is true data shape, non-padded)
    shape = tuple(
        [_shp - 2 * bdr if i < 2 else _shp for i, _shp in enumerate(arr.shape)]
    )
    test_arr = BinaryBackendArray(binfile, shape, arr.dtype, SerializableLock())

    # generate list of all possible indices
    idxlist = list(itertools.product(*map(list, map(range, shape))))

    # test slice access
    for idx in idxlist:
        _idx = tuple(map(lambda x: slice(x, None, None), idx))
        _test = test_arr._raw_indexing_method(_idx)
        _arr = arr if bdr == 0 else arr[bdr:-bdr, bdr:-bdr, ...]
        _comp = _arr[_idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {_idx} at {arr} did not work"

    for idx in idxlist:
        _idx = tuple(map(lambda x: slice(None, x, None), idx))
        _test = test_arr._raw_indexing_method(_idx)
        _arr = arr if bdr == 0 else arr[bdr:-bdr, bdr:-bdr, ...]
        _comp = _arr[_idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {_idx} at {arr!r} did not work"

    dim_slices = []
    for dim in shape:
        slicelist = []
        for i, idx in enumerate(range(1, dim)):
            for _i in range(idx):
                slicelist.append(slice(_i, idx))
        dim_slices.append(slicelist)
    idxlist = list(itertools.product(*dim_slices))
    for idx in idxlist:
        _test = test_arr._raw_indexing_method(idx)
        _arr = arr if bdr == 0 else arr[bdr:-bdr, bdr:-bdr, ...]
        _comp = _arr[idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {idx} at {arr!r} did not work"

    # generate list of steps
    idxlist = list(
        itertools.product(*map(list, map(range, map(lambda x: x // 2, shape))))
    )
    for idx in idxlist:
        _idx = tuple(map(lambda x: slice(None, None, x or None), idx))
        _test = test_arr._raw_indexing_method(_idx)
        _arr = arr if bdr == 0 else arr[bdr:-bdr, bdr:-bdr, ...]
        _comp = _arr[_idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {_idx} at {arr!r} did not work"
