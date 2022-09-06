import copy
import itertools

import numpy as np
import pytest
from dask.utils import SerializableLock

from wps_xr.backend_array import BinaryBackendArray

np_arr1 = np.array([[1, 2, 3], [4, 5, 6]]).astype("int8").T
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
    "binfile,arr",
    [tuple([np_arr1] * 2), tuple([np_arr2] * 2), tuple([np_arr3] * 2)],
    indirect=["binfile"],
)
def test_raw_indexing_method_integers(binfile, arr):
    test_arr = BinaryBackendArray(binfile, arr.shape, arr.dtype, SerializableLock())

    # test element access
    # generate list of all possible indices
    idxlist = list(itertools.product(*map(list, map(range, arr.shape))))
    for idx in idxlist:
        _test = test_arr._raw_indexing_method(tuple(idx))
        _comp = arr[tuple(idx)]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert _test == _comp, f"Access {idx} at {arr} did not work"
        _idx = copy.copy(list(idx))
        _idx[-1] = -idx[-1]
        assert (
            test_arr._raw_indexing_method(tuple(_idx)) == arr[tuple(_idx)]
        ), f"Access {idx} at {arr} did not work"

    # test trailing slice filling
    idxlist = list(range(arr.shape[0]))
    for idx in idxlist:
        assert (test_arr._raw_indexing_method(idx) == arr[idx, :]).all()
        assert (test_arr._raw_indexing_method(tuple([idx])) == arr[idx, :]).all()


@pytest.mark.parametrize(
    "binfile,arr",
    [tuple([np_arr1] * 2), tuple([np_arr2] * 2), tuple([np_arr3] * 2)],
    indirect=["binfile"],
)
def test_raw_indexing_method_slices(binfile, arr):
    test_arr = BinaryBackendArray(binfile, arr.shape, arr.dtype, SerializableLock())

    # generate list of all possible indices
    idxlist = list(itertools.product(*map(list, map(range, arr.shape))))

    # test slice access
    for idx in idxlist:
        _idx = tuple(map(lambda x: slice(x, None, None), idx))
        _test = test_arr._raw_indexing_method(_idx)
        _comp = arr[_idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {_idx} at {arr} did not work"

    for idx in idxlist:
        _idx = tuple(map(lambda x: slice(None, x, None), idx))
        _test = test_arr._raw_indexing_method(_idx)
        _comp = arr[_idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {_idx} at {arr!r} did not work"

    dim_slices = []
    for dim in arr.shape:
        slicelist = []
        for i, idx in enumerate(range(1, dim)):
            slicelist.append(slice(list(range(dim - 1))[i], idx))
        dim_slices.append(slicelist)
    idxlist = list(itertools.product(*dim_slices))
    for idx in idxlist:
        _test = test_arr._raw_indexing_method(idx)
        _comp = arr[idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {idx} at {arr!r} did not work"

    # generate list of steps
    idxlist = list(
        itertools.product(*map(list, map(range, map(lambda x: x // 2, arr.shape))))
    )
    for idx in idxlist:
        _idx = tuple(map(lambda x: slice(None, None, x or None), idx))
        _test = test_arr._raw_indexing_method(_idx)
        _comp = arr[_idx]
        assert getattr(_test, "shape", 0) == getattr(_comp, "shape", 0)
        assert (_test == _comp).all(), f"Access {_idx} at {arr!r} did not work"
