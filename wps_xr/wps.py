import os
import xarray as xr

from .backend import BinaryBackend
from .config import config
from .index import _construct_index

COMBINED_PARAMS = [('tile_z_start', 'tile_z_end'), ('category_min', 'category_max')]

def _add_latlon_coords(ds):
    assert config.get('index.projection') == 'regular_ll'
    lat = (ds.y.values - config.get('index.known_y'))*config.get('index.dy') + config.get('index.known_lat')
    lon = (ds.x.values - config.get('index.known_x'))*config.get('index.dx') + config.get('index.known_lon')
    ds['lat'] = ('y', lat)
    ds['lon'] = ('x', lon)
    ds = ds.assign_coords({'lat': ds.lat, 'lon': ds.lon})
    return ds

def _generate_dtype():
    int_str = "u" if config.get("index.signed") == "no" else "i"
    endian_str = "" if config.get("index.wordsize") == 1 else "<" if config.get("index.endian") == 'little' else ">"
    return f"{endian_str}{int_str}{config.get('index.wordsize')}"

def open_dataset(pathname_or_obj):
    if not os.path.isdir(pathname_or_obj) and not os.path.exists(os.path.join(pathname_or_obj,"index")):
        raise Exception("Please provide the directory of a proper WPS binary dataset.")

    index = _construct_index(pathname_or_obj)
    config.update(dict(index=index), priority='new')

    # construct field variable
    index_str = "?" * config.get("index.filename_digits")
    ds = xr.open_mfdataset(
            os.path.join(pathname_or_obj, f"{index_str}-{index_str}.{index_str}-{index_str}"),
            engine=BinaryBackend,
            dtype=_generate_dtype(),
            combine="by_coords"
        )
    if 'missing_value' in config.get('index'):
        ds.foo = ds.where(ds.foo != config.get('index.missing_value'))
    ds['foo'] = ds.foo * config.get('index.scale_factor')
    ds.foo.attrs = {key:config.get('index')[key] for key in config.get('general.VARIABLE_ATTRS') if key in config.get('index').keys()}
    ds = ds.rename({"foo": os.path.basename(pathname_or_obj)})

    # add global attributes
    ds.attrs = {"directory": pathname_or_obj}
    ds.attrs.update({key:config.get('index')[key] for key in config.get('index').keys() if key not in config.get('general.VARIABLE_ATTRS')})

    ds = _add_latlon_coords(ds)
    return ds
