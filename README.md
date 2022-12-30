# WPS_XR
[![GitHub Workflow Status][github-ci-badge]][github-ci-link]
[![Code Coverage Status][codecov-badge]][codecov-link]
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d8e051334b1d4f13bc24d7852378866d)](https://www.codacy.com/gh/lpilz/wps_xr/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lpilz/wps_xr&amp;utm_campaign=Badge_Grade)
[![Maintainability](https://api.codeclimate.com/v1/badges/3b985f17e8d7491c94b7/maintainability)](https://codeclimate.com/github/lpilz/wps_xr/maintainability)

A collection of tools to integrate WPS geogrid binary datasets into the xarray ecosystem.
This package provides a function to open geogrid datasets, a configuration to change output format and a function to write data to disk.

## Usage
### Reading data
To open a WPS geogrid dataset located under `<path>`, use:
```
import wps_xr
ds = wps_xr.open_dataset(<path>)
```
This provides the data in an `xarray.Dataset` object.
It also populates the [`donfig`](https://github.com/pytroll/donfig) object `wps_xr.config`, which contains the configuration to be eventually written to the `index` file.

### Writing data to disk
The routine to write data to disk is provided via the `wps` accessor.
An example for `usgs` data might look like this:
```
ds.wps.to_disk(<output_path>, var="usgs", tile_size=(1200,1200), force=True)
```
This method will **not use the global and variable attributes**.
For output format configuration, please refer to the next section.

### Configuring the output
At the moment, the Dataset and DataArray attributes are only populated once and don't have an impact on the data being written to disk.
If you want to change the way the data is written, you have to use the `index` dict in the [`donfig`](https://github.com/pytroll/donfig) object populated by `open_dataset`.
```
from wps_xr import config
config.set({"index.missing_value": -9999})
```

### Plotting data
The `wps` accessor also provides a convenient plotting method:
```
ds.wps.plot(var="usgs")
```
It is mainly concerned with building the correct colorbar when encountering `categorical` data.

## Installation
Dependencies in this package are managed by `conda` and `poetry`.
To install this package follow the following steps:
```
conda env create -f ci/environment.yml && conda activate wps_xr
poetry build wheel
```
Then the `wps-xr*.whl` file is built in `./dist/` and can be installed using `pip install <file>`.

## Development
In order to develop this package, again use `conda` and `poetry`.
```
conda env create -f ci/environment.yml && conda activate wps_xr
poetry install
```
Then the `pytest`-powered testing can be run using `poetry run pytest .`

To use `pre-commit`, after installing the dependencies execute `poetry run pre-commit install`.

## TODOS:
- [ ] Add projections other than `regular_ll`
- [ ] Drop hard `dask` dependency
- [ ] Add tests for `index.missing_value`

[github-ci-badge]: https://img.shields.io/github/actions/workflow/status/lpilz/wps_xr/ci.yaml?branch=main
[github-ci-link]: https://github.com/lpilz/wps_xr/actions?query=workflow%3ACI
[codecov-badge]: https://img.shields.io/codecov/c/github/lpilz/wps_xr.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/lpilz/wps_xr
