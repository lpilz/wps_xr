import os

import pytest
import yaml


@pytest.fixture(autouse=True)
def fix_config_state():
    # reset config to defaults
    from wps_xr.config import config

    config.clear()

    fn = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        os.path.join("wps_xr", "config.yaml"),
    )
    with open(fn) as f:
        defaults = yaml.safe_load(f)

    config.update(defaults)

    # execute test
    yield
