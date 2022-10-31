from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def fix_config_state():
    # reset config to defaults
    from wps_xr.config import config

    config.clear()

    fn = Path(__file__).parents[1] / "wps_xr" / "config.yaml"
    with open(fn) as f:
        defaults = yaml.safe_load(f)

    config.update(defaults)

    # execute test
    yield
