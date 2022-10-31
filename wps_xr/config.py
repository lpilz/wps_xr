from pathlib import Path

import yaml
from donfig import Config

fn = Path(__file__).parents[0] / "config.yaml"

with open(fn) as f:
    defaults = yaml.safe_load(f)

config = Config("wps_xr", defaults=[defaults])
config.ensure_file(fn, comment=True)
