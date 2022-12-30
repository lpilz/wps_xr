from .config import config  # noqa: F401 silence pyflakes
from .wps import open_dataset
from .wps_accessor import WPSAccessor  # noqa: F401 silence pyflakes

__all__ = ["open_dataset"]
