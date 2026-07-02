"""Multislice electron scattering simulation in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from torch_multislice._core import (
    fresnel_propagator,
    interaction_parameter,
    multislice_step,
    transmission_function,
)
from torch_multislice.multislice import multislice

try:
    __version__ = version("torch-multislice")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "fresnel_propagator",
    "interaction_parameter",
    "multislice",
    "multislice_step",
    "transmission_function",
]
