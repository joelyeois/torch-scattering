"""High-level multislice wrapper.

Wires up frequency grids, wavelength, and the per-slice loop on top of the
pure-math primitives in `_core`.
"""

import torch
from torch_ctf import calculate_relativistic_electron_wavelength
from torch_grid_utils import fftfreq_grid

from torch_multislice._core import (
    fresnel_propagator,
    interaction_parameter,
    multislice_step,
)


def multislice(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 2D exit wave from a 3D scattering potential.

    Parameters
    ----------
    potential : torch.Tensor
        Complex-valued 3D scattering potential, shape (..., Z, H, W), where
        Z is the number of slices along the beam direction.
    pixel_size : float | torch.Tensor
        Pixel size in Angstroms. The slice thickness `dz` is assumed to
        equal `pixel_size`.
    energy : float | torch.Tensor
        Electron beam energy in kiloelectronvolts (e.g. 300 for 300 kV).

    Returns
    -------
    torch.Tensor
        Complex-valued 2D exit wave, shape (..., H, W).

    Notes
    -----
    The incident wave entering the specimen is a uniform (unit-amplitude,
    zero-phase) plane wave. Each slice is applied via `multislice_step`,
    alternating transmission through the slice with Fresnel propagation to
    the next one.

    `torch_grid_utils.fftfreq_grid` only accepts a plain float for its
    `spacing` argument, so if `pixel_size` is a tensor, the frequency grid
    itself is built from a detached float copy; `pixel_size` still flows
    as a tensor into the propagator and transmission function, so gradients
    with respect to it are not lost, just not routed through the grid
    spacing.
    """
    wavelength_m = calculate_relativistic_electron_wavelength(energy * 1.0e3)
    wavelength = wavelength_m * 1.0e10  # meters -> Angstroms
    sigma = interaction_parameter(energy=energy)

    height, width = potential.shape[-2:]
    frequency_grid = fftfreq_grid(
        image_shape=(height, width),
        rfft=False,
        spacing=float(pixel_size),
        norm=True,
        device=potential.device,
    )
    propagator = fresnel_propagator(
        frequency_grid, wavelength=wavelength, dz=pixel_size
    )

    wave = torch.ones(
        (*potential.shape[:-3], height, width),
        dtype=potential.dtype,
        device=potential.device,
    )
    n_slices = potential.shape[-3]
    for i in range(n_slices):
        wave = multislice_step(
            wave, potential[..., i, :, :], propagator, sigma, pixel_size
        )
    return wave
