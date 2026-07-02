"""Pure-math primitives for the multislice algorithm.

Functions in this module take physical quantities (wavelength, sigma,
frequency grid) as plain inputs supplied by the caller and have no
dependency on other teamtomo packages - they only require ``torch``
and ``scipy.constants``.
"""

import torch
from scipy import constants as C


def fresnel_propagator(
    frequency_grid: torch.Tensor, wavelength: float, dz: float
) -> torch.Tensor:
    """
    Compute the Fresnel free-space propagator for one multislice step.

    Parameters
    ----------
    frequency_grid : torch.Tensor
        Real-valued grid of spatial frequency magnitudes (1/Angstrom), e.g.
        from ``torch_grid_utils.fftfreq_grid(..., norm=True)``.
    wavelength : float
        Relativistic electron wavelength in Angstroms.
    dz : float
        Propagation distance (slice thickness) in Angstroms.

    Returns
    -------
    torch.Tensor
        Complex-valued Fresnel propagator with the same shape as
        `frequency_grid`.

    Notes
    -----
    ``H(k) = exp(i * pi * wavelength * dz * k^2)``. This is a pure phase
    transfer function, so ``|H(k)| == 1`` for all `k`.

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Springer US, Boston, MA, 2010.
    """
    return torch.exp(1j * C.pi * wavelength * dz * frequency_grid**2)


def transmission_function(
    potential_slice: torch.Tensor, sigma: float, dz: float
) -> torch.Tensor:
    """
    Compute the transmission function for one slice of scattering potential.

    Parameters
    ----------
    potential_slice : torch.Tensor
        Scattering potential for a single slice. Real-valued for a
        non-absorbing (weak phase object) specimen, or complex-valued with
        a positive imaginary part to model absorption.
    sigma : float
        Electron-specimen interaction parameter, e.g. from
        `interaction_parameter`, in rad/(V*Angstrom).
    dz : float
        Slice thickness in Angstroms.

    Returns
    -------
    torch.Tensor
        Complex-valued transmission function, same shape as
        `potential_slice`.

    Notes
    -----
    ``T = exp(i * sigma * dz * V)``. The phase shift imparted by a slice is
    the line integral of `sigma * V` along the beam direction through the
    slice; treating `V` as constant over the (thin) slice thickness reduces
    that integral to `sigma * V * dz`. For real `V` this makes `T` a pure
    phase mask (``|T| == 1``); a positive imaginary part of `V` attenuates
    the wave.
    """
    return torch.exp(1j * sigma * dz * potential_slice)


def multislice_step(
    wave: torch.Tensor,
    potential_slice: torch.Tensor,
    propagator: torch.Tensor,
    sigma: float,
    dz: float,
) -> torch.Tensor:
    """
    Advance the electron wave through one slice of scattering potential.

    Parameters
    ----------
    wave : torch.Tensor
        Complex-valued incident wave for this slice, shape (..., H, W).
    potential_slice : torch.Tensor
        Scattering potential for this slice, shape (..., H, W).
    propagator : torch.Tensor
        Fresnel propagator for this slice thickness, e.g. from
        `fresnel_propagator`, shape (H, W).
    sigma : float
        Electron-specimen interaction parameter, in rad/(V*Angstrom).
    dz : float
        Slice thickness in Angstroms.

    Returns
    -------
    torch.Tensor
        Complex-valued wave after transmission through and propagation
        past this slice, same shape as `wave`.

    Notes
    -----
    Each step alternates transmission through the slice (real space) with
    Fresnel propagation to the next slice (Fourier space):
    ``wave_out = IFFT(FFT(wave * T) * H)``
    """
    transmitted = wave * transmission_function(potential_slice, sigma, dz)
    # torch.fft.fft2/ifft2 are C-extension functions with no static return
    # type, so mypy sees them as returning Any; annotate explicitly.
    propagated_wave: torch.Tensor = torch.fft.ifft2(
        torch.fft.fft2(transmitted) * propagator
    )
    return propagated_wave


def interaction_parameter(wavelength: float, energy: float) -> float:
    """
    Calculate the electron-specimen interaction parameter.

    Computes the interaction constant sigma for electron scattering,
    following Kirkland Eq. (5.6).

    Parameters
    ----------
    wavelength : float
        Relativistic electron wavelength in Angstroms.
    energy : float
        Electron beam energy in kiloelectronvolts.

    Returns
    -------
    float
        Interaction parameter sigma in units of rad/(V*Angstrom).

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Eq. (5.6), Springer US, Boston, MA, 2010.
    """
    # Electron rest mass energy, m0*c^2, in electronvolts.
    rest_ev = C.electron_mass * C.speed_of_light**2 / C.elementary_charge
    ev = energy * 1.0e3  # [eV]
    sigma: float = (
        2.0 * C.pi / (wavelength * ev) * ((ev + rest_ev) / (ev + 2.0 * rest_ev))
    )
    return sigma
