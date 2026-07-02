"""Tests for pure-math multislice primitives in torch_multislice._core."""

import torch
from torch_grid_utils import fftfreq_grid

from torch_multislice._core import (
    fresnel_propagator,
    interaction_parameter,
    multislice_step,
    transmission_function,
)


def test_interaction_parameter_matches_kirkland_300kev():
    """sigma at 300 keV should match Kirkland's tabulated interaction constant.

    wavelength = 0.019687 Angstrom is the well-known relativistic electron
    wavelength at 300 keV (Kirkland, Advanced Computing in Electron
    Microscopy). The expected sigma = 6.526e-4 rad/(V*Angstrom) is the
    corresponding tabulated interaction parameter.
    """
    sigma = interaction_parameter(wavelength=0.019687, energy=300.0)
    assert abs(sigma - 6.526e-4) < 1e-7


def test_interaction_parameter_matches_kirkland_100kev():
    """sigma at 100 keV should match Kirkland's tabulated interaction constant."""
    sigma = interaction_parameter(wavelength=0.037013, energy=100.0)
    assert abs(sigma - 9.244e-4) < 1e-7


def test_fresnel_propagator_is_unitary():
    """Free-space propagation is lossless: |H(k)| == 1 for all frequencies.

    A pure phase transfer function conserves wave energy. A bug that turns
    the exponent real (e.g. a missing `1j`) would break this invariant.
    """
    frequency_grid = torch.tensor([[0.0, 0.1, 0.5], [1.0, 2.0, 10.0]])
    propagator = fresnel_propagator(frequency_grid, wavelength=0.019687, dz=2.0)
    assert torch.allclose(propagator.abs(), torch.ones_like(frequency_grid))


def test_fresnel_propagator_half_turn_case():
    """wavelength=0.5, dz=2, k=1 makes the phase pi*0.5*2*1 == pi, so H == -1."""
    frequency_grid = torch.tensor([[1.0]])
    propagator = fresnel_propagator(frequency_grid, wavelength=0.5, dz=2.0)
    assert torch.allclose(propagator, torch.tensor([[-1.0 + 0.0j]]), atol=1e-6)


def test_transmission_function_is_unitary_for_real_potential():
    """A real-valued (non-absorbing) potential must give a pure phase mask.

    |T| == 1 for any real potential slice, sigma, dz - a real material with
    no absorption cannot attenuate the wave, only phase-shift it.
    """
    potential_slice = torch.tensor([[0.0, 0.5, -2.0], [10.0, -100.0, 3.3]])
    transmission = transmission_function(potential_slice, sigma=2.5, dz=1.0)
    assert torch.allclose(transmission.abs(), torch.ones_like(potential_slice))


def test_transmission_function_attenuates_absorptive_potential():
    """A purely imaginary (absorptive) potential must attenuate the wave.

    potential = 1j (pure absorption term), sigma=1, dz=1 gives phase
    i*sigma*dz*V = i*1*1*1j = -1 (real and negative), so T = exp(-1) is a
    real number less than 1 - the wave loses amplitude, as expected for an
    absorbing specimen.
    """
    potential_slice = torch.tensor([[1.0j]])
    transmission = transmission_function(potential_slice, sigma=1.0, dz=1.0)
    expected = torch.exp(torch.tensor(-1.0)).to(torch.complex64).reshape(1, 1)
    assert torch.allclose(transmission, expected)


def test_multislice_step_conserves_energy_for_real_potential():
    """One slice of a real (non-absorbing) potential must conserve wave energy.

    transmission_function is unitary for real V, and fresnel_propagator is
    always unitary. Composing two unitary operators through an FFT/IFFT
    pair (also unitary up to a self-cancelling scale factor) must preserve
    the L2 norm of the wave - a weak-phase-object slice cannot create or
    destroy electrons.
    """
    torch.manual_seed(0)
    wave = torch.randn(8, 8, dtype=torch.complex64)
    potential_slice = torch.randn(8, 8)  # real -> non-absorbing
    wavelength = 0.019687
    dz = 2.0
    frequency_grid = fftfreq_grid(
        image_shape=(8, 8), rfft=False, spacing=1.0, norm=True
    )
    propagator = fresnel_propagator(frequency_grid, wavelength=wavelength, dz=dz)
    sigma = interaction_parameter(wavelength=wavelength, energy=300.0)

    new_wave = multislice_step(wave, potential_slice, propagator, sigma, dz)

    input_energy = wave.abs().pow(2).sum()
    output_energy = new_wave.abs().pow(2).sum()
    assert torch.allclose(output_energy, input_energy, rtol=1e-4)
