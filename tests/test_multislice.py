"""Tests for the high-level torch_multislice.multislice() wrapper."""

import torch

from torch_multislice.multislice import multislice


def test_multislice_vacuum_leaves_plane_wave_unchanged():
    """Zero potential (vacuum) must leave a normally-incident plane wave unchanged.

    With no material, there is nothing to scatter the beam. The incident
    wave is a uniform (constant) plane wave, whose Fourier transform is
    concentrated entirely at the zero-frequency component - and the
    propagator is always 1 at zero frequency - so it must pass through
    every slice unchanged.
    """
    potential = torch.zeros(1, 5, 8, 8, dtype=torch.complex64)
    exit_wave = multislice(potential, pixel_size=1.0, energy=300.0)
    assert torch.allclose(exit_wave, torch.ones_like(exit_wave), atol=1e-5)


def test_multislice_conserves_energy_for_real_potential():
    """A real (non-absorbing) potential volume must conserve wave energy.

    Every slice composes a unitary transmission function (real V) with the
    always-unitary Fresnel propagator, so after propagating through all
    slices, the total wave energy must equal the incident plane wave's
    energy (H * W, since the incident wave has unit amplitude everywhere).
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 5, 8, 8, dtype=torch.complex64).real.to(torch.complex64)
    exit_wave = multislice(potential, pixel_size=1.0, energy=300.0)
    output_energy = exit_wave.abs().pow(2).sum()
    expected_energy = torch.tensor(8.0 * 8.0)
    assert torch.allclose(output_energy, expected_energy, rtol=1e-4)
