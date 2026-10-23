#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
"""Regression tests for the thermal continuum-to-bridging boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from Aerothermo import aerothermo
from Configuration.configuration import Aerothermo as AerothermoOptions
from Freestream import mix_properties


_LREF = 1.0
_VELOCITY = 5000.0
_WALL_TEMPERATURE = 300.0
_EPSILON = 1e-2
_KN_RTOL = 1e-8
_ST_RTOL = 7e-3
_END_TO_END_EPSILONS = (1e-3, 1e-4)
_KN_IDENTITY_RTOL = 1e-12
_KN_TARGET_RTOL = 1e-10
_CONTINUITY_GAP_RTOL = 2e-4
_FLOW_DIRECTION = np.array([0.0, 0.0, -1.0])
_VISIBLE_FACET = np.array([0])


def _make_options():
    """Return the smallest options object needed by the thermal path."""
    aerothermo_options = AerothermoOptions()
    aerothermo_options.cat_method = "constant"
    aerothermo_options.cat_rate = 1.0
    aerothermo_options.vel_grad = "fr"
    aerothermo_options.standoff = "freeman"

    return SimpleNamespace(
        planet=SimpleNamespace(name="earth"),
        freestream=SimpleNamespace(method="Standard"),
        aerothermo=aerothermo_options,
    )


def _make_assembly(lref=_LREF):
    """Create one stagnation-facing facet with a one-metre local radius."""
    return SimpleNamespace(
        Lref=lref,
        objects=[],
        mesh=SimpleNamespace(
            facet_normal=np.array([[0.0, 0.0, 1.0]]),
            facet_radius=np.array([1.0]),
        ),
        aerothermo=SimpleNamespace(
            heatflux=np.zeros(1),
            temperature=np.array([_WALL_TEMPERATURE]),
            partial_factor=np.ones(1),
        ),
    )


def _altitude_interpolator(lref=_LREF):
    h_grid = np.linspace(1000.0, 300000.0, 25000)
    return mix_properties.interpolate_atmosphere_knudsen(
        "NRLMSISE00", lref, h_grid
    )


def _evaluate_at_knudsen(
    knudsen, altitude_from_knudsen, assembly, options, lref=_LREF
):
    """Run the production heat-flux selector at a prescribed atmospheric Kn."""
    altitude = float(altitude_from_knudsen(knudsen))
    freestream = SimpleNamespace()

    mix_properties.compute_freestream(
        "NRLMSISE00",
        altitude,
        _VELOCITY,
        lref,
        freestream,
        assembly,
        options,
    )
    mix_properties.compute_stagnation(freestream, options.freestream)

    assembly.freestream = freestream
    assembly.aerothermo.heatflux.fill(0.0)
    aerothermo.compute_aerothermodynamics(
        assembly, [], _VISIBLE_FACET, _FLOW_DIRECTION, options
    )

    stanton_scale = max(freestream.density * freestream.velocity**3 / 2.0, 0.05)
    heatflux = assembly.aerothermo.heatflux[0]
    return altitude, freestream, heatflux, heatflux / stanton_scale


def test_default_thermal_knudsen_limit_is_one_e_minus_three():
    """The default thermal continuum limit is 1e-3."""
    options = _make_options()
    assert options.aerothermo.knc_heatflux == pytest.approx(1e-3)


def test_thermal_bridge_is_continuous_at_default_knudsen_limit():
    """Verify the real thermal path is continuous at the default limit."""
    options = _make_options()

    assembly = _make_assembly()
    altitude_from_knudsen = _altitude_interpolator()
    knudsen_targets = np.array(
        [
            1e-3 * (1.0 - _EPSILON),
            1e-3,
            1e-3 * (1.0 + _EPSILON),
        ]
    )

    stanton_numbers = []
    for target in knudsen_targets:
        _, freestream, _, stanton = _evaluate_at_knudsen(
            target, altitude_from_knudsen, assembly, options
        )
        assert freestream.knudsen == pytest.approx(target, rel=_KN_RTOL)
        stanton_numbers.append(stanton)

    st_minus, st_zero, st_plus = stanton_numbers
    assert np.isclose(st_minus, st_zero, rtol=_ST_RTOL)
    assert np.isclose(st_plus, st_zero, rtol=_ST_RTOL)


@pytest.mark.parametrize("lref", (0.1, 5.0))
def test_default_thermal_transition_converges_without_knudsen_override(lref):
    """End-to-end Earth/NRLMSISE00 transition at 5 km/s and 300 K wall.

    This one-facet, one-metre-radius regression covers representative Lref
    extremes only; it is not a general physical validation of the threshold.
    """
    options = _make_options()
    assembly = _make_assembly(lref)
    altitude_from_knudsen = _altitude_interpolator(lref)
    threshold = options.aerothermo.knc_heatflux

    relative_gaps = []
    for epsilon in _END_TO_END_EPSILONS:
        target_minus = threshold * (1.0 - epsilon)
        target_plus = threshold * (1.0 + epsilon)

        _, freestream_minus, heatflux_minus, _ = _evaluate_at_knudsen(
            target_minus, altitude_from_knudsen, assembly, options, lref
        )
        continuum_minus, _ = _thermal_model_heatfluxes(
            assembly, freestream_minus, options
        )
        _, freestream_plus, heatflux_plus, _ = _evaluate_at_knudsen(
            target_plus, altitude_from_knudsen, assembly, options, lref
        )
        _, bridging_plus = _thermal_model_heatfluxes(
            assembly, freestream_plus, options
        )

        for freestream, target in (
            (freestream_minus, target_minus),
            (freestream_plus, target_plus),
        ):
            # compute_freestream defines Kn as mfp / Lref; this permits only
            # a small floating-point round-off in that production identity.
            assert freestream.knudsen == pytest.approx(
                freestream.mfp / lref, rel=_KN_IDENTITY_RTOL
            )
            # The inverse atmosphere lookup was measured within 3e-13 relative;
            # retain margin for platform-level interpolation variation.
            assert freestream.knudsen == pytest.approx(
                target, rel=_KN_TARGET_RTOL
            )

        # Even at epsilon=1e-4, the reconstructed states are about 1e-7 from
        # the threshold, versus a measured interpolation error below 3e-16 in
        # absolute Kn.  The branch classification is therefore unambiguous.
        assert freestream_minus.knudsen <= threshold
        assert freestream_plus.knudsen > threshold
        assert freestream_plus.knudsen < options.aerothermo.knf
        assert heatflux_minus == pytest.approx(continuum_minus, rel=1e-10)
        assert heatflux_plus == pytest.approx(bridging_plus, rel=1e-10)

        relative_gaps.append(abs(heatflux_plus - heatflux_minus) / heatflux_minus)

    coarse_gap, fine_gap = relative_gaps
    # The audited gap contracts by about ten when epsilon contracts by ten;
    # 2e-4 retains a modest margin over the observed 1.5e-4 fine gap.
    assert fine_gap < coarse_gap / 5.0
    assert fine_gap < _CONTINUITY_GAP_RTOL


def _thermal_model_heatfluxes(assembly, freestream, options):
    """Return production continuum and thermal-bridging heat fluxes."""
    stanton_scale = max(freestream.density * freestream.velocity**3 / 2.0, 0.05)
    partial_factor = assembly.aerothermo.partial_factor[_VISIBLE_FACET][0]
    continuum_heatflux = aerothermo.aerothermodynamics_module_continuum(
        assembly.mesh.facet_normal,
        assembly.mesh.facet_radius,
        freestream,
        _VISIBLE_FACET,
        assembly.aerothermo.temperature,
        _FLOW_DIRECTION,
        options,
        assembly,
    )[0] * stanton_scale * partial_factor
    bridging_heatflux = aerothermo.aerothermodynamics_module_bridging(
        assembly.mesh.facet_normal,
        assembly.mesh.facet_radius,
        freestream,
        _VISIBLE_FACET,
        assembly.aerothermo.temperature,
        _FLOW_DIRECTION,
        "NRLMSISE00",
        options.aerothermo.knc_heatflux,
        options.aerothermo.knf,
        assembly.Lref,
        assembly,
        options,
    )[0] * stanton_scale * partial_factor
    return continuum_heatflux, bridging_heatflux


def test_heatflux_selector_uses_continuum_at_exact_knudsen_threshold():
    """Unit-level equality contract; this deliberately injects Kn, not physics."""
    options = _make_options()
    options.aerothermo.knc_heatflux = 5e-3
    assembly = _make_assembly()
    altitude_from_knudsen = _altitude_interpolator()

    threshold = options.aerothermo.knc_heatflux
    _, freestream, _, _ = _evaluate_at_knudsen(
        threshold, altitude_from_knudsen, assembly, options
    )
    assert freestream.knudsen == pytest.approx(
        freestream.mfp / assembly.Lref, rel=_KN_IDENTITY_RTOL
    )
    assert freestream.knudsen == pytest.approx(threshold, rel=_KN_TARGET_RTOL)

    # This test isolates the selector's equality semantics.  At 5e-3 the
    # production continuum and bridge fluxes differ, unlike at the default.
    freestream.knudsen = threshold
    continuum_heatflux, bridging_heatflux = _thermal_model_heatfluxes(
        assembly, freestream, options
    )
    assert bridging_heatflux / continuum_heatflux > 1.10

    assembly.aerothermo.heatflux.fill(0.0)
    aerothermo.compute_aerothermodynamics(
        assembly, [], _VISIBLE_FACET, _FLOW_DIRECTION, options
    )
    assert assembly.aerothermo.heatflux[0] == pytest.approx(continuum_heatflux)


def test_knudsen_point_zero_point_zero_zero_five_uses_thermal_bridge():
    """Ensure Kn=0.005 is inside, rather than at, the thermal bridge."""
    options = _make_options()
    assembly = _make_assembly()
    altitude_from_knudsen = _altitude_interpolator()

    target_knudsen = 5e-3
    _, freestream, titan_heatflux, _ = _evaluate_at_knudsen(
        target_knudsen, altitude_from_knudsen, assembly, options
    )
    assert freestream.knudsen == pytest.approx(target_knudsen, rel=_KN_RTOL)

    stanton_scale = max(freestream.density * freestream.velocity**3 / 2.0, 0.05)
    continuum_stanton = aerothermo.aerothermodynamics_module_continuum(
        assembly.mesh.facet_normal,
        assembly.mesh.facet_radius,
        freestream,
        _VISIBLE_FACET,
        assembly.aerothermo.temperature,
        _FLOW_DIRECTION,
        options,
        assembly,
    )[0]
    continuum_heatflux = continuum_stanton * stanton_scale

    assert titan_heatflux / continuum_heatflux > 1.10
