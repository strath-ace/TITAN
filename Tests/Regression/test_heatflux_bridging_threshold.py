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


def _make_assembly():
    """Create one stagnation-facing facet with a one-metre local radius."""
    return SimpleNamespace(
        Lref=_LREF,
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


def _altitude_interpolator():
    h_grid = np.linspace(1000.0, 300000.0, 25000)
    return mix_properties.interpolate_atmosphere_knudsen(
        "NRLMSISE00", _LREF, h_grid
    )


def _evaluate_at_knudsen(knudsen, altitude_from_knudsen, assembly, options):
    """Run the production heat-flux selector at a prescribed atmospheric Kn."""
    altitude = float(altitude_from_knudsen(knudsen))
    freestream = SimpleNamespace()

    mix_properties.compute_freestream(
        "NRLMSISE00",
        altitude,
        _VELOCITY,
        _LREF,
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


def test_default_thermal_knudsen_limit_matches_fostrad():
    """The default thermal continuum limit follows the FOSTRAD model."""
    options = _make_options()
    assert options.aerothermo.knc_heatflux == pytest.approx(1e-3)


def test_thermal_bridge_is_continuous_at_fostrad_knudsen_limit():
    """Verify the real thermal path is continuous at the FOSTRAD limit."""
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
