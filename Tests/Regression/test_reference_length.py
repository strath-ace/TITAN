# Tests/Regression/test_reference_length.py
from types import SimpleNamespace

import numpy as np
import pytest

from Geometry import assembly as AssemblyModule
from Geometry import mesh as Mesh


def _asymmetric_box_component():
    """Create a closed body-frame box: X=2 m, Y=4 m, Z=1 m."""

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 4.0, 1.0],
            [0.0, 4.0, 1.0],
        ]
    )

    facets = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # z = 0
            [4, 5, 6], [4, 6, 7],  # z = 1
            [0, 1, 5], [0, 5, 4],  # y = 0
            [1, 2, 6], [1, 6, 5],  # x = 2
            [2, 3, 7], [2, 7, 6],  # y = 4
            [3, 0, 4], [3, 4, 7],  # x = 0
        ]
    )

    mesh = Mesh.Mesh([])
    mesh.v0 = vertices[facets[:, 0]]
    mesh.v1 = vertices[facets[:, 1]]
    mesh.v2 = vertices[facets[:, 2]]

    mesh = Mesh.compute_mesh(mesh, compute_radius=False)

    return SimpleNamespace(mesh=mesh, temperature=300.0)


def test_reference_length_is_the_body_longitudinal_x_extent(monkeypatch):
    # Curvature is unrelated to Lref and can be expensive for a unit test.
    def cheap_curvature(nodes, facets, *_):
        return (
            np.ones(len(nodes)),
            np.ones(len(facets)),
            np.ones(len(nodes)),
            np.ones((len(facets), 3)),
        )

    monkeypatch.setattr(Mesh, "compute_curvature", cheap_curvature)

    options = SimpleNamespace(
        ablation_mode="0D",
        post_fragment_tetra_ablation=False,
    )

    assembly = AssemblyModule.Assembly(
        objects=[_asymmetric_box_component()],
        options=options,
    )

    np.testing.assert_allclose(
        assembly.mesh.xmax - assembly.mesh.xmin,
        [2.0, 4.0, 1.0],
    )

    # TITAN documents body-frame X as the longitudinal/forward axis.
    assert assembly.Lref == pytest.approx(2.0)
