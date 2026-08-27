import numpy as np
import pandas as pd
import trimesh
import pathlib

POINTS_PER_RING = 0


def rings_to_mesh(rings, cap_ends=True):
    """
    Convert successive rings of 36 points into a triangular trimesh mesh.

    Parameters
    ----------
    rings : ndarray, shape (N, 36, 3)
        Ordered 3D points for each ring.

    cap_ends : bool
        Whether to cap the first and last rings.

    Returns
    -------
    trimesh.Trimesh
    """

    # rings = np.asarray(rings, dtype=float)

    # if rings.ndim != 3:
    #     raise ValueError(
    #         f"Expected a 3D array, got shape {rings.shape}"
    #     )

    # if rings.shape[1] != POINTS_PER_RING:
    #     raise ValueError(
    #         f"Expected {POINTS_PER_RING} points per ring, "
    #         f"got {rings.shape[1]}"
    #     )

    # if rings.shape[2] != 3:
    #     raise ValueError("Each point must have 3 coordinates.")

    n_rings = int(np.floor(len(rings)/POINTS_PER_RING))

    # Flatten all points into one vertex array.
    vertices = rings.reshape(-1, 3)

    faces = []

    # ------------------------------------------------------------
    # Connect neighboring rings
    #
    # Each pair of corresponding points creates a quad:
    #
    #       a ----- b
    #       |       |
    #       |       |
    #       c ----- d
    #
    # which is split into two triangles.
    # ------------------------------------------------------------

    for i in range(n_rings - 1):

        ring0 = i * POINTS_PER_RING
        ring1 = (i + 1) * POINTS_PER_RING

        for j in range(POINTS_PER_RING):

            k = (j + 1) % POINTS_PER_RING

            a = ring0 + j
            b = ring0 + k
            c = ring1 + j
            d = ring1 + k

            faces.append([a, c, b])
            faces.append([b, c, d])

    # ------------------------------------------------------------
    # End caps
    # ------------------------------------------------------------

    if cap_ends:

        # Add a center vertex for each cap.
        start_center = len(vertices)
        end_center = start_center + 1

        vertices = np.vstack([
            vertices,
            rings[0].mean(axis=0),
            rings[-1].mean(axis=0),
        ])

        # Start cap
        for j in range(POINTS_PER_RING):
            k = (j + 1) % POINTS_PER_RING

            faces.append([
                start_center,
                k,
                j,
            ])

        # End cap
        last = (n_rings - 1) * POINTS_PER_RING

        for j in range(POINTS_PER_RING):
            k = (j + 1) % POINTS_PER_RING

            faces.append([
                end_center,
                last + j,
                last + k,
            ])

    faces = np.asarray(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    # Let trimesh calculate consistent normals.
    mesh.fix_normals()

    return mesh


def write_stl(rings, filename="tube.stl", cap_ends=True):
    mesh = rings_to_mesh(rings, cap_ends=cap_ends)

    mesh.export(filename)

    print(f"Saved: {filename}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces:    {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Volume:     {mesh.volume}")


# ----------------------------------------------------------------
# Example
# ----------------------------------------------------------------

if __name__ == "__main__":

    csv_path = pathlib.Path('./reachable.csv').resolve()
    csv = pd.read_csv(csv_path)
    rings = csv[['ECEF_X','ECEF_Y','ECEF_Z']].to_numpy()
    POINTS_PER_RING = len(csv[csv['Iter']==1]['Iter'].to_numpy())
    write_stl(rings, "tube.stl", cap_ends=False)
    rings = csv[['Longitude','Latitude','Altitude']].to_numpy()
    rings[:,-1] = 1e-3*rings[:,-1]
    write_stl(rings, "tube_geo.stl", cap_ends=False)
    rings = csv[['ECEF_U','ECEF_V','ECEF_W']].to_numpy()
    #rings[:,-1] = 1e-3*rings[:,-1]
    write_stl(rings, "tube_vel.stl", cap_ends=False)

    rings = csv[['Time','Mass','T']].to_numpy()
        #rings[:,-1] = 1e-3*rings[:,-1]
    write_stl(rings, "tube_demise.stl", cap_ends=False)
