"""
Geometric substepping with local retriangulation for remesh-only ablation.

Caps nodal motion per substep relative to mean edge length. After each
substep, triangles that have collapsed below A_min are detected; the bad
patch (+ 1-ring) is removed, boundary loop extracted, projected to 2D,
retriangulated with constrained Delaunay + interior points, and the new
triangles spliced back in.

Requires the 'triangle' library (pip install triangle) for CDT.
"""

import numpy as np
import trimesh
import triangle as tr
from collections import defaultdict
from Geometry.mesh import sync_surface_from_nodes, map_edges_connectivity

A_MIN_RATIO = 0.05
#0.05

def compute_mean_edge_length(nodes, facets):
    """Return scalar mean edge length of the triangle mesh."""
    e0 = nodes[facets[:, 1]] - nodes[facets[:, 0]]
    e1 = nodes[facets[:, 2]] - nodes[facets[:, 1]]
    e2 = nodes[facets[:, 0]] - nodes[facets[:, 2]]
    lengths = np.concatenate([
        np.linalg.norm(e0, axis=1),
        np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1),
    ])
    return float(np.mean(lengths))


def substep_count(full_disp, max_disp_per_substep_m):
    """Number of substeps so max nodal motion <= max_disp_per_substep_m [m]."""
    max_disp = float(np.max(np.linalg.norm(full_disp, axis=1)))
    if max_disp_per_substep_m <= 0 or max_disp <= 0:
        return 1
    return max(1, int(np.ceil(max_disp / max_disp_per_substep_m)))


def triangle_areas(nodes, facets):
    """Area of each triangle via half cross-product magnitude."""
    v0 = nodes[facets[:, 0]]
    v1 = nodes[facets[:, 1]]
    v2 = nodes[facets[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def flag_bad_triangles(areas):
    """Return (bad_area_mask, A_min). bad_area_mask is True for triangles below A_min."""
    mean_area = float(np.mean(areas)) if len(areas) > 0 else 0.0
    a_min = A_MIN_RATIO * mean_area
    return areas < a_min, a_min


def triangle_min_angles_deg(nodes, facets):
    """Minimum angle [degrees] for each triangle (from edge lengths, law of cosines)."""
    v0 = nodes[facets[:, 0]]
    v1 = nodes[facets[:, 1]]
    v2 = nodes[facets[:, 2]]
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    # Avoid div by zero; angles in [0, 180]
    a, b, c = np.maximum(a, 1e-12), np.maximum(b, 1e-12), np.maximum(c, 1e-12)
    cos_a = np.clip((b*b + c*c - a*a) / (2*b*c), -1.0, 1.0)
    cos_b = np.clip((a*a + c*c - b*b) / (2*a*c), -1.0, 1.0)
    cos_c = np.clip((a*a + b*b - c*c) / (2*a*b), -1.0, 1.0)
    ang_a = np.degrees(np.arccos(cos_a))
    ang_b = np.degrees(np.arccos(cos_b))
    ang_c = np.degrees(np.arccos(cos_c))
    return np.minimum(np.minimum(ang_a, ang_b), ang_c)


def _build_face_adjacency(facets, nodes):
    """Build face adjacency (nbr_map: face_id -> set of neighbour face ids)."""
    mesh = trimesh.Trimesh(vertices=nodes, faces=facets, process=False)
    adj = mesh.face_adjacency
    nbr_map = defaultdict(set)
    for a, b in adj:
        nbr_map[a].add(b)
        nbr_map[b].add(a)
    return nbr_map


def build_patch_1ring(bad_mask, facets, nodes):
    """
    Return sorted array of face indices: bad faces + their 1-ring neighbours.
    Uses trimesh face_adjacency for connectivity.
    """
    return build_patch_n_ring(bad_mask, facets, nodes, n_rings=1)


def build_patch_n_ring(bad_mask, facets, nodes, n_rings=1):
    """
    Return sorted array of face indices: bad faces + n_rings of neighbours.
    n_rings=1 -> bad + 1-ring, n_rings=2 -> bad + 2-ring, n_rings=3 -> bad + 3-ring.
    """
    bad_ids = set(np.where(bad_mask)[0])
    if not bad_ids:
        return np.array([], dtype=int)

    nbr_map = _build_face_adjacency(facets, nodes)
    patch = set(bad_ids)
    frontier = set(bad_ids)
    for _ in range(n_rings):
        next_frontier = set()
        for fid in frontier:
            next_frontier.update(nbr_map.get(fid, set()))
        next_frontier -= patch
        patch.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return np.array(sorted(patch), dtype=int)


def _boundary_loop(patch_faces, facets):
    """
    Extract boundary edges of the patch and chain them into closed loop(s).

    Returns a list of loops; each loop is an ordered list of global vertex ids.
    """
    patch_set = set(patch_faces)
    edge_count = defaultdict(int)
    edge_face = defaultdict(list)

    for fid in patch_faces:
        tri = facets[fid]
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = (min(a, b), max(a, b))
            edge_count[key] += 1
            edge_face[key].append(fid)

    boundary_edges = []
    for key, cnt in edge_count.items():
        if cnt == 1:
            boundary_edges.append(key)

    if not boundary_edges:
        return []

    adj = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)

    visited_edges = set()
    loops = []
    remaining_verts = set()
    for a, b in boundary_edges:
        remaining_verts.add(a)
        remaining_verts.add(b)

    while remaining_verts:
        start = next(iter(remaining_verts))
        loop = [start]
        remaining_verts.discard(start)
        current = start

        while True:
            nbrs = adj[current]
            next_node = None
            for n in nbrs:
                edge_key = (min(current, n), max(current, n))
                if edge_key not in visited_edges:
                    next_node = n
                    visited_edges.add(edge_key)
                    break

            if next_node is None or next_node == start:
                break

            loop.append(next_node)
            remaining_verts.discard(next_node)
            current = next_node

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def _patch_average_normal(nodes, facets, patch_faces):
    """Compute area-weighted average normal of the patch faces."""
    v0 = nodes[facets[patch_faces, 0]]
    v1 = nodes[facets[patch_faces, 1]]
    v2 = nodes[facets[patch_faces, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    avg = normals.sum(axis=0)
    norm = np.linalg.norm(avg)
    if norm < 1e-30:
        return np.array([0.0, 0.0, 1.0])
    return avg / norm


def _build_2d_frame(normal):
    """Return (e1, e2) orthonormal to the given normal."""
    n = normal / (np.linalg.norm(normal) + 1e-30)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, candidate)
    e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2) + 1e-30
    return e1, e2


def _point_in_polygon_2d(p, poly_2d):
    """Ray-casting (even-odd): True if p (2,) is inside polygon poly_2d (N, 2)."""
    x, y = p[0], p[1]
    n = len(poly_2d)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly_2d[i][0], poly_2d[i][1]
        xj, yj = poly_2d[j][0], poly_2d[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _interior_points_2d(poly_2d, h_target, target_count=None):
    """
    Generate 2D points inside the polygon on a grid with spacing ~h_target.
    poly_2d: (N, 2) ordered boundary vertices. Returns (K, 2) interior points.
    If target_count is set, return exactly that many points (subsample or refine grid).
    """
    if poly_2d.size == 0:
        return np.zeros((0, 2))
    xmin, ymin = poly_2d.min(axis=0)
    xmax, ymax = poly_2d.max(axis=0)
    if h_target <= 0:
        return np.zeros((0, 2))
    if target_count is not None and target_count <= 0:
        return np.zeros((0, 2))

    def grid_inside(h):
        nx = max(1, int((xmax - xmin) / h) + 1)
        ny = max(1, int((ymax - ymin) / h) + 1)
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        inside = np.array([_point_in_polygon_2d(c, poly_2d) for c in candidates])
        return candidates[inside]

    points = grid_inside(h_target)
    if target_count is not None:
        h = h_target
        while len(points) < target_count:
            h = 0.5 * h
            if h <= 0:
                break
            points = grid_inside(h)
        if len(points) > target_count:
            idx = np.linspace(0, len(points) - 1, target_count, dtype=int)
            points = points[idx]
        elif len(points) < target_count:
            points = points  # return what we have (patch may be small)
    return points


def _retriangulate_patch(nodes, facets, patch_faces, boundary_ids, h_target):
    """
    Project boundary loop to 2D, add interior points, run constrained Delaunay
    triangulation, map back to 3D global indices.
    Interior point count is chosen so new patch has same facet count as removed:
    T = 2*I + B - 2 => I_target = (n_patch - B + 2) // 2.

    Returns (new_faces, new_nodes_3d) or (None, None) on failure.
    new_faces: (M, 3) int array of global node indices (boundary + new interior).
    new_nodes_3d: (K, 3) new node positions to append to assembly.mesh.nodes.
    """
    n_nodes = len(nodes)
    boundary_ids = np.asarray(boundary_ids, dtype=int)
    n_boundary = len(boundary_ids)
    if n_boundary < 3:
        return None, None

    n_patch = len(patch_faces)
    I_target = max(0, (n_patch - n_boundary + 2) // 2)

    avg_normal = _patch_average_normal(nodes, facets, patch_faces)
    e1, e2 = _build_2d_frame(avg_normal)
    centroid = nodes[boundary_ids].mean(axis=0)

    local_pts = nodes[boundary_ids] - centroid
    boundary_2d = np.column_stack([local_pts @ e1, local_pts @ e2])

    interior_2d = _interior_points_2d(boundary_2d, h_target, target_count=I_target)
    n_interior = len(interior_2d)
    vertices_2d = np.vstack([boundary_2d, interior_2d]) if n_interior > 0 else boundary_2d

    segments = np.column_stack([
        np.arange(n_boundary),
        np.roll(np.arange(n_boundary), -1),
    ]).astype(np.int32)

    try:
        tri_input = dict(vertices=vertices_2d.astype(np.float64), segments=segments)
        result = tr.triangulate(tri_input, "p")
    except Exception:
        return None, None

    if "triangles" not in result or result["triangles"].size == 0:
        return None, None

    tris = result["triangles"]

    def to_global(idx):
        if idx < n_boundary:
            return int(boundary_ids[idx])
        return int(n_nodes + (idx - n_boundary))

    new_faces = np.array(
        [[to_global(int(t[0])), to_global(int(t[1])), to_global(int(t[2]))]
         for t in tris],
        dtype=np.int64,
    )

    if n_interior > 0:
        new_nodes_3d = centroid + interior_2d @ np.stack([e1, e2], axis=0)
    else:
        new_nodes_3d = np.zeros((0, 3))

    return new_faces, new_nodes_3d


def _sync_object_mesh_from_assembly(assembly_mesh, obj):
    """
    Update obj.mesh from assembly_mesh using obj.facet_index (global node ids).
    Rebuilds local nodes, local facets, node_index, edges, and syncs surface.
    """
    obj.mesh.v0 = assembly_mesh.v0[obj.facet_index]
    obj.mesh.v1 = assembly_mesh.v1[obj.facet_index]
    obj.mesh.v2 = assembly_mesh.v2[obj.facet_index]
    obj.mesh.facets = assembly_mesh.facets[obj.facet_index]

    global_to_local = {}
    local_idx = 0
    for tri in obj.mesh.facets:
        for gid in tri:
            if gid not in global_to_local:
                global_to_local[gid] = local_idx
                local_idx += 1
    n_local_nodes = local_idx
    obj.mesh.nodes = np.zeros((n_local_nodes, 3))
    for gid, lid in global_to_local.items():
        obj.mesh.nodes[lid] = assembly_mesh.nodes[gid]
    local_facets = np.zeros_like(obj.mesh.facets)
    for i, tri in enumerate(obj.mesh.facets):
        for j in range(3):
            local_facets[i, j] = global_to_local[tri[j]]
    obj.mesh.facets = local_facets

    local_to_global = {v: k for k, v in global_to_local.items()}
    obj.node_index = np.array([local_to_global[i] for i in range(n_local_nodes)], dtype=int)

    obj.mesh.edges, obj.mesh.facet_edges = map_edges_connectivity(obj.mesh.facets)
    sync_surface_from_nodes(obj.mesh)


def replace_patch(assembly_mesh, obj, patch_face_indices, new_faces):
    """
    Splice new_faces into assembly_mesh.facets in place of patch_face_indices.
    Update obj.facet_index so it stays consistent.

    Returns
    -------
    new_to_old_obj : np.ndarray (int)
        For each new object facet index j, old object facet index, or -1 if
        this facet is a new patch facet (no direct counterpart). Used to map
        PATO boundary fields after patch replacement.
    new_to_old_assembly : np.ndarray (int)
        For each new assembly facet index j, old assembly facet index, or -1
        for new patch facets. Used to remap assembly-level per-facet arrays.
    old_to_new : np.ndarray (int)
        old_to_new[old_assembly_idx] = new assembly index (or -1 if removed).
        Used to update other objects' facet_index in multi-object assemblies.
    """
    old_facets = assembly_mesh.facets
    n_old = len(old_facets)
    n_patch = len(patch_face_indices)
    n_new = len(new_faces)

    keep_mask = np.ones(n_old, dtype=bool)
    keep_mask[patch_face_indices] = False
    kept_facets = old_facets[keep_mask]

    insert_pos = int(patch_face_indices[0])
    assembly_mesh.facets = np.vstack([
        kept_facets[:insert_pos],
        new_faces,
        kept_facets[insert_pos:],
    ])

    old_to_new = np.full(n_old, -1, dtype=int)
    new_idx = 0
    for old_idx in range(n_old):
        if keep_mask[old_idx]:
            if old_idx < insert_pos:
                old_to_new[old_idx] = new_idx
            else:
                old_to_new[old_idx] = new_idx + n_new
            new_idx += 1

    updated_facet_index = []
    new_to_old_obj = []
    patch_set = set(patch_face_indices)
    new_patch_indices = list(range(insert_pos, insert_pos + n_new))
    patch_entries_replaced = 0

    for k, fi in enumerate(obj.facet_index):
        if fi in patch_set:
            if patch_entries_replaced < n_new:
                updated_facet_index.append(new_patch_indices[patch_entries_replaced])
                new_to_old_obj.append(-1)  # new patch facet, interpolate later
                patch_entries_replaced += 1
        else:
            mapped = old_to_new[fi]
            if mapped >= 0:
                updated_facet_index.append(mapped)
                new_to_old_obj.append(k)

    for i in range(patch_entries_replaced, n_new):
        updated_facet_index.append(new_patch_indices[i])
        new_to_old_obj.append(-1)

    obj.facet_index = np.array(updated_facet_index, dtype=int)

    sync_surface_from_nodes(assembly_mesh)
    _sync_object_mesh_from_assembly(assembly_mesh, obj)

    # Assembly-level mapping: for each new assembly facet j, old index or -1 (new patch)
    n_new_assembly = n_old - n_patch + n_new
    new_to_old_assembly = np.full(n_new_assembly, -1, dtype=np.int64)
    for j in range(insert_pos):
        new_to_old_assembly[j] = j
    for j in range(insert_pos + n_new, n_new_assembly):
        new_to_old_assembly[j] = j - n_new + n_patch

    return (
        np.array(new_to_old_obj, dtype=np.int64),
        new_to_old_assembly,
        old_to_new,
    )


def map_pato_fields_after_patch_replace(obj, old_arrays, new_to_old_obj, patch_avg=None):
    """
    Map PATO boundary arrays from old facet count to new after patch replacement.
    Copies values where new facet has a direct old counterpart; for new patch
    facets (new_to_old_obj[j] == -1), uses patch_avg when provided (preserves
    thermal/field content of the replaced region), otherwise interpolates from
    neighboring facets.

    Parameters
    ----------
    obj : component with obj.pato and obj.mesh
    old_arrays : dict
        Keys: 'q_conv_field', 'mdot_field', 'v_n_field', 'temperature',
        'hf_cond', 'mDotVapor', 'mDotMelt', 'molten' (and optionally others).
        Values: 1D arrays of length = old object facet count.
    new_to_old_obj : np.ndarray (int)
        new_to_old_obj[j] = old object facet index, or -1 for new patch facets.
    patch_avg : dict, optional
        If provided, patch_avg[name] = mean value over the removed patch for
        that field. New patch facets are initialized to this value so the
        remeshed region does not appear cold.
    """
    n = len(obj.facet_index)
    nbr_map = _build_face_adjacency(obj.mesh.facets, obj.mesh.nodes)

    pato_float = np.float64
    for name, old_arr in old_arrays.items():
        if old_arr is None or len(old_arr) == 0:
            # Resize to avoid length mismatch; fill with 0
            z = np.zeros(n, dtype=pato_float)
            setattr(obj.pato, name, z)
            if name == "temperature":
                obj.temperature = z
            continue
        new_arr = np.zeros(n, dtype=old_arr.dtype)
        # Copy from old where we have a direct mapping
        for j in range(n):
            o = new_to_old_obj[j]
            if o >= 0:
                new_arr[j] = old_arr[o]
        # New patch facets: use patch average when available, else interpolate from neighbors
        use_patch_avg = patch_avg is not None and name in patch_avg
        for j in range(n):
            if new_to_old_obj[j] != -1:
                continue
            if use_patch_avg:
                new_arr[j] = float(patch_avg[name])
            else:
                neighbors = nbr_map.get(j, set())
                vals = [new_arr[nbr] for nbr in neighbors if new_to_old_obj[nbr] >= 0]
                if vals:
                    new_arr[j] = float(np.mean(vals))
                else:
                    filled = new_arr[new_to_old_obj >= 0]
                    new_arr[j] = float(np.mean(filled)) if len(filled) > 0 else 0.0
        # Single smoothing pass: blend new patch facets with all neighbors (avoids sharp step at boundary)
        for j in range(n):
            if new_to_old_obj[j] != -1:
                continue
            neighbors = nbr_map.get(j, set())
            vals = [new_arr[nbr] for nbr in neighbors]
            if vals:
                new_arr[j] = float(np.mean(vals))

        setattr(obj.pato, name, new_arr)

    # Keep obj.temperature in sync if we updated pato.temperature
    if "temperature" in old_arrays and old_arrays["temperature"] is not None:
        obj.temperature = np.asarray(obj.pato.temperature, dtype=np.float64)


def map_assembly_fields_after_patch_replace(assembly, new_to_old_assembly):
    """
    Remap assembly-level per-facet arrays to the new facet count after patch
    replacement. Copies values where new facet has an old counterpart;
    interpolates from neighbors for new patch facets (new_to_old_assembly[j]==-1).
    Works for single- and multi-object assemblies; only remaps if array length
    does not match the new count.
    """
    n = len(new_to_old_assembly)
    nbr_map = _build_face_adjacency(assembly.mesh.facets, assembly.mesh.nodes)

    def _remap_array(old_arr, ndim=1):
        if old_arr is None or getattr(old_arr, "size", 0) == 0:
            return None
        if len(old_arr) == n:
            return np.asarray(old_arr)
        if ndim == 1:
            new_arr = np.zeros(n, dtype=old_arr.dtype)
        else:
            new_arr = np.zeros((n,) + old_arr.shape[1:], dtype=old_arr.dtype)
        for j in range(n):
            o = new_to_old_assembly[j]
            if o >= 0:
                new_arr[j] = old_arr[o]
        for j in range(n):
            if new_to_old_assembly[j] != -1:
                continue
            neighbors = nbr_map.get(j, set())
            vals = [new_arr[nbr] for nbr in neighbors if new_to_old_assembly[nbr] >= 0]
            if vals:
                new_arr[j] = np.mean(vals, axis=0)
            else:
                filled = new_arr[new_to_old_assembly >= 0]
                new_arr[j] = np.mean(filled, axis=0) if len(filled) > 0 else 0
        for j in range(n):
            if new_to_old_assembly[j] != -1:
                continue
            neighbors = nbr_map.get(j, set())
            vals = [new_arr[nbr] for nbr in neighbors]
            if vals:
                new_arr[j] = np.mean(vals, axis=0)
        return new_arr

    # Aerothermo 1D
    for name in ("temperature", "pressure", "heatflux", "theta", "he", "hw", "Te",
                 "rhoe", "ue", "debug_alpha", "density"):
        if hasattr(assembly.aerothermo, name):
            arr = getattr(assembly.aerothermo, name)
            if getattr(arr, "shape", ()) and len(arr) != n:
                setattr(assembly.aerothermo, name, _remap_array(arr))
    # Aerothermo 2D
    for name in ("shear", "momentum", "ce_i"):
        if hasattr(assembly.aerothermo, name):
            arr = getattr(assembly.aerothermo, name)
            if getattr(arr, "shape", ()) and len(arr) != n:
                setattr(assembly.aerothermo, name, _remap_array(arr, ndim=2))

    # Other assembly per-facet 1D arrays
    for name in ("emissive_power", "emissivity", "material_density", "hf_cond",
                 "blackbody_emissions_OI_surf", "blackbody_emissions_AlI_surf",
                 "atomic_emissions_OI_surf", "atomic_emissions_AlI_surf",
                 "angle_blackbody", "angle_atomic"):
        if hasattr(assembly, name):
            arr = getattr(assembly, name)
            if getattr(arr, "shape", ()) and len(arr) != n:
                setattr(assembly, name, _remap_array(arr))
    # PATO ablation (may not exist if not PATO or Ta_bc != 'ablation')
    for name in ("mDotVapor", "mDotMelt", "mVapor", "mMelt", "updated_gas_density", "LOS"):
        if hasattr(assembly, name):
            arr = getattr(assembly, name)
            if getattr(arr, "shape", ()) and len(arr) != n:
                setattr(assembly, name, _remap_array(arr))


def _attempt_repair(nodes, facets, patch_faces, a_min, h_target, debug_label=""):
    """
    Try to repair collapsed triangles using CDT + interior points.
    Supports a single boundary loop or multiple separate loops (disconnected bad regions).
    Returns (new_faces, new_nodes_3d). new_faces is None if repair failed.
    Caller must append new_nodes_3d to assembly.mesh.nodes (if non-empty)
    before using new_faces, then run area check and replace_patch.
    """
    loops = _boundary_loop(patch_faces, facets)

    if len(loops) == 0:
        print("[Substep] WARNING: no boundary loop found; "
              "skipping repair for this substep", flush=True)
        return None, None

    patch_faces = np.asarray(patch_faces, dtype=int)
    n_nodes = len(nodes)

    # --- Single-loop case: keep existing behaviour. ---
    if len(loops) == 1:
        boundary_ids = loops[0]
        new_faces, new_nodes_3d = _retriangulate_patch(
            nodes, facets, patch_faces, boundary_ids, h_target
        )
        if new_faces is None or len(new_faces) == 0:
            print("[Substep] WARNING: retriangulation produced no triangles; "
                  "accepting mesh as-is", flush=True)
            return None, None
        return new_faces, new_nodes_3d

    # --- Multiple loops: separate bad regions. ---
    # Partition patch faces by which loop's 2D polygon contains each face centroid,
    # then retriangulate each region and merge results (one append, one replace_patch).
    print(f"[Substep] INFO: {len(loops)} boundary loops detected; "
          "treating them as separate regions for repair.", flush=True)

    # Build a single local 2D frame and centroid for the full patch so all loops and
    # face centroids are projected consistently.
    avg_normal = _patch_average_normal(nodes, facets, patch_faces)
    e1, e2 = _build_2d_frame(avg_normal)
    all_loop_verts = np.unique(np.concatenate(loops))
    centroid = nodes[all_loop_verts].mean(axis=0)

    # Project each boundary loop to 2D for point-in-polygon tests.
    loops_2d = []
    for loop in loops:
        local_pts = nodes[loop] - centroid
        poly_2d = np.column_stack([local_pts @ e1, local_pts @ e2])
        loops_2d.append(poly_2d)

    # Assign each patch face to exactly one loop: the one whose 2D polygon contains
    # its centroid. For separate regions the polygons do not overlap, so each face
    # should naturally belong to a single loop. If no polygon strictly contains the
    # centroid (e.g. numerical edge case), assign to the nearest loop in 2D.
    patch_faces_per_loop = [[] for _ in loops]
    for fid in patch_faces:
        c_3d = nodes[facets[fid]].mean(axis=0)
        c_2d = (c_3d - centroid) @ np.column_stack([e1, e2])
        assigned = False
        for i, poly_2d in enumerate(loops_2d):
            if _point_in_polygon_2d(c_2d, poly_2d):
                patch_faces_per_loop[i].append(fid)
                assigned = True
                break
        if not assigned:
            dists = [np.linalg.norm(c_2d - p.mean(axis=0)) for p in loops_2d]
            i = int(np.argmin(dists))
            patch_faces_per_loop[i].append(fid)

    # Retriangulate each region independently using the existing single-loop routine.
    new_faces_list = []
    new_nodes_list = []
    for i, loop in enumerate(loops):
        faces_i = np.array(patch_faces_per_loop[i], dtype=int)
        if len(faces_i) == 0:
            continue
        boundary_ids_i = np.asarray(loop, dtype=int)
        region_faces, region_nodes = _retriangulate_patch(
            nodes, facets, faces_i, boundary_ids_i, h_target
        )
        if region_faces is None or len(region_faces) == 0:
            print(f"[Substep] WARNING: retriangulation failed for region {i+1}/{len(loops)}; "
                  "skipping patch repair.", flush=True)
            return None, None
        new_faces_list.append(region_faces)
        new_nodes_list.append(region_nodes)

    if not new_faces_list:
        print("[Substep] WARNING: no new faces produced for any region; "
              "skipping patch repair.", flush=True)
        return None, None

    # Concatenate all new interior nodes into a single block to append to the mesh.
    new_nodes_3d_all = np.vstack(new_nodes_list) if new_nodes_list else np.zeros((0, 3))

    # Merge region-wise face lists into one, adjusting interior vertex indices so that
    # each region's interior nodes occupy a disjoint range after concatenation.
    # For region i, interior indices are initially in [n_nodes, n_nodes + n_i);
    # after concatenation they must be in [n_nodes + prefix_i, n_nodes + prefix_i + n_i),
    # where prefix_i = sum_{j < i} n_j.
    new_faces_all_list = []
    prefix = 0
    for region_faces, region_nodes in zip(new_faces_list, new_nodes_list):
        n_i = len(region_nodes)
        offset = prefix
        prefix += n_i
        renumbered = np.where(
            region_faces >= n_nodes,
            n_nodes + offset + (region_faces - n_nodes),
            region_faces,
        )
        new_faces_all_list.append(renumbered)

    new_faces_all = np.vstack(new_faces_all_list)

    return new_faces_all, new_nodes_3d_all


def run_substep_loop(assembly, obj, full_disp, options=None):
    """
    Apply full_disp to assembly.mesh.nodes in geometric substeps.
    After each substep, detect collapsed triangles (area or min angle) and attempt local repair.
    Tries 1-ring patch first; on invalid result retries with 2-ring, then 3-ring.
    Updates assembly.mesh (nodes, facets, derived) and obj mesh in place.

    options: if provided, options.pato.max_substep_displacement_mm [mm] and
    options.pato.min_triangle_angle_deg [deg] are used; otherwise defaults 0.2 and 20.
    """
    nodes = assembly.mesh.nodes
    facets = assembly.mesh.facets
    mel = compute_mean_edge_length(nodes, facets)

    max_disp_mm = 0.2
    min_angle_deg = 15
    if options is not None and hasattr(options, "pato"):
        max_disp_mm = getattr(options.pato, "max_substep_displacement_mm", 0.2)
        min_angle_deg = getattr(options.pato, "min_triangle_angle_deg", 15)
    max_disp_per_substep_m = max_disp_mm * 1e-3
    n_sub = substep_count(full_disp, max_disp_per_substep_m)

    if n_sub > 1:
        print(f"[Substep] Splitting recession into {n_sub} substeps "
              f"(max displacement per substep = {max_disp_mm:.2f} mm)", flush=True)

    disp_chunk = full_disp / n_sub
    max_disp_total = float(np.max(np.linalg.norm(full_disp, axis=1)))
    max_disp_per_substep = max_disp_total / n_sub

    for s in range(n_sub):
        print(f"[Substep {s+1}/{n_sub}] Displacement capped to {max_disp_per_substep*1e3:.4f} mm "
              f"(limit {max_disp_mm:.2f} mm). Applying chunk...", flush=True)
        nodes_backup = assembly.mesh.nodes.copy()

        assembly.mesh.nodes += disp_chunk
        sync_surface_from_nodes(assembly.mesh)

        areas = triangle_areas(assembly.mesh.nodes, assembly.mesh.facets)
        bad_area, a_min = flag_bad_triangles(areas)
        min_angles_deg = triangle_min_angles_deg(assembly.mesh.nodes, assembly.mesh.facets)
        bad_angle = min_angles_deg < min_angle_deg
        bad_mask = bad_area | bad_angle

        if not np.any(bad_mask):
            print(f"[Substep {s+1}/{n_sub}] Accepted (no bad triangles).", flush=True)
            continue

        n_bad = int(np.sum(bad_mask))
        n_bad_area = int(np.sum(bad_area))
        n_bad_angle = int(np.sum(bad_angle))
        print(f"[Substep {s+1}/{n_sub}] Bad triangles detected: {n_bad} "
              f"(area<A_min: {n_bad_area}, angle<{min_angle_deg}°: {n_bad_angle}, A_min = {a_min:.3e}). Attempting patch...", flush=True)

        new_faces = None
        success = False
        patch_faces = None

        for n_ring in range(1, 4):
            patch_faces = build_patch_n_ring(
                bad_mask, assembly.mesh.facets, assembly.mesh.nodes, n_rings=n_ring
            )
            if len(patch_faces) == 0:
                print(f"[Substep {s+1}/{n_sub}] Patch ({n_ring}-ring): no faces, skipping.", flush=True)
                continue

            print(f"[Substep {s+1}/{n_sub}] Attempting repair with {n_ring}-ring patch "
                  f"({len(patch_faces)} triangles)...", flush=True)

            new_faces, new_nodes_3d = _attempt_repair(
                assembly.mesh.nodes, assembly.mesh.facets, patch_faces, a_min, h_target=mel
            )

            if new_faces is not None:
                if new_nodes_3d is not None and len(new_nodes_3d) > 0:
                    assembly.mesh.nodes = np.vstack([
                        assembly.mesh.nodes,
                        new_nodes_3d.astype(assembly.mesh.nodes.dtype),
                    ])
                new_areas = triangle_areas(assembly.mesh.nodes, new_faces)
                success = not np.any(new_areas < a_min)
                print(f"[Substep {s+1}/{n_sub}] New patch created: {len(new_faces)} triangles; "
                      f"valid={success}.", flush=True)
                if success:
                    break
                if n_ring < 3:
                    print(f"[Substep {s+1}/{n_sub}] Patch still has small triangles; "
                          f"retrying with {n_ring + 1}-ring...", flush=True)
                else:
                    print(f"[Substep {s+1}/{n_sub}] 3-ring patch still invalid; accepting mesh as-is.", flush=True)
                    break
            else:
                if n_ring < 3:
                    print(f"[Substep {s+1}/{n_sub}] Repair failed; retrying with {n_ring + 1}-ring...", flush=True)
                else:
                    print(f"[Substep {s+1}/{n_sub}] Repair failed at 3-ring; accepting displacement without patch repair.", flush=True)

        # If repair failed or was skipped for all rings, keep the displacement and continue
        # (do not revert to the previous node positions). This ensures that ablation-driven
        # motion is always applied, even when local patch repair cannot be constructed.
        if new_faces is None:
            print(f"[Substep {s+1}/{n_sub}] Accepted displacement (no patch replacement).", flush=True)
            continue

        n_old_assembly = len(assembly.mesh.facets)
        # Save PATO boundary fields before facet count changes (for accurate mapping)
        n_old_facets = len(obj.facet_index)
        old_pato_arrays = None
        patch_avg = None
        if hasattr(obj, "pato") and getattr(obj.pato, "flag", False):
            def _copy_pato_field(name):
                a = getattr(obj.pato, name, None)
                return np.asarray(a).copy() if a is not None and np.size(a) > 0 else None
            old_pato_arrays = {
                "q_conv_field": _copy_pato_field("q_conv_field"),
                "mdot_field": _copy_pato_field("mdot_field"),
                "v_n_field": _copy_pato_field("v_n_field"),
                "temperature": _copy_pato_field("temperature"),
                "hf_cond": _copy_pato_field("hf_cond"),
                "mDotVapor": _copy_pato_field("mDotVapor"),
                "mDotMelt": _copy_pato_field("mDotMelt"),
                "molten": _copy_pato_field("molten"),
            }
            # Mean field values over the patch being replaced; new patch facets will
            # be initialized to these so the remeshed region does not appear cold.
            patch_set = set(patch_faces)
            old_patch_obj_indices = np.array(
                [k for k in range(len(obj.facet_index)) if obj.facet_index[k] in patch_set],
                dtype=int,
            )
            if len(old_patch_obj_indices) > 0:
                patch_avg = {}
                for name, old_arr in old_pato_arrays.items():
                    if old_arr is not None and len(old_arr) > 0:
                        patch_avg[name] = float(np.mean(old_arr[old_patch_obj_indices]))

        new_to_old_obj, new_to_old_assembly, old_to_new = replace_patch(
            assembly.mesh, obj, patch_faces, new_faces
        )

        # Multi-object: update other objects' facet_index and sync their meshes
        for other in assembly.objects:
            if other is not obj:
                other.facet_index = old_to_new[other.facet_index]
                _sync_object_mesh_from_assembly(assembly.mesh, other)

        if len(assembly.mesh.facets) != n_old_assembly:
            map_assembly_fields_after_patch_replace(assembly, new_to_old_assembly)

        if len(obj.facet_index) != n_old_facets and hasattr(obj, "pato"):
            if old_pato_arrays is not None:
                map_pato_fields_after_patch_replace(
                    obj, old_pato_arrays, new_to_old_obj, patch_avg=patch_avg
                )
            else:
                n_new = len(obj.facet_index)
                for fname in ("q_conv_field", "mdot_field", "v_n_field", "temperature",
                             "hf_cond", "mDotVapor", "mDotMelt", "molten"):
                    if hasattr(obj.pato, fname):
                        setattr(obj.pato, fname, np.zeros(n_new, dtype=np.float64))
                if hasattr(obj.pato, "temperature"):
                    obj.temperature = np.asarray(obj.pato.temperature, dtype=np.float64)

        if not success:
            print(f"[Substep {s+1}/{n_sub}] Accepted patch with small triangles (after up to 3-ring).", flush=True)

    obj_disp = full_disp[obj.node_index]
    obj.mesh.nodes = assembly.mesh.nodes[obj.node_index]
    sync_surface_from_nodes(obj.mesh)
