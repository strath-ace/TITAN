#!/usr/bin/env python3
# Mesh update utilities used via thermal.py; not intended for direct execution.
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def strip_foam_comments(text: str) -> str:
    """Remove OpenFOAM // and /* */ comments so list headers can be parsed."""
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def foam_list_payload(text: str):
    """Parse OpenFOAM 'length ( ... )' list; return (length, payload_string)."""
    text = strip_foam_comments(text)
    m = re.search(r"\n\s*(\d+)\s*\(\s*\n", text)
    if not m:
        m = re.search(r"\n\s*(\d+)\s*\(\s*", text)
    if not m:
        raise ValueError("Could not locate OpenFOAM list count followed by '('")
    n = int(m.group(1))
    payload = text[m.end():]
    return n, payload


def read_foam_points_ascii(path: Path) -> np.ndarray:
    n, payload = foam_list_payload(path.read_text(errors="ignore"))
    vecs = re.findall(r"\(\s*([eE0-9\+\-\.]+)\s+([eE0-9\+\-\.]+)\s+([eE0-9\+\-\.]+)\s*\)", payload)
    if len(vecs) < n:
        raise ValueError(f"Expected {n} points in {path}, found {len(vecs)}")
    return np.array([(float(a), float(b), float(c)) for a, b, c in vecs[:n]], dtype=float)


def read_foam_faces_ascii(path: Path):
    n, payload = foam_list_payload(path.read_text(errors="ignore"))
    face_matches = re.findall(r"\b(\d+)\s*\(\s*([0-9\s]+?)\s*\)", payload)
    if len(face_matches) < n:
        raise ValueError(f"Expected {n} faces in {path}, found {len(face_matches)}")
    faces = []
    for k_str, body in face_matches[:n]:
        faces.append([int(x) for x in body.split()])
    return faces


def patch_face_range(boundary_path: Path, patch_name: str):
    raw = strip_foam_comments(boundary_path.read_text(errors="ignore"))
    m = re.search(
        rf"{re.escape(patch_name)}\s*\{{.*?\bnFaces\s+(\d+)\s*;\s*startFace\s+(\d+)\s*;",
        raw, flags=re.DOTALL
    )
    if not m:
        raise ValueError(f"Patch '{patch_name}' not found in {boundary_path}")
    return int(m.group(2)), int(m.group(1))  # startFace, nFaces


def patch_point_ids_first_appearance(poly_dir: Path, patch_name: str):
    faces = read_foam_faces_ascii(poly_dir / "faces")
    startFace, nFaces = patch_face_range(poly_dir / "boundary", patch_name)
    patch_faces = faces[startFace:startFace + nFaces]

    seen = set()
    ordered = []
    for f in patch_faces:
        for pid in f:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
    return ordered


def map_nearest(src_xyz: np.ndarray, src_disp: np.ndarray, tgt_xyz: np.ndarray) -> np.ndarray:
    """Map displacement from source positions to target positions by nearest neighbor."""
    tree = cKDTree(src_xyz)
    _, idx = tree.query(tgt_xyz, k=1)
    return src_disp[idx]


def write_foam_points_ascii(path: Path, points: np.ndarray) -> None:
    """
    Write an OpenFOAM points file.  If the file already exists the FoamFile
    header is preserved; otherwise a generic header is written.
    """
    n = points.shape[0]
    header = None
    if path.exists():
        orig = path.read_text(errors="ignore")
        m = re.search(r"\n\s*\d+\s*\(\s*\n", orig)
        if m:
            header = orig[: m.start() + 1]

    if header is None:
        header = (
            "FoamFile\n{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            "    class       vectorField;\n"
            "    location    \"constant/polyMesh\";\n"
            "    object      points;\n"
            "}\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        f.write(header)
        f.write(f"\n{n}\n(\n")
        for p in points:
            f.write(f"({p[0]:.16e} {p[1]:.16e} {p[2]:.16e})\n")
        f.write(")\n")


def apply_displacement_to_mesh(
    case: Path, region: str, patch: str,
    mesh_nodes: np.ndarray, vertex_disp: np.ndarray,
    n_cores: int = 2,
) -> None:
    """
    Apply recession displacement to the root constant/<region>/polyMesh/points,
    then run decomposePar to regenerate processor meshes.
    """
    root_poly = case / "constant" / region / "polyMesh"
    pts = read_foam_points_ascii(root_poly / "points")
    pids = patch_point_ids_first_appearance(root_poly, patch)
    outer_pids, _ = split_patch_inner_outer(pts, pids, root_poly, patch)

    outer_xyz = pts[np.array(outer_pids, dtype=int)]
    outer_disp = map_nearest(mesh_nodes, vertex_disp, outer_xyz)

    for i, gpid in enumerate(outer_pids):
        pts[gpid] += outer_disp[i]

    write_foam_points_ascii(root_poly / "points", pts)
    print(f"Updated root constant/{region}/polyMesh/points "
          f"({len(outer_pids)} outer pts moved)", flush=True)

    # Remove old processor meshes so decomposePar creates fresh ones
    for p in range(n_cores):
        proc_const = case / f"processor{p}" / "constant"
        if proc_const.exists():
            shutil.rmtree(proc_const)

    run(["decomposePar", "-region", region, "-case", str(case)],
        cwd=case)
    print(f"decomposePar -region {region} complete", flush=True)


def print_mesh_quality(case, region="subMat1", time_value=None, n_substeps=None):
    """
    Run checkMesh and print a compact quality summary.
    Returns a dict with parsed quality metrics (or empty dict on failure).
    """
    case = Path(case).resolve()
    quality = {}
    try:
        result = subprocess.run(
            CONDA_PREAMBLE + ["checkMesh", "-region", region, "-case", str(case)],
            cwd=str(case),
            capture_output=True,
            text=True,
            timeout=120,
        )
        txt = (result.stdout or "") + (result.stderr or "")
    except Exception as e:
        msg = f"[Mesh quality] checkMesh failed: {e}"
        print(msg, flush=True)
        _append_mesh_quality_log(case, time_value, n_substeps, [msg])
        return quality

    keywords = [
        "cells:", "points:",
        "non-orthogonality", "max skewness",
        "minimum volume", "min volume",
        "Mesh OK", "Failed", "***Error",
    ]
    summary = []
    for line in txt.splitlines():
        stripped = line.strip()
        if any(kw.lower() in stripped.lower() for kw in keywords):
            summary.append(stripped)

    # Parse numeric metrics
    for line in txt.splitlines():
        low = line.lower()
        m = re.search(r"max non-orthogonality\D*([0-9.]+)", low)
        if m:
            quality["max_non_ortho"] = float(m.group(1))
        m = re.search(r"max skewness\D*([0-9.]+)", low)
        if m:
            quality["max_skewness"] = float(m.group(1))
        if "mesh ok" in low:
            quality["mesh_ok"] = True
        if "***" in line or "failed" in low:
            quality["mesh_ok"] = False

    if not summary:
        tail = [l.strip() for l in txt.splitlines() if l.strip()]
        summary = tail[-5:]

    summary = summary[:5]
    print("[Mesh quality]", flush=True)
    for s in summary:
        print(f"    {s}", flush=True)

    _append_mesh_quality_log(case, time_value, n_substeps, summary)
    return quality


def should_remesh(quality, options, time_value=None):
    """
    Decide whether a full remesh is needed based on checkMesh quality metrics.
    Thresholds can be set via options.pato attributes or use defaults.
    """
    max_ortho_thresh = getattr(options.pato, 'remesh_max_non_ortho', 70.0)
    max_skew_thresh = getattr(options.pato, 'remesh_max_skewness', 4.0)

    if not quality:
        return False

    if quality.get("mesh_ok") is False:
        print(f"[Remesh trigger] checkMesh reported failure at t={time_value}", flush=True)
        return True

    non_ortho = quality.get("max_non_ortho", 0)
    skewness = quality.get("max_skewness", 0)

    if non_ortho > max_ortho_thresh:
        print(f"[Remesh trigger] Non-orthogonality {non_ortho:.1f} > {max_ortho_thresh} at t={time_value}", flush=True)
        return True
    if skewness > max_skew_thresh:
        print(f"[Remesh trigger] Skewness {skewness:.2f} > {max_skew_thresh} at t={time_value}", flush=True)
        return True

    return False


def _append_mesh_quality_log(case: Path, time_value, n_substeps, summary_lines: list) -> None:
    """Append mesh quality summary to mesh_quality_log.txt in the case directory."""
    log_path = Path(case) / "mesh_quality_log.txt"
    header = "=== "
    if time_value is not None:
        header += f"t={time_value} "
    if n_substeps is not None:
        header += f"({n_substeps} substep{'s' if n_substeps != 1 else ''}) "
    header += "===\n"
    body = "\n".join(f"  {s}" for s in summary_lines) + "\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(body)
    except OSError as e:
        print(f"[Mesh quality] Could not append to {log_path}: {e}", flush=True)


def save_mesh_snapshot(case: Path, region: str, time_value: float) -> None:
    """
    Copy the root constant/<region>/polyMesh to mesh_evolution/<time>/polyMesh
    for ParaView visualization.  Also creates constant/polyMesh (required by
    ParaView's OpenFOAM reader), system/controlDict, and case.foam.
    """
    case = Path(case).resolve()
    src = case / "constant" / region / "polyMesh"
    if not src.exists():
        return
    mesh_evol = case / "mesh_evolution"

    # ParaView needs constant/polyMesh -- copy it on the first snapshot
    const_poly = mesh_evol / "constant" / "polyMesh"
    if not const_poly.exists():
        const_poly.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(const_poly))

    # Copy current mesh into time directory
    time_dir = mesh_evol / str(time_value) / "polyMesh"
    if time_dir.exists():
        shutil.rmtree(time_dir)
    time_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = time_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    (mesh_evol / "case.foam").touch()
    sys_dir = mesh_evol / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    main_control = case / "system" / "controlDict"
    ev_control = sys_dir / "controlDict"
    if main_control.exists() and not ev_control.exists():
        shutil.copy2(main_control, ev_control)
    print(f"Saved mesh snapshot: mesh_evolution/{time_value}/polyMesh", flush=True)


def split_patch_inner_outer(
    pts: np.ndarray, pids: list, poly_dir: Path, patch_name: str
) -> tuple:
    """
    Split a single boundary patch into inner and outer surfaces using
    trimesh connectivity analysis. Patch faces are extracted, triangulated,
    and split into connected components. The component whose centroid is
    farther from the overall patch centroid is labelled "outer".
    Returns (outer_pids, inner_pids).
    """
    all_faces = read_foam_faces_ascii(poly_dir / "faces")
    startFace, nFaces = patch_face_range(poly_dir / "boundary", patch_name)
    patch_faces = all_faces[startFace:startFace + nFaces]

    pid_arr = np.array(pids, dtype=int)
    global_to_local = {g: i for i, g in enumerate(pid_arr)}
    local_xyz = pts[pid_arr]

    tri_faces = []
    for f in patch_faces:
        local_f = [global_to_local[g] for g in f]
        for k in range(1, len(local_f) - 1):
            tri_faces.append([local_f[0], local_f[k], local_f[k + 1]])
    tri_faces = np.array(tri_faces, dtype=int)

    mesh = trimesh.Trimesh(vertices=local_xyz, faces=tri_faces, process=False)

    centroid = local_xyz.mean(axis=0)

    face_components = trimesh.graph.connected_components(mesh.face_adjacency)
    if len(face_components) == 1:
        print(f"Patch split: solid geometry (1 component), all {len(pids)} pts on outer surface", flush=True)
        return pids, []
    comp_verts = []
    for fc in face_components:
        vert_ids = set()
        for fi in fc:
            vert_ids.update(tri_faces[fi])
        comp_verts.append(vert_ids)

    mean_radii = []
    for vs in comp_verts:
        vs_arr = np.array(list(vs), dtype=int)
        dists = np.linalg.norm(local_xyz[vs_arr] - centroid, axis=1)
        mean_radii.append(float(np.mean(dists)))

    outer_idx = int(np.argmax(mean_radii))
    inner_idx = int(np.argmin(mean_radii))

    outer_local = comp_verts[outer_idx]
    inner_local = comp_verts[inner_idx]

    outer_pids = [pids[i] for i in sorted(outer_local)]
    inner_pids = [pids[i] for i in sorted(inner_local)]

    print(f"Patch split: {len(outer_pids)} outer pts, "
          f"{len(inner_pids)} inner pts, "
          f"{len(face_components)} components", flush=True)

    return outer_pids, inner_pids


def check_minimum_thickness(
    case: Path,
    region: str,
    patch: str,
    threshold: float = 0.5e-3,
    n_cores: int = 2,
) -> bool:
    """
    Measure the minimum wall thickness (outer-to-nearest-inner distance)
    and return True if the object is intact, False if punctured.
    """
    poly = case / "constant" / region / "polyMesh"
    pts = read_foam_points_ascii(poly / "points")
    pids = patch_point_ids_first_appearance(poly, patch)
    outer_pids, inner_pids = split_patch_inner_outer(pts, pids, poly, patch)

    if len(inner_pids) == 0:
        print("Solid geometry, skipping thickness check", flush=True)
        return True

    outer_xyz = pts[np.array(outer_pids, dtype=int)]
    inner_xyz = pts[np.array(inner_pids, dtype=int)]

    tree = cKDTree(inner_xyz)
    dists, _ = tree.query(outer_xyz, k=1)
    min_thickness = float(np.min(dists))

    print(f"Min wall thickness = {min_thickness*1e3:.4f} mm "
          f"(threshold = {threshold*1e3:.4f} mm)", flush=True)

    return min_thickness >= threshold


CONDA_PREAMBLE = ['conda', 'run', '-n', 'pato']


def run(cmd, cwd, use_conda=True):
    if use_conda:
        cmd = CONDA_PREAMBLE + cmd
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def run_one_step(
    case: Path,
    mesh_nodes: np.ndarray,
    vertex_disp: np.ndarray,
    region: str = "subMat1",
    patch: str = "top",
    n_cores: int = 2,
) -> None:
    """
    Update root mesh with recession displacement (outer surface only),
    then decomposePar to regenerate processor meshes.
    """
    apply_displacement_to_mesh(
        case, region, patch, mesh_nodes, vertex_disp, n_cores=n_cores,
    )


def run_mesh_update(
    case,
    mesh_nodes: np.ndarray,
    vertex_disp: np.ndarray,
    region: str = "subMat1",
    patch: str = "top",
    steps: int = 1,
    check_thickness: bool = True,
    thickness_threshold: float = 0.5e-3,
    n_cores: int = 2,
) -> bool:
    """
    Run mesh update loop. Returns True if intact, False if demised (thickness check failed).
    """
    case = Path(case).resolve()
    for s in range(steps):
        print(f"\n=== Mesh update step {s+1}/{steps} ===", flush=True)
        run_one_step(
            case=case,
            mesh_nodes=mesh_nodes,
            vertex_disp=vertex_disp,
            region=region,
            patch=patch,
            n_cores=n_cores,
        )
        if check_thickness and not check_minimum_thickness(
            case, region, patch, threshold=thickness_threshold,
            n_cores=n_cores,
        ):
            print("\n**********\nOBJECT HAS DEMISED. SIMULATION ENDED\n**********", flush=True)
            return False
    print("\nDone.", flush=True)
    return True


def archive_processor_VTK_to_history(case: Path, time_value=None) -> None:
    """
    Copy processor*/VTK/ files to VTK_history/ before remeshing destroys
    the processor directories.
    """
    case = Path(case).resolve()
    history = case / "VTK_history"
    history.mkdir(parents=True, exist_ok=True)

    for proc_dir in sorted(case.glob("processor*")):
        vtk_dir = proc_dir / "VTK"
        if not vtk_dir.exists():
            continue
        dest = history / proc_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for item in vtk_dir.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    tag = f"_t{time_value}" if time_value is not None else ""
    print(f"[VTK archive] Saved processor VTK to VTK_history/{tag}", flush=True)
