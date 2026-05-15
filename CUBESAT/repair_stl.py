import trimesh
import os

def repair_stl(input_file, output_file):
    print(f"Loading {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    # Load the mesh
    mesh = trimesh.load(input_file)
    print(f"Original mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices.")

    # 1. Merge vertices (This is Blender's "Merge by Distance")
    # It snaps together points that are microscopically close
    mesh.merge_vertices()
    print("Merged duplicate/close vertices.")

    # 2. Remove degenerate faces
    # This specifically targets and deletes the 56 zero-area triangles
    mesh.remove_degenerate_faces()
    print("Removed degenerate (zero-area) faces.")

    # 3. Remove duplicate faces
    # Fixes the "dihedral angle 0.00" self-intersecting ghost triangles
    mesh.remove_duplicate_faces()
    print("Removed exactly overlapping duplicate faces.")

    # 4. Remove any vertices that are no longer attached to anything
    mesh.remove_unreferenced_vertices()

    # 5. Fix surface normals (ensures Gmsh knows what is 'inside' vs 'outside')
    mesh.fix_normals()

    print(f"Repaired mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices.")
    
    # Export the clean file
    mesh.export(output_file)
    print(f"--- SUCCESS: Repaired STL saved as '{output_file}' ---")

if __name__ == "__main__":
    repair_stl("cubesattest.stl", "cubesattest_repaired.stl")