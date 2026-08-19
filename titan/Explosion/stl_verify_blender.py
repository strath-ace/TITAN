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
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""stl_verify_blender module."""
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd

try:
    import bpy, bmesh, bpy_extras
except ImportError:
    print("Module 'bpy' could not be imported. This probably means you are not using Blender to run this script.")
    #sys.exit(1)

def output_stats(stats_dict,filepath):
    """Documentation for the function.
:param stats_dict: Dictionary of stats.
:type stats_dict: dict
:param filepath: Path to the relevant file.
:type filepath: str"""
    stats_df = pd.DataFrame.from_dict(stats_dict)
    do_header = True
    if pathlib.Path(filepath).resolve().exists(): do_header = False
    stats_df.to_csv(str(pathlib.Path(filepath).resolve()),
                    mode='a',header=do_header, index=False)

def mesh_verify_selected(fix=True, merge_dist=0.0):
    """Documentation for the function.
:param fix: Value for fix.
:type fix: Any
:param merge_dist: Value for merge dist.
:type merge_dist: Any
:return: Return value.
:rtype: Any"""
    ## Delete loose, fill holes and check normals
    is_manifold = True
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=True)

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.select_all(action='DESELECT')

    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_non_manifold(extend=False)

    if bpy.context.active_object.data.total_vert_sel>0:
        is_manifold = False
        if fix:
            try:
                bpy.ops.mesh.fill(use_beauty=True)

                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.normals_make_consistent(inside=False)
            except: 
                sys.stderr.write('Error filling holes')
    if merge_dist>0:
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.dissolve_degenerate(threshold=merge_dist)
        bpy.ops.mesh.remove_doubles(threshold=merge_dist)
        is_manifold = mesh_verify_selected(merge_dist=0)
    bpy.ops.object.mode_set(mode='OBJECT')
    return is_manifold

prog_name = "Verify stls "

if __name__=='__main__':
    if '--' not in sys.argv:
        print(prog_name + "No '--' found in command line arguments. '--' is needed to pass arguments to this script.")
        sys.exit(1)

    arguments = sys.argv[sys.argv.index("--")+1:]
    parser = argparse.ArgumentParser(description="Verify an stl using blenders cleanup tools")
    parser.add_argument('--file',type=str,help='Path to target .stl object')
    parser.add_argument('--stats', type=str,default='', help='Path to stats file, leave blank to disable stat collection')
    parser.add_argument('-d', type=float,default=2810, help='Density for statistics computation')
    parser.add_argument('-o', action=argparse.BooleanOptionalAction, help='Overwrite file with cleaned mesh')
    parser.add_argument('-w', type=float,default=0.0, help='Wall thickness if thin-walled')
    parser.add_argument('-m', type=float,default=0.0, help='Merge distance to remove degenerate geometry')

    args = parser.parse_args(arguments)
    path = args.file
    stats_file = args.stats
    density = args.d
    wall_thickness = args.w
    merge_dist = args.m
    ## Collect important references
    obj_name = path.split('/')[-1].split('.')[0]
    scene = bpy.context.scene
    stats = {'name' : [obj_name], 'volume' : [np.nan], 'surf_area' : [np.nan], 'mass' : [np.nan], 'area_mass_ratio' : [np.nan], 'reference_length' : [np.nan]}
    sys.stderr.write(obj_name)
    ## Clean up blender scene (kill the default cube)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    ## Add our object 
    bpy.ops.wm.stl_import(filepath=path)

    if wall_thickness:
        mesh_verify_selected(fix=False)
        bpy.ops.object.modifier_add(type='SOLIDIFY')
            
        bpy.context.object.modifiers['Solidify'].thickness = wall_thickness
        bpy.context.object.modifiers['Solidify'].solidify_mode = 'NON_MANIFOLD'
        bpy.context.object.modifiers['Solidify'].nonmanifold_boundary_mode = 'ROUND'
        bpy.context.object.modifiers['Solidify'].nonmanifold_merge_threshold = 0.0001
        bpy.ops.object.modifier_apply(modifier='Solidify')


    if not mesh_verify_selected(fix=True, merge_dist=merge_dist):
        sys.stderr.write('Object {} is non manifold!\n'.format(path))
        if not mesh_verify_selected(fix=True, merge_dist=0): 
            sys.stderr.write('Failed to recover Object {}\n'.format(path))
            exit(1)

    bm = bmesh.new()
    bm.from_mesh(bpy.data.objects[obj_name].data)
    volume = bm.calc_volume()
    surf_area = sum([f.calc_area() for f in bm.faces])
    bm.free()

    if not volume>5e-7:
        sys.stderr.write('Object {} is not well behaved! (vol={})\n'.format(obj_name, volume))
        exit(1)
        if not stats_file=='': output_stats(stats, stats_file)
        

    bb_top = np.array(bpy.data.objects[obj_name].bound_box[0][:])
    bb_bot = np.array(bpy.data.objects[obj_name].bound_box[6][:])
    l_diag = np.linalg.norm(bb_top-bb_bot)

    if not stats_file=='':
        stats['volume'] = [volume]
        stats['surf_area'] = [surf_area]
        stats['mass'] = [volume * density]
        stats['area_mass_ratio'] = [surf_area / (volume * density)]
        stats['reference_length'] = [l_diag]
        output_stats(stats, stats_file)

    bpy.ops.wm.stl_export(filepath=path)
