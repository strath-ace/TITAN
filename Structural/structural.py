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
import numpy as np
from Structural.FENICS import FEniCS_FE_v08 as fenics
from Geometry import mesh
from Forces.forces import compute_inertial_forces

#assembly, fenics, options, i,
#current_assembly_index, frag_bool = False, save_displacement = False,
#save_vonMises = False

def run_FENICS(titan, options):

    #Flag to regenerate subdomain
#    regen_subdom = False
    
    #Initializing FENICS 
#    if titan.iter == 0:
#    fenics.x_nodes = current_assembly.mesh.vol_coords[:,0].copy()
#    fenics.y_nodes = current_assembly.mesh.vol_coords[:,1].copy()
#    fenics.z_nodes = current_assembly.mesh.vol_coords[:,2].copy()

#        create_fenics_nodes(fenics,assembly[current_assembly_index])
        
#        exit()

        # fenics.map_physical_volume = assembly[0].map_physical_volume
####        if options.FE_MPI: print ('Using MPI with %i cores'%(options.FE_MPI_cores))
####        regen_subdom = True
        
####    if frag_bool:
####        regen_subdom = True

#    save_subdomains = True if i == 0 else False
    
    #TEST
    regen_subdomains = True

    for assembly in titan.assembly:
        if len(assembly.objects) <= 1: continue

        inertial_forces = compute_inertial_forces(assembly,options)

        force_x = assembly.body_force.force_nodes[:,0].copy()
        force_y = assembly.body_force.force_nodes[:,1].copy()
        force_z = assembly.body_force.force_nodes[:,2].copy()

        force_x[assembly.objects[0].node_index] = 0
        force_y[assembly.objects[0].node_index] = 0
        force_z[assembly.objects[0].node_index] = 0

        forces = [force_x, force_y, force_z]

        num_surf_points = len(assembly.mesh.nodes)

        map_physical_volume = []

        E = 1E20
        for obj in assembly.objects:
            print(vars(obj.material))
            E = np.min([E,obj.material.youngModulus(obj.temperature)])

        #Young Modulus
        surf_displacement, surf_vM, stress_ratio_dict, surf_force, disp_arr, max_displacements, vM_arr = fenics.run_fenics(forces, num_surf_points,
                                                               map_physical_volume, assembly = assembly, options = options,
                                                               regen_subdomains = regen_subdomains, inertial_forces = inertial_forces, E = E)
        max_stress_ratio = -1e6 
        max_vm = -1e6
        for vol_id, vals in stress_ratio_dict.items():
            vm, ratio = vals['Max vm'], vals['Stress ratio']
            if ratio > max_stress_ratio:
                max_stress_ratio = ratio
                max_stress_ratio_vol_id = vol_id
                max_vm = vm
        
        #Update surface and volume coords
        mesh.update_surface_displacement(assembly.mesh, surf_displacement - assembly.mesh.surface_displacement)
        mesh.update_volume_displacement(assembly.mesh, disp_arr - assembly.mesh.volume_displacement)
        
        assembly.mesh.surface_displacement = surf_displacement
        assembly.mesh.volume_displacement = disp_arr
        assembly.mesh.volume_vonMises = vM_arr

        #Update the inertia matrix due to displacement
        assembly.compute_mass_properties()

        #disp_change_arr = disp_arr - fenics.disp_arr_prev
        #fenics.disp_arr_prev = disp_arr.copy()
        #disp_change_arr = disp_arr
        #Update the mesh with the new values





    # for it in range(len(assembly)):
    # VTK.write_surface_vtk_no_aero(assembly[current_assembly_index],i,it, output_folder=options.output_folder)#,obj, flag)
    # print ('Max surf vm',surf_vM.max())
    # VTK.write_surface_vtk(assembly[current_assembly_index],assembly[current_assembly_index].aerothermo,i,current_assembly_index, surf_displacement, surf_vM, surf_force, output_folder=options.output_folder)
    
    # VTK.write_surface_vtk_assembly(assembly,aerothermo,i,surf_displacement, surf_vM, surf_force, output_folder=options.output_folder)
    # VTK.write_surface_vtk_assembly(assembly,aerothermo,i, output_folder=options.output_folder)
    
    # print ('disp arr shape:',disp_arr.shape)
    # print ('disp change arr shape:',disp_change_arr.shape)
    # print ('mesh coords shape:', assembly[current_assembly_index].mesh.vol_coords.shape)
    #    assembly[current_assembly_index].update_displacement(disp_change_arr, _3D = True)

    
        #for obj, vol_bc_id in fenics.map_physical_volume.items():
        #    vol_id = vol_bc_id['Vol id']
        #    if vol_id == max_stress_ratio_vol_id:
        #        max_stress_ratio_obj = vol_bc_id['Name'].split('/')[-1]
        
        #print ('Maximum stress in: '+ str(max_stress_ratio_obj) + ' VM: ' + str(max_vm))
        #print ('---------', max_stress_ratio,'---------')
        #assembly[current_assembly_index].trajectory.stress_ratio = max_stress_ratio 
        #assembly[current_assembly_index].trajectory.stress_max_vm = max_vm 
        #assembly[current_assembly_index].trajectory.max_stress_obj = max_stress_ratio_obj.split('.')[0] 
        #assembly[current_assembly_index].trajectory.max_stress_ratio_vol_id = max_stress_ratio_vol_id
        #assembly[current_assembly_index].trajectory.max_x_disp = max_displacements[0]
        #assembly[current_assembly_index].trajectory.max_y_disp = max_displacements[1]
        #assembly[current_assembly_index].trajectory.max_z_disp = max_displacements[2]
        #if max_stress_ratio >= 0:
        #    fenics.obj_ids.remove(max_stress_ratio_vol_id)


