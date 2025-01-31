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
from Geometry import mesh

def compute_thermal_tetra(titan, options):
  
    dt = options.dynamics.time_step

    for assembly in titan.assembly:
        if assembly.ablation_mode != 'tetra': continue
        Tref = assembly.freestream.temperature

        #array that will contain facets and keys to delete tetras
        delete_array = []
        
        for obj in assembly.objects:

            facet_area = np.linalg.norm(assembly.mesh.facet_normal[obj.facet_index], ord = 2, axis = 1)
            heatflux = assembly.aerothermo.heatflux[obj.facet_index]
            
            #Properties for each facet
            Qin = heatflux*facet_area
            temperature = assembly.aerothermo.temperature[obj.facet_index]
            cp  = obj.material.specificHeatCapacity(temperature)
            emissivity = obj.material.emissivity(temperature)

            # Estimating the radiation heat-flux
            Qrad = 5.670373e-8*emissivity*(temperature**4 - Tref**4)*facet_area
            #TODO missing plasma radiation

            # Retrieve key to map surf to tetra
            key = np.round(assembly.mesh.facet_COG[obj.facet_index],5).astype(str)
            key = np.char.add(np.char.add(key[:,0],key[:,1]),key[:,2])

            # Retrieve tetras and respective properties
            tetra_array    = np.array([assembly.mesh.index_surf_tetra[k][0] for k in key])

            tag_id         = assembly.mesh.vol_tag == obj.id
            tetra_density  = assembly.mesh.vol_density[tetra_array]
            tetra_vol      = assembly.mesh.vol_volume[tetra_array]
            tetra_T        = assembly.mesh.vol_T[tetra_array]
            tetra_heatflux = np.zeros(len(assembly.mesh.vol_elements))
            tetra_cp       = obj.material.specificHeatCapacity(tetra_T)
            tetra_mass     = tetra_density * tetra_vol

            np.add.at(tetra_heatflux, tetra_array, Qin-Qrad)

            #Compute the heatflux that goes in for each tetra with faces at the surface
            tetra_heatflux = tetra_heatflux[tetra_array]

            # Computing temperature change
            dT = tetra_heatflux*dt/(tetra_mass*cp)
            dm = np.zeros(len(tetra_T))

            for index in range(len(tetra_array)):
                if tetra_T[index]+dT[index] > obj.material.meltingTemperature:
                    dT_melt = obj.material.meltingTemperature - tetra_T[index]
                    melt_Q = (tetra_mass[index]*tetra_cp[index])*(dT[index]-dT_melt)
                    dm[index] = -melt_Q/(obj.material.meltingHeat)
                    dT[index] = dT_melt

            #If the mass goes negative, we set it to 0. This means the tetra has ablated
            new_mass = tetra_mass + dm           
            new_mass[new_mass < 0] = 0

            assembly.mesh.vol_T[tetra_array] += dT

            assembly.mesh.vol_density[tetra_array] *= new_mass/tetra_mass 

            #The are some densities that are NaN whe using multiple objects, need to check why
            assembly.mesh.vol_density[np.isnan(assembly.mesh.vol_density)] = 0

            index_delete = np.where(assembly.mesh.vol_density[tetra_array]<=0)[0]

            if len(index_delete) != 0:
                for index in index_delete:
                    delete_array.append([index, tetra_array[index]])
        
        if delete_array:
            mesh.remove_ablated_elements(assembly, delete_array)

        #Map the tetra temperature to surface mesh
        COG = np.round(assembly.mesh.facet_COG,5).astype(str)
        COG = np.char.add(np.char.add(COG[:,0],COG[:,1]),COG[:,2])

        #Limit Tetras temperature so it does not go negative due to small mass
        assembly.mesh.vol_T[assembly.mesh.vol_T<273] = 273

        for index, COG in enumerate(COG):
            assembly.aerothermo.temperature[index] = assembly.mesh.vol_T[assembly.mesh.index_surf_tetra[str(COG)][0]]

        assembly.compute_mass_properties()


    #Removing assembly with no more tetra elements to prevent problems

    index = []

    for i,assembly in enumerate(titan.assembly):
        if len(assembly.mesh.nodes) == 0:
            index.append(i)

    titan.assembly = list(np.delete(titan.assembly,index))

    return 

def compute_thermal_0D(titan, options):

    dt = options.dynamics.time_step
    Tref = 273

    for assembly in titan.assembly:
        if assembly.ablation_mode != '0d': continue

        for obj in assembly.objects:

            facet_area = np.linalg.norm(obj.mesh.facet_normal, ord = 2, axis = 1)
            heatflux = assembly.aerothermo.heatflux[obj.facet_index]
            Qin = np.sum(heatflux*facet_area)
            
            cp  = obj.material.specificHeatCapacity(obj.temperature)
            emissivity = obj.material.emissivity(obj.temperature)

            Atot = np.sum(facet_area)

            # Estimating the radiation heat-flux
            Qrad = 5.670373e-8*emissivity*(obj.temperature**4 - Tref**4)*Atot

            # Computing temperature change
            dT = (Qin-Qrad)*dt/(obj.mass*cp)

            if obj.temperature+dT > obj.material.meltingTemperature:
                dT_melt = obj.material.meltingTemperature - obj.temperature
                melt_Q = (obj.mass*cp)*(dT-dT_melt)
                dm = -melt_Q/(obj.material.meltingHeat)
                dT = dT_melt
            else:
                dm = 0

            new_mass = obj.mass + dm
            new_T = obj.temperature + dT

            obj.material.density *= new_mass/obj.mass
            obj.mass = new_mass
            obj.temperature = new_T

            if obj.material.density < 0:
                obj.material.density = 0
                obj.mass = 0
            
            assembly.mesh.vol_density[assembly.mesh.vol_tag == obj.id] = obj.material.density
            assembly.aerothermo.temperature[obj.facet_index] = obj.temperature

        assembly.compute_mass_properties()

    return 