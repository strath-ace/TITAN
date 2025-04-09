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
from scipy import integrate
import pandas as pd
import os
from Thermal import pato
from Aerothermo import aerothermo as Aerothermo
from scipy.spatial.transform import Rotation as Rot
import vg
import sys
from Geometry.tetra import inertia_tetra
import requests

def compute_thermal(titan, options):

    if options.thermal.ablation_mode == "tetra":
        compute_thermal_tetra(titan = titan, options = options)
    elif options.thermal.ablation_mode == "0d":
        compute_thermal_0D(titan = titan, options = options)
    elif options.thermal.ablation_mode == "pato":
        compute_thermal_PATO(titan = titan, options = options)
    else:
        raise ValueError("Ablation Mode can only be 0D, Tetra or PATO")

def compute_thermal_0D(titan, options):

    dt = titan.delta_t
    Tref = 273

    for assembly in titan.assembly:
        #if assembly.ablation_mode != '0d': continue

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
            #obj.pato.temperature[:] = obj.temperature

            if obj.material.density < 0:
                obj.material.density = 0
                obj.mass = 0
            
            assembly.mesh.vol_density[assembly.mesh.vol_tag == obj.id] = obj.material.density
            assembly.aerothermo.temperature[obj.facet_index] = obj.temperature
    
            #obj.photons = compute_radiance(obj.temperature, Atot, emissivity)

        assembly.compute_mass_properties()

    return

def compute_thermal_tetra(titan, options):
  
    dt = titan.delta_t

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

            #facet_COG of surface facets for the current object
            key = np.round(assembly.mesh.facet_COG[obj.facet_index],5).astype(str)
            key = np.char.add(np.char.add(key[:,0],key[:,1]),key[:,2])

            # Retrieve tetras corresponding to object surface facets
            tetra_array    = np.array([assembly.mesh.index_surf_tetra[k][0] for k in key])
            ##print('tetra_array:', tetra_array)
            ##print('tetra_array:',np.shape(tetra_array))

            ##print('tetra_array:', assembly.mesh.vol_elements[tetra_array])
            tetras_cat = assembly.mesh.vol_elements[tetra_array]
            c0 = tetras_cat[:,0]
            c1 = tetras_cat[:,1]
            c2 = tetras_cat[:,2]
            c3 = tetras_cat[:,3]
            
            ##for i in range(len(tetras_cat)):
            ##    print('tetra ', i, ' coord:', assembly.mesh.vol_coords[c0[i]], assembly.mesh.vol_coords[c1[i]], assembly.mesh.vol_coords[c2[i]], assembly.mesh.vol_coords[c3[i]])
            #exit()

            tag_id         = assembly.mesh.vol_tag == obj.id
            tetra_density  = assembly.mesh.vol_density[tetra_array]
            tetra_vol      = assembly.mesh.vol_volume[tetra_array]
            tetra_T        = assembly.mesh.vol_T[tetra_array]
            tetra_heatflux = np.zeros(len(assembly.mesh.vol_elements))
            tetra_cp       = obj.material.specificHeatCapacity(tetra_T)
            tetra_mass     = tetra_density * tetra_vol

            #tetra_heatflux has size of all volume tetras in the assembly
            #assign heatflux only to tetras in tetra_array, equivalent to tetra_heatflux[tetra_array_indices] += Qin-Qrad
            #if tetra show up more than once (example of corner tetra, connected to more than 1 facet), this contribution is
            #added more than once too
            np.add.at(tetra_heatflux, tetra_array, Qin-Qrad)

            #Compute the heatflux that goes in for each tetra with faces at the surface
            tetra_heatflux = tetra_heatflux[tetra_array]

            # Computing temperature change
            dT = tetra_heatflux*dt/(tetra_mass*cp)
            dm = np.zeros(len(tetra_T))

            for index in range(len(tetra_array)):
                if tetra_T[index]+dT[index] > obj.material.meltingTemperature:
                    #print('tetra_T[index]+dT[index]:', tetra_T[index]+dT[index])
                    #print('obj.material.meltingTemperature:', obj.material.meltingTemperature)
                    #print('Tetra should demise')
                    dT_melt = obj.material.meltingTemperature - tetra_T[index]
                    melt_Q = (tetra_mass[index]*tetra_cp[index])*(dT[index]-dT_melt)
                    dm[index] = -melt_Q/(obj.material.meltingHeat)
                    dT[index] = dT_melt

            #If the mass goes negative, we set it to 0. This means the tetra has ablated
            new_mass = tetra_mass + dm           
            new_mass[new_mass < 0] = 0
            #print('new_mass:', new_mass)

            assembly.mesh.vol_T[tetra_array] += dT

            assembly.mesh.vol_density[tetra_array] *= new_mass/tetra_mass 

            ##print('vol_density:', assembly.mesh.vol_density[tetra_array])

            #The are some densities that are NaN whe using multiple objects, need to check why
            assembly.mesh.vol_density[np.isnan(assembly.mesh.vol_density)] = 0

            #to delete: index of surface tetras to delete
            index_delete = np.where(assembly.mesh.vol_density[tetra_array]<=0)[0]

            ##print('index_delete:', index_delete)
            #index_delete = [10, 15, 3, 11, 1, 14]
            #print('index_delete:', index_delete)

            if len(index_delete) != 0:
                for index in index_delete:
                    #index - index of surface tetra to delete
                    #tetra_array[index] - tetra to delete
                    ##print('index:', index)
                    ##print('tetra_array[index]:', tetra_array[index])
                    delete_array.append([index, tetra_array[index]])

            #print('delete_array:', delete_array)        
            
        print('N tetras assembly:', len(assembly.mesh.vol_elements))
        if delete_array:
            #print('0 N facets assembly:', len(assembly.mesh.v0))
            #print('N tetras assembly:', len(assembly.mesh.vol_elements))
            print('Removing ablated elements')
            mesh.remove_ablated_elements(assembly, delete_array)
            #print('N facets assembly:', len(assembly.mesh.facets))
            #print('N tetras assembly:', len(assembly.mesh.vol_elements))
        
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

def compute_thermal_PATO(titan, options):

    for assembly in titan.assembly: 

        if options.pato.conduction_flag:
            pato.compute_heat_conduction(assembly)

        Tinf = assembly.freestream.temperature        

        for obj in assembly.objects:

            if obj.pato.flag: 
                hf = obj.pato.hf_cond + assembly.aerothermo.heatflux[obj.facet_index]
                he = assembly.aerothermo.he[obj.facet_index]
                hw = assembly.aerothermo.hw[obj.facet_index]
                rhoe = assembly.aerothermo.rhoe[obj.facet_index]
                ue = assembly.aerothermo.ue[obj.facet_index]
                pw = assembly.aerothermo.pressure[obj.facet_index]
                pato.compute_thermal(obj, titan.time, titan.iter, options, hf, Tinf)
                assembly.aerothermo.temperature[obj.facet_index] = obj.temperature

                if options.pato.Ta_bc == 'ablation':
                    assembly.mDotVapor[obj.facet_index] = obj.pato.mDotVapor
                    assembly.mDotMelt[obj.facet_index] = obj.pato.mDotMelt
                

        if options.pato.Ta_bc == 'ablation':

            #if there is mass loss, update assembly mass and inertia properties
            if any(obj.density_ratio < 1 for obj in assembly.objects):

                assembly.mesh.vol_density[assembly.mesh.vol_tag == obj.id] = obj.material.density

                coords = assembly.mesh.vol_coords
                elements = assembly.mesh.vol_elements
                density = assembly.mesh.vol_density
                vol = assembly.mesh.vol_volume
        
                #Computes the mass of every single tetrahedral
                assembly.mesh.vol_mass  = vol*density
                assembly.mass = np.sum(assembly.mesh.vol_mass)
        
                #Computes the Center of Mass
                if assembly.mass <= 0:
                    assembly.COG = np.array([0,0,0])
                else:
                    assembly.COG = np.sum(0.25*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*assembly.mesh.vol_mass[:,None], axis = 0)/assembly.mass
        
                #Computes the inertia matrix
                assembly.inertia = inertia_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], vol, assembly.COG, density)

            #Update gas mass fractions to include vaporized material from ablation

            assembly.mVapor = assembly.mDotVapor*options.dynamics.time_step
            assembly.mMelt  = assembly.mDotMelt*options.dynamics.time_step

    return

def compute_black_body_emissions(titan, options, q = []):

    print('Computing polar emissions ...')

    h = 6.62607015e-34 # m2.kg.s-1        planck constant
    c = 3e8            # m.s-1           light speed in vaccum
    k = 1.380649e-23   # m2.kg.s-2.K-1 boltzmann constant

    phi_min = options.radiation.phi_min
    phi_max = options.radiation.phi_max
    phi_n_values = options.radiation.phi_n_values
    theta_min = 0 #options.radiation.theta_min
    theta_max = 360*np.pi/180.0 #options.radiation.theta_max
    theta_n_values = 36 #options.radiation.theta_n_values

    phi   = np.linspace(phi_min, phi_max, phi_n_values)
    theta = np.linspace(theta_min, theta_max, theta_n_values)

    wavelength_min = options.radiation.wavelength_min
    wavelength_max = options.radiation.wavelength_max

    for theta_i in range(len(theta)):
        for phi_i in range(len(phi)):
            for assembly_id, assembly in enumerate(titan.assembly):
                for obj in assembly.objects:
                    emissivity_obj = obj.material.emissivity(obj.temperature)
                    assembly.emissivity[obj.facet_index] = emissivity_obj
                    assembly.emissivity[obj.facet_index] = np.clip(assembly.emissivity[obj.facet_index], 0, 1)

                viewpoint = np.array([np.sin(theta[theta_i])*np.cos(phi[phi_i]), np.sin(theta[theta_i])*np.sin(phi[phi_i]), np.cos(theta[theta_i])])

                if len(q) == 0: quat = assembly.quaternion_prev
                else: 
                    quat = q[assembly_id]

                viewpoint = -Rot.from_quat(quat).inv().apply(viewpoint)/np.linalg.norm(viewpoint)

                index = Aerothermo.ray_trace(assembly, viewpoint)
                
                facet_area = assembly.mesh.facet_area

                vec1 = -viewpoint
                vec2 = np.array(assembly.mesh.facet_normal)
                angle = vg.angle(vec1, vec2) #degrees
                cosine = np.cos(angle*np.pi/180)

                temperature = assembly.aerothermo.temperature                

                planck_integral = np.zeros(len(assembly.mesh.facets))

                for facet in range(len(assembly.mesh.facets)):
                    planck_integral[facet] = integrate_planck(wavelength_min, wavelength_max, temperature[facet])

                assembly.emissive_power[:] = 0
                assembly.emissive_power[index] = assembly.emissivity[index]*planck_integral[index]*cosine[index]*facet_area[index] #Units: W.sr-1 (after integrating over wavelength range)

                assembly.total_emissive_power = np.sum(assembly.emissive_power) # units W.sr−1

                d = {'Phi': [phi[phi_i]*180/np.pi],
                     'Theta': [theta[theta_i]*180/np.pi],
                     'Spectral directional emissive power': [assembly.total_emissive_power],

                    }

                df = pd.DataFrame(data=d)

                df.to_csv(options.output_folder + '/Data/'+ 'thermal_signature_'+str(titan.iter)+'_'+str(assembly.id)+'.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/'+ 'thermal_signature_'+str(titan.iter)+'_'+str(assembly.id)+'.csv'), index = False)

def integrate_planck(lambd_min, lambd_max, T):

    integral, error = integrate.quad(black_body, lambd_min, lambd_max,args=(T), epsabs=1.0e-4, epsrel=1.0e-4 )

    return integral

def black_body(wavelength, T):

    h = 6.62607015e-34 # m2.kg.s-1        planck constant
    c = 3e8            # m.s-1           light speed in vaccum
    k = 1.380649e-23   # m2.kg.s-2.K-1 boltzmann constant

    exp = np.exp((h*c)/(k*wavelength*T))

    #b = (2*c/pow(wavelength,4)) *(1/(exp-1)) #units: photons*m-2*m-1*s-1*sr-1

    b = ((2*h*c*c)/(np.power(wavelength, 5))) * (1/(exp-1)) # units W.sr−1.m−3 #Radiance in terms of wavelength

    return b

def compute_black_body_spectral_emissions(assembly, wavelength):

    h = 6.62607015e-34 # m2.kg.s-1        planck constant
    c = 3e8            # m.s-1           light speed in vaccum
    k = 1.380649e-23   # m2.kg.s-2.K-1 boltzmann constant

    #index = assembly.index_blackbody
    #angle = assembly.angle_blackbody[index]

    #cosine = np.cos(angle*np.pi/180)

    index = np.arange(len(assembly.aerothermo.temperature))

    temperature = assembly.aerothermo.temperature[index]

    print('max T:', max(temperature))

    for obj in assembly.objects:
        emissivity_obj = obj.material.emissivity(obj.temperature)
        assembly.emissivity[obj.facet_index] = emissivity_obj
        assembly.emissivity[obj.facet_index] = np.clip(assembly.emissivity[obj.facet_index], 0, 1)
    
    emissivity = assembly.emissivity[index]
    facet_area = assembly.mesh.facet_area[index]

    emissions = np.zeros(len(wavelength))
    distribution = np.zeros(len(assembly.mesh.facets))

    for wavelength_i in range(len(wavelength)):

        lamb = wavelength[wavelength_i]

        exp = np.exp((h*c)/(k*lamb*temperature))   

        b = (((2*c)/(np.power(lamb, 4))) * (1/(exp-1)))*emissivity # units photons/[s*m2*m*sr]  #Spectral radiance in terms of wavelength

        #seen_b = b*cosine

        #distribution[index] += seen_b

        #weighing facet contribution by its area
        #facet_area = assembly.mesh.facet_area[index]
        #weighted_b = seen_b*facet_area # units photons/[s*m*sr]

        #sum contributions of all facets
        #emissions[wavelength_i] += np.sum(weighted_b) #photons/[s*m*sr]

        #b = b * facet_area


        #emissions[wavelength_i] += np.sum(b)/np.sum(assembly.mesh.facet_area) #photons/[s*m2*m*sr]

        emissions[wavelength_i] += np.sum(b)/len(assembly.mesh.facet_area[index]) #photons/[s*m2*m*sr]

    return emissions, distribution

def compute_particle_spectral_emissions_AlI(assembly, wavelength, wavelength_index):

    #index = assembly.index_atomic
    #angle = assembly.angle_atomic[index]

    index = assembly.aero_index #np.arange(len(assembly.aerothermo.temperature))

    h = 6.62607015e-34 # m2.kg.s-1        planck constant
    c = 3e8            # m.s-1           light speed in vaccum
    k = 1.380649e-23   # m2.kg.s-2.K-1 boltzmann constant

    Na = 6.023e23      # 1/mol Avogadro number

    A_O = [4.99e+07, 9.85e+07] #s-1
    E_O = [25347.756, 25347.756] #cm-1
    g_O = [2, 2]

    molarMass_Al = 26.98*1e-3 #kg/mol

    print('\nComputing spectral particle emissions Al I ...')

    #cosine = np.cos(angle*np.pi/180)

    for obj in assembly.objects:    
        assembly.LOS[obj.facet_index] = obj.LOS
    
    facet_area   = assembly.mesh.facet_area[index]
    temperature  = assembly.aerothermo.Te[index]
    #temperature = assembly.aerothermo.temperature[index]
    rhoe_i       = assembly.aerothermo.rhoe_i[index]
    LOS          = assembly.LOS[index]
    mMelt        = assembly.mMelt[index]

    #print('Al:', rhoe_i[:,-1])
    

    # ntot for Al
    #ntot = rhoe_i[:,-1]*Na/molarMass_Al
    #ntot = 1e13 #rhoe_i*Na/molarMass_Al

    ntot = np.where(mMelt == 0, 0, 7e16)

    #print('ntot:', ntot)
    #print('mMelt:', np.shape(mMelt))
    #print('ntot:', np.shape(ntot))
    #exit()

    emissions = np.zeros(len(wavelength))
    distribution = np.zeros(len(assembly.mesh.facets))

    max_index = np.argmax(assembly.aerothermo.heatflux[index])

    heatflux = assembly.aerothermo.heatflux[index]

    print('\nData for maximum heat flux point:')
    print('q:', heatflux[max_index])
    print('Te:', temperature[max_index])
    #print('ntot:', ntot[max_index])
    print('LOS:', LOS[max_index])


    wavelength_i = wavelength_index

    lamb = wavelength[wavelength_i]
    A    = A_O[wavelength_i]
    E    = E_O[wavelength_i]
    g    = g_O[wavelength_i]

    ZZ = partition('Al I', temperature)

    exp = np.exp(-h*c*E*100/(k*temperature))

    nn = g * exp * ntot / ZZ


    intensity = (h * c * A * nn / lamb)/(4*np.pi) # W/m3-sr

    #print('h:', h)
    #print('c:', c)
    print('A:', A)
    print('E:', E)
    print('g:', g)
    print('nn:', nn[max_index])
    #print('lamb:', lamb)
    #print('\ntemperature:', temperature)
    #print('Z:', ZZ)
    #print('exp:', exp)
    
    print('intensity:', intensity[max_index])

    intensity = intensity * LOS #W/m2-sr

    print('intensity*LOS:', intensity[max_index])

    

    #grey-body radiation seen from viewpoint
    #seen_intensity = intensity*cosine

    #distribution[index] += seen_intensity

    #weighing facet contribution by its area

    weighted_seen_intensity = intensity*facet_area # [W/sr]

    #sum contributions of all facets
    emissions[wavelength_i] += np.sum(weighted_seen_intensity)

    print('emissions[wavelength_i]:',emissions[wavelength_i])

    #emissions[wavelength_i] += np.sum(intensity)/len(facet_area)#W/m2-sr

    #intensity = intensity*facet_area

    #emissions[wavelength_i] += np.sum(intensity)/np.sum(assembly.mesh.facet_area) # W/m2-sr

    #emissions[wavelength_i] += np.sum(intensity)/len(assembly.mesh.facet_area) # W/m2-sr
    
    return emissions, distribution

def compute_particle_spectral_emissions_OI(assembly, wavelength):

    #index = assembly.index_atomic
    #angle = assembly.angle_atomic[index]

    index = assembly.aero_index #np.arange(len(assembly.aerothermo.temperature))

    max_index = np.argmax(assembly.aerothermo.heatflux[index])

    h = 6.62607015e-34 # m2.kg.s-1        planck constant
    c = 3e8            # m.s-1           light speed in vaccum
    k = 1.380649e-23   # m2.kg.s-2.K-1 boltzmann constant

    Na = 6.023e23      # 1/mol Avogadro number

    A_O = [3.69e+07, 3.69e+07, 3.69e+07] #s-1
    E_O = [86631.454, 86627.778, 86625.757] #cm-1
    g_O = [7, 5, 3]

    molarMass_O = 16*1e-3 #kg/mol

    print('\nComputing spectral particle emissions O I ...')

    #cosine = np.cos(angle*np.pi/180)
    
    for obj in assembly.objects:
        assembly.LOS[obj.facet_index] = obj.LOS
    
    facet_area = assembly.mesh.facet_area[index]
    temperature = assembly.aerothermo.Te[index]
    rhoe_i = assembly.aerothermo.rhoe_i[index]
    LOS = assembly.LOS[index]

    # ntot for O
    ntot = rhoe_i[:,1]*Na/molarMass_O

    emissions = np.zeros(len(wavelength))
    distribution = np.zeros(len(assembly.mesh.facets))

    heatflux = assembly.aerothermo.heatflux[index]

    print('\nData for maximum heat flux point:')
    print('q:', heatflux[max_index])
    print('Te:', temperature[max_index])
    print('rhoe_i:', rhoe_i[max_index])
    print('ntot:', ntot[max_index])
    print('LOS:', LOS[max_index])
    

    for wavelength_i in range(len(wavelength)):

        lamb = wavelength[wavelength_i]
        A    = A_O[wavelength_i]
        E    = E_O[wavelength_i]
        g    = g_O[wavelength_i]

        ZZ = partition('O I', temperature)

        exp = np.exp(-h*c*E*100/(k*temperature))

        nn = g * exp * ntot / ZZ

        intensity = (h * c * A * nn / lamb)/(4*np.pi) # W/m3-sr

        intensity = intensity * LOS # W/m2-sr

        print('wavelength:', lamb)
        print('intensity:', intensity[max_index])

        #grey-body radiation seen from viewpoint
        #seen_intensity = intensity*cosine

        #distribution[index] += seen_intensity

        #weighing facet contribution by its area
        weighted_seen_intensity = intensity*facet_area # [W/sr]

        #sum contributions of all facets
        emissions[wavelength_i] += np.sum(weighted_seen_intensity)

        #intensity = intensity*facet_area

        #emissions[wavelength_i] += np.sum(intensity)/np.sum(assembly.mesh.facet_area) # W/m2-sr
        #emissions[wavelength_i] += np.sum(intensity)/len(assembly.mesh.facet_area[index]) # W/m2-sr

    return emissions, distribution

def get_energy_levels(element):
    # Define the NIST Levels Tool URL
    url = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
    
    # Define parameters for the query
    params = {
        "spectrum": element,  # Neutral oxygen
        "units": "0",       # Units in cm^-1
        "format": "1",      # Machine-readable text output
        "output": "1",      # Request data for programmatic use
        "conf_out": "on",   # Include configurations
        "term_out": "on",   # Include terms
        "level_out": "on",  # Include level energies
        "j_out": "on",      # Include J quantum numbers
        "g_out": "on",      # Include statistical weights (g)
    }
    
    # Make the GET request
    response = requests.get(url, params=params)
    response.raise_for_status()  # Ensure the request was successful

    # Process the response
    lines = response.text.splitlines()

    levels = []
    
    for line in lines:
        # Skip lines that are separators or irrelevant
        if set(line.strip()) in [{"-"}, {"|"}, {" "}, {""}]:
            continue
        
        # Attempt to parse valid level data
        parts = line.split("|")
        if len(parts) >= 5:  # Check if there are enough columns
            try:
                g = float(parts[3].strip())  # Parse 'g' (statistical weight)
                energy = float(parts[4].strip())  # Parse 'Level (cm^-1)'
                levels.append((g, energy))
            except ValueError:
                # Skip lines that do not contain valid numeric data
                continue

    return levels

def calculate_partition_function(levels, T_e):
    """
    Calculate the partition function for O I at a given electron temperature T_e (in eV).
    Args:
        levels: List of tuples [(g, E), ...] where g is the statistical weight 
                and E is the energy in cm^-1.
        T_e: Electron temperature in eV.

    Returns:
        Partition function Z(T).
    """
    k_B_eV = 8.617333262145e-5  # Boltzmann constant in eV/K
    h = 6.62607015e-34          # Planck constant in J*s
    c = 2.99792458e10           # Speed of light in cm/s
    eV_conversion_factor = 1.60218e-19  # 1 eV in Joules

    # Convert cm^-1 to eV (E = h * c * E_cm / eV_conversion_factor)
    levels_in_eV = [(g, (h * c * E) / eV_conversion_factor) for g, E in levels]
    
    # Find the minimum energy level for numerical stability
    min_energy = min(E for _, E in levels_in_eV)
    
    # Calculate the partition function using the log-sum-exp trick
    Z = np.zeros(len(T_e))
    for g, E in levels_in_eV:
        Z[:] += g * np.exp(-(E - min_energy) / T_e[:])
    
    # Adjust for the minimum energy
    Z *= np.exp(-min_energy / T_e)
    
    return Z

# Main execution
def partition(element, temperature):
    # Retrieve energy levels for O I
    levels = get_energy_levels(element)
    
    # Define electron temperature in eV
    T_e = temperature/1.1605e4  # Example electron temperature in eV

    # Calculate partition function
    Z = calculate_partition_function(levels, T_e)
    
    return Z