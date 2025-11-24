#
# Copyright (c) 2025 TITAN Contributors (cf. AUTHORS.md).
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

import subprocess, pathlib, glob
import numpy as np
import pymap3d
import pandas as pd
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import Voronoi
from Geometry import component, assembly
from scipy.stats import norm
from Explosion.generate_seed_distribution import optimal_seeds
from Explosion.gmsh_voronoi_fracture import generate_fragment_meshes
def fracture_object(explobject, parent_COG, options):
    ## This function takes a demising component 
    ## and returns a collection of fragments to be added to the simulation
    expl_name = explobject.name.split('/')[-1].split('.')[0]
    expl_dir = options.output_folder+'/Generated_fragments/'+expl_name
    
    frag_material = explobject.material_name
    frag_temp = explobject.temperature
    n_frags = explobject.explosive.n_fragments
    do_remesh = '--remesh' if options.explosion.remesh else '--no-remesh'

    obj_len = np.linalg.norm(explobject.mesh.max-explobject.mesh.min)
    if not pathlib.Path(expl_dir+'/points.csv').exists():
        pathlib.Path(expl_dir).mkdir(exist_ok=True, parents=True)
    #     optimal_seeds(expl_dir=expl_dir, n_fragments=n_frags, CoG=explobject.COG,plot=False, obj_len=obj_len, compute_budget=2e2, method='spiral')
    from scipy.stats import multivariate_normal
    points = multivariate_normal(explobject.COG, np.diag([0.3,0.3,0.3])).rvs(n_frags)#np.genfromtxt(expl_dir+'/points.csv',delimiter=',',dtype=np.float64)
    vor = Voronoi(points)
    
    generate_fragment_meshes(explobject.mesh, explobject, vor, write=True, output_folder=expl_dir)
    # subprocess.run(['blender', '-b', '-P', './Explosion/voronoi_fracture_manifold.py', '--', 
    #                 '--file={}'.format(explobject.name), 
    #                 '-n={}'.format(n_frags), 
    #                 '--noise={}'.format(0.25), 
    #                 '-o={}'.format(expl_dir),
    #                 '-d={}'.format(explobject.material.density),
    #                 '--seeds={}'.format(expl_dir+'/points.csv'),
    #                 '--vol={}'.format(1e-6),
    #                 '--debug',
    #                 '--wall={}'.format(1e-2)
    #                 ])
    #'--vol={}'.format(1e-6),
    n_frags = len(glob.glob("{}/{}_*.stl".format(expl_dir, expl_name)))
    data = pd.read_csv(expl_dir+'/' + expl_name + '_data.csv')
    explosion_parameters = {'nucleus' : explobject.COG,
                            'characteristic_velocity' : explobject.explosive.char_velocity,
                            'energy' : explobject.explosive.energy,
                            'kinetic_factor' : explobject.explosive.kinetic_factor,
                            'volume' : data['volume'].to_numpy(),
                            'mass' : data['mass'].to_numpy(),
                            'area' :data['area'].to_numpy(),
                            'area_mass' : data['area_mass'].to_numpy(),}

    explosion_parameters['velocities'] = sample_fragment_velocities(explosion_parameters, n_frags, options)
    new_fragments = component.Component_list()

    for i_frag in range(n_frags):
        new_fragments.insert_component(filename=expl_dir+'/'+expl_name+'_frag_'+str(i_frag)+'.stl',
                                                  file_type='Primitive',material=frag_material,
                                                  temperature=frag_temp, options=options, 
                                                  global_ID=-1*(i_frag+1), alpha=explobject.debug_alpha)
        #new_fragments.object[i_frag].compute_mass_properties()
        #TODO Assign correct fragment velocities
    
    return new_fragments, explosion_parameters

def build_new_assemblies(fragmentlist, titan, options, i_parent, explosion_parameters):
    ## This function takes a collection of fragments
    ## and generates a new "assembly" for each one
    ## and adds it to the main titan.assembly list
    parent = titan.assembly[i_parent]
    angle = np.array([parent.roll, parent.pitch, parent.yaw])
    angle_vel = np.array([parent.roll_vel, parent.pitch_vel, parent.yaw_vel])
    distance_travelled = parent.distance_travelled
    #explosion_parameters['nucleus']-=parent.COG
    for i_fragment, fragment in enumerate(fragmentlist.object):
        if options.verbose: print('Creating fragment {}'.format(fragment.name.split('/')[-1]))
        new_assem = assembly.Assembly_list([fragment])
        new_assem.create_assembly(np.array([]),aoa=parent.aoa, slip=parent.slip, roll=parent.roll, options=options)
        new_assem.assembly[0].id = titan.id
        titan.assembly.append(new_assem.assembly[0])
        titan.id+=1
        titan.assembly[-1].generate_inner_domain(size_override=0.05)
        titan.assembly[-1].compute_mass_properties()

        titan.assembly[-1].roll  = angle[0]
        titan.assembly[-1].pitch = angle[1]
        titan.assembly[-1].yaw   = angle[2]

        titan.assembly[-1].roll_vel_last = deepcopy(parent.roll_vel)
        titan.assembly[-1].pitch_vel_last = deepcopy(parent.pitch_vel)
        titan.assembly[-1].yaw_vel_last = deepcopy(parent.yaw_vel)


        #Vector of COM difference
        dx = titan.assembly[-1].COG - parent.COG
        
        #Vector from body frame to ECEF frame
        R_B_ECEF = Rot.from_quat(parent.quaternion)
        dx_ECEF = R_B_ECEF.apply(dx)
        angle_vel_ECEF = R_B_ECEF.apply(angle_vel)

        titan.assembly[-1].position = np.copy(parent.position) + dx_ECEF
        dv = add_fragment_dv(titan.assembly[-1],i_fragment,explosion_parameters,options)
        titan.assembly[-1].velocity = np.copy(parent.velocity) + np.cross(angle_vel_ECEF,dx_ECEF)+dv
        
        titan.assembly[-1].position_nlast = deepcopy(titan.assembly[-1].position)
        titan.assembly[-1].velocity_nlast = deepcopy(titan.assembly[-1].velocity)

        titan.assembly[-1].roll_vel  = angle_vel[0]
        titan.assembly[-1].pitch_vel = angle_vel[1]
        titan.assembly[-1].yaw_vel   = angle_vel[2]

        titan.assembly[-1].trajectory = deepcopy(parent.trajectory)
        titan.assembly[-1].trajectory.dyPrev = None
        titan.assembly[-1].quaternion = deepcopy(parent.quaternion)

        #Compute the trajectory and angular quantities
        [latitude, longitude, altitude] = pymap3d.ecef2geodetic(titan.assembly[-1].position[0], titan.assembly[-1].position[1], titan.assembly[-1].position[2],
                                        ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                        deg = False);
        titan.assembly[-1].trajectory.latitude = latitude
        titan.assembly[-1].trajectory.longitude = longitude
        titan.assembly[-1].trajectory.altitude = altitude
        titan.assembly[-1].distance_travelled = distance_travelled 

        [vEast, vNorth, vUp] = pymap3d.uvw2enu(titan.assembly[-1].velocity[0], titan.assembly[-1].velocity[1], titan.assembly[-1].velocity[2], latitude, longitude, deg=False)

        titan.assembly[-1].trajectory.gamma = np.arcsin(np.dot(titan.assembly[-1].position, titan.assembly[-1].velocity)/(np.linalg.norm(titan.assembly[-1].position)*np.linalg.norm(titan.assembly[-1].velocity)))
        titan.assembly[-1].trajectory.chi = np.arctan2(vEast,vNorth)

        #ECEF_2_B
        [Vx_B, Vy_B, Vz_B] =  Rot.from_quat(titan.assembly[-1].quaternion).inv().apply(titan.assembly[-1].velocity)
        titan.assembly[-1].trajectory.velocity = np.linalg.norm([Vx_B, Vy_B, Vz_B])
        
        # titan.assembly[-1].aoa = parent.aoa
        # titan.assembly[-1].slip = parent.slip

        from Dynamics.propagation import construct_state_vector
        construct_state_vector(titan.assembly[-1])
        titan.assembly[-1].unmodded_angles = parent.unmodded_angles

        if options.collision.flag: 
            from Dynamics.collision import generate_collision_mesh, generate_collision_handler
            generate_collision_mesh(titan.assembly[-1], options)
            generate_collision_handler(titan, options)

def add_fragment_dv(fragment_assem, frag_id, explosion_parameters, options):
    explosion_dir =  explosion_parameters['nucleus'] - fragment_assem.COG
    explosion_dir /= np.linalg.norm(explosion_dir)
    v = explosion_parameters['velocities'][frag_id]
    if options.verbose: print('Added velocity to fragment {} of v={}m/s'.format(frag_id,v))
    return explosion_dir*v

def sample_fragment_velocities(explosion_parameters, n_fragments, options):
    velocities = np.zeros(n_fragments)
    if options.verbose:
        print('Applying {} velocity method with a Total Energy of {}J'.format(options.explosion.vel_method, 
                                                                              explosion_parameters['energy']))
    method = options.explosion.vel_method.lower()
    match method:
        case 'nasa':
            for i_fragment in range(n_fragments):
                velocities[i_fragment] = evolve_4_explosion_velocity(explosion_parameters['characteristic_velocity'],
                                                                     explosion_parameters['area_mass'][i_fragment])
        case 'nasa_conservation':
            for i_fragment in range(n_fragments):
                velocities[i_fragment] = evolve_4_explosion_velocity(explosion_parameters['characteristic_velocity'],
                                                                     explosion_parameters['area_mass'][i_fragment])
            kinetic_energy = 0.5*np.array(explosion_parameters['mass'])*velocities*velocities
            available_energy = explosion_parameters['energy']*explosion_parameters['kinetic_factor']
            scale_factor = np.sqrt(available_energy/np.sum(kinetic_energy))
            if options.verbose:
                print('Scaling velocities by {} such that {}J maps onto {}J'.format(scale_factor,
                                                                                    np.sum(kinetic_energy),
                                                                                    available_energy))
            velocities*= scale_factor
    return velocities

def evolve_4_explosion_velocity(base_v, area_mass_ratio):
    ## NASA EVOLVE 4.0 Explosion Velocity Distribution Function
    # https://doi.org/10.1016/S0273-1177(01)00423-9
    v = np.log10(base_v)
    chi = np.log10(area_mass_ratio)
    mu  = 0.2 * chi + 1.85
    std = 0.4
    return norm.rvs(loc=mu,scale=std)
    return (1/std*np.sqrt(2*np.pi))*np.exp(-(v - mu)**2/(2*std**2))