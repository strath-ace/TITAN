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

"""explosion_fragments module."""
import subprocess, pathlib, glob
import numpy as np
import pymap3d
import pandas as pd
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import Voronoi
from ..Geometry import component, assembly, mesh
from scipy.stats import norm
from ..Explosion.generate_seed_distribution import optimal_seeds, SpiralSampler
from ..Explosion.manifold_voronoi_fracture import generate_fragment_meshes, mesh_check
from ..Dynamics.propagation import construct_state_vector
from scipy.spatial.transform import Rotation as Rot

def fracture_object(explobject, parent, options, dt = None, base_rng = None):
    '''
Takes a demising component and returns a collection of fragments to be added to the simulation
    :param explobject: Object of class Component to fracture and disperse
    :type explobject: Any
    :param parent_state: Object's parent assembly
    :type parent_state: Any
    :param options: Object of class Options
    :type options: object
    :param dt: Duration of explosion in seconds
    :type dt: float
    :param base_rng: Object of class numpy.random.RandomState defining global rng
    :type base_rng: Any
'''
    expl_name = explobject.name.split('/')[-1].split('.')[0]
    expl_dir = options.output_folder+'/Generated_fragments/'+expl_name
    
    frag_material = explobject.material_name
    frag_temp = explobject.temperature
    n_frags = explobject.explosive.n_fragments

    obj_len = np.linalg.norm(explobject.mesh.max-explobject.mesh.min)
    obj_mesh = mesh.Mesh(explobject.name)
    explobject.mesh = mesh.compute_mesh(obj_mesh)

    if not pathlib.Path(expl_dir+'/points.csv').exists():
        pathlib.Path(expl_dir).mkdir(exist_ok=True, parents=True)

    max_attempts = options.explosion.max_mesh_attempts
    # Bias nucleus towards CoG, helps with generating well-behaved Voronoi patterns
    cog_bias = options.explosion.nucleus_CoG_bias
    if options.explosion.voronoi_budget>0: 
        spiral_weights = optimal_seeds(n_fragments=n_frags, CoG=[0,0,0], plot=True, obj_len=obj_len, 
                                       compute_budget=options.explosion.voronoi_budget, method='spiral')
    else: 
        spiral_weights = np.array([0.8 * 0.6064932580341802, 
                                         0.7066237981471332, 
                                        35.36336186280215, 
                                        34.24156856002203])
    if options.verbose: print('Sample weights of...\n',spiral_weights)

    base_state = base_rng.get_state() if base_rng is not None else None

    for i_attempt in range(max_attempts):
        # Want a sequence of random states that can be recovered
        if base_state is None:
            rng = np.random.RandomState(6+i_attempt)
        else: 
            rng = np.random.RandomState(None)
            rng.set_state(base_state)
            rng = np.random.RandomState(rng.randint(0,2**16)+i_attempt)

        match options.explosion.nucleus_choice.lower():
            case 'random':
                f_id = rng.choice(np.arange(explobject.mesh.facet_area.shape[0]))
            case 'heat_flux':
                f_id = np.argmax(parent.aerothermo.heatflux[explobject.facet_index])
            case 'temperature':
                f_id = np.argmax(parent.aerothermo.temperature[explobject.facet_index])
            case 'pressure':
                f_id = np.argmax(parent.aerothermo.pressure[explobject.facet_index])

        facet = explobject.mesh.facet_COG[f_id,:]
        nucleus = cog_bias*explobject.COG + (1-cog_bias)*facet
        
        points  = SpiralSampler(spiral_weights[0],spiral_weights[1],spiral_weights[2],spiral_weights[3], rng=rng).rvs(n_frags)
        points += np.full_like(points,nucleus)
        vor = Voronoi(points)

        try:
            generate_fragment_meshes(explobject.name,vor, expl_dir, extrude=explobject.explosive.crack_width)

            if mesh_check(expl_dir,explobject.material.density, explobject.volume, 
                          threshold=options.explosion.mesh_err_pct, delete_bad=True, 
                          quiet=(not options.verbose)): break
        except Exception as e: 
            print(e)
        if i_attempt<max_attempts-1:
            print('Voronoi {} failed mesh check! Recalculating...'.format(i_attempt))
            if pathlib.Path(expl_dir+'/stats.csv').resolve().exists():
                pathlib.Path(expl_dir+'/stats.csv').resolve().unlink()
            for frag in glob.glob("{}/*.stl".format(expl_dir)): pathlib.Path(frag).unlink()
        else: raise Exception('Could not build voronoi fragments after {} attempts'.format(i_attempt+1))

    n_frags = len(glob.glob("{}/frag_*.stl".format(expl_dir)))

    data = pd.read_csv(expl_dir+'/' + 'stats.csv')
    ids = [int(frag_name.split('_')[-1]) for frag_name in data['name'].to_list()]
    explosion_parameters = {'nucleus' : parent.state_vector[:3]+Rot.from_quat(parent.state_vector[6:10]).apply(nucleus-parent.COG),
                            'characteristic_velocity' : explobject.explosive.char_velocity,
                            'energy' : explobject.explosive.energy,
                            'kinetic_factor' : explobject.explosive.kinetic_factor,
                            'volume' : data['volume'].to_numpy(),
                            'mass' : data['mass'].to_numpy(),
                            'area' :data['surf_area'].to_numpy(),
                            'area_mass' : data['area_mass_ratio'].to_numpy(),
                            'lref' : data['reference_length'].to_numpy(),
                            'ids' : np.array(ids)}

    explosion_parameters['velocities'] = sample_fragment_velocities(explosion_parameters, n_frags, options, dt)
    new_fragments = component.Component_list()

    for i_frag in range(n_frags):
        new_fragments.insert_component(filename=expl_dir+'/'+'frag_'+str(ids[i_frag])+'.stl',
                                                  file_type='Primitive',material=frag_material,
                                                  temperature=frag_temp, options=options, 
                                                  global_ID=-1*(i_frag+1), alpha=explobject.debug_alpha, 
                                                  mixture=explobject.mixture, mass_fractions=explobject.mass_fraction,
                                                  species=explobject.species)
    explobject.mass = 0.0
    return new_fragments, explosion_parameters

def build_new_assemblies(fragment_list, titan, options, i_parent, explosion_parameters):
    '''
Takes a collection of fragments, builds each a new "assembly" and appends it to titan.assembly
    :param fragment_list: Object of class AssemblyList, generated by fracture_object()
    :type fragment_list: list
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
    :param i_parent: Index of the parent assembly of pre-exploded object
    :type i_parent: Any
    :param explosion_parameters: Dict of kinetic parameters, generated by fracture_object()
    :type explosion_parameters: Any
'''

    parent = titan.assembly[i_parent]
    angle = np.array([parent.roll, parent.pitch, parent.yaw])
    angle_vel = np.array([parent.roll_vel, parent.pitch_vel, parent.yaw_vel])
    distance_travelled = parent.distance_travelled

    for i_fragment, fragment in enumerate(fragment_list.object):
        if options.verbose: print('Creating fragment {}'.format(fragment.name.split('/')[-1]))
        new_assem = assembly.Assembly_list([fragment])
        new_assem.create_assembly(np.array([]),aoa=parent.aoa, slip=parent.slip, roll=parent.roll, options=options)
        new_assem.assembly[0].id = titan.id
        titan.assembly.append(new_assem.assembly[0])
        titan.id+=1
        titan.assembly[-1].generate_inner_domain(size_override=explosion_parameters['lref'][i_fragment]*1e-2, 
                                                 min_size=explosion_parameters['lref'][i_fragment]*5e-3)
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
        R_ECEF_from_B = Rot.from_quat(parent.quaternion)
        dx_ECEF = R_ECEF_from_B.apply(dx)
        angle_vel_ECEF = R_ECEF_from_B.apply(angle_vel)

        titan.assembly[-1].position = np.copy(parent.position) + dx_ECEF
        dv = calculate_fragment_dv(titan.assembly[-1],i_fragment,explosion_parameters,options)
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

        construct_state_vector(titan.assembly[-1], options.dynamics.augmented_state)
        titan.assembly[-1].unmodded_angles = parent.unmodded_angles

        if options.collision.flag: 
            from ..Dynamics.collision import generate_collision_mesh, generate_collision_handler
            generate_collision_mesh(titan.assembly[-1], options)
            generate_collision_handler(titan, options)

def calculate_fragment_dv(fragment_assem, frag_id, explosion_parameters, options):
    '''
Computes change in velocity for each fragment in an explosion
    :param fragment_assem: Object of Class Assembly
    :type fragment_assem: Any
    :param frag_id: Index of fragment in explosion_parameters dict
    :type frag_id: int
    :param explosion_parameters: Dict of kinetic parameters, generated by fracture_object()
    :type explosion_parameters: Any
    :param options: Object of Class Options
    :type options: object
'''

    explosion_dir  = fragment_assem.position - explosion_parameters['nucleus']
    explosion_dir /= np.linalg.norm(explosion_dir)
    v = explosion_parameters['velocities'][frag_id]
    if options.verbose: print('Added velocity to fragment {} of v={}m/s'.format(frag_id,v))
    return explosion_dir*v

def sample_fragment_velocities(explosion_parameters, n_fragments, options, dt = None):
    '''
Select delta velocities assigned to fragments based upon a velocity method
    :param explosion_parameters: Dict of kinetic parameters, generated by fracture_object()
    :type explosion_parameters: Any
    :param n_fragments: Number of fragments
    :type n_fragments: int
    :param options: Object of Class Options
    :type options: object
    :param dt: Duration of explosion in seconds
    :type dt: float
'''
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

        case 'amr':
            if dt is None: raise Exception('Must provide delta t for AMR-based velocity calculation!')
            # Define characteristic_pressure as the pressure to give characteristic_velocity to a fragment of AMR 1
            characteristic_pressure = explosion_parameters['characteristic_velocity']/dt
            for i_fragment in range(n_fragments):
                velocities[i_fragment] = 0.5*characteristic_pressure*explosion_parameters['area_mass'][i_fragment]*dt

        case 'amr_conservation':
            if dt is None: raise Exception('Must provide delta t for AMR-based velocity calculation!')
            # Define characteristic_pressure as the pressure to give characteristic_velocity to a fragment of AMR 1
            characteristic_pressure = explosion_parameters['characteristic_velocity']/dt
            for i_fragment in range(n_fragments):
                velocities[i_fragment] = 0.5*characteristic_pressure*explosion_parameters['area_mass'][i_fragment]*dt
            
            kinetic_energy = 0.5*np.array(explosion_parameters['mass'])*velocities*velocities
            available_energy = explosion_parameters['energy']*explosion_parameters['kinetic_factor']
            scale_factor = np.sqrt(available_energy/np.sum(kinetic_energy))
            if options.verbose:
                print('Scaling velocities by {} such that {}J maps onto {}J'.format(scale_factor,
                                                                                    np.sum(kinetic_energy),
                                                                                    available_energy))
    return velocities

def evolve_4_explosion_velocity(base_v, area_mass_ratio):
    '''
NASA EVOLVE 4.0 Explosion Velocity Distribution Function
    :param base_v: Base velocity
    :type base_v: Any
    :param area_mass_ratio: Area to mass ratio of the fragment
    :type area_mass_ratio: float
'''
    
    v = np.log10(base_v)
    chi = np.log10(area_mass_ratio)
    mu  = 0.2 * chi + 1.85
    std = 0.4
    return norm.rvs(loc=mu,scale=std)
    return (1/std*np.sqrt(2*np.pi))*np.exp(-(v - mu)**2/(2*std**2))
