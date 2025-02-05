from Dynamics import dynamics, frames, collision
from Aerothermo import aerothermo
from Forces import forces
import pymap3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy import integrate
from functools import partial
from Output import output

## Current implented integrators (define in cfg under [Time] as Time_integration='')...

## Constant time-step methods  
## - euler  : Euler method
## - bwd    : Backward difference, using information from 1 previous time step (2nd order)
## - AB[N]  : Adams-Bashford Nth order for N = 2-5 (AB2,...,AB5), using information from N-1 previous time step(s)

## Adaptive time-step methods (via scipy.integrate)
## - RK23   : Time stepping according to 3rd order Runge-Kutta with 2nd order error control
## - RK45   : Time stepping according to 5th order Runge-Kutta with 4th order error control
## - DOP853 : The DOP8(5,3) adaptive algorithm 
## See docs.scipy.org/doc/scipy/reference/integrate.html for more info

## NB: All current methods are explicit

def propagate(titan, options):
    # Main propagator function, updates the state of all assemblies according to...
    #  a 13-D state vector of form [Position(x/y/z),Velocity(u/v/w),Quaternion(w/i/j/k),Angular velocity(roll_vel,pitch_vel,yaw_vel)]
    #  a propagator specified by options.dynamics.propagator

    # TODO: Manage collision handling properly
    if options.collision.flag and len(titan.assembly)>1:
        flag_collision, __ = collision.check_collision(titan, options, 0)
        if flag_collision: collision.collision_physics(titan, options)
        #if flag_collision: collision.collision_physics_simultaneous(titan, options)

    # If we go to switch.py or su2.py, Because we call deepcopy() function, we need to rebuild
    #the collision mesh
    if options.collision.flag and options.fidelity.lower() in ['multi','high']:
        for assembly in titan.assembly: collision.generate_collision_mesh(assembly, options)
        collision.generate_collision_handler(titan, options)

    # Retrieve state vectors from each assembly object...
    current_state_vectors, state_vectors_prior, derivatives_prior = collect_state_vectors(titan, options)

    ## NB: Highly unsure if I've correctly applied this collision implementation, see earlier todo
    time_step = options.dynamics.time_step
    if options.collision.flag and len(titan.assembly)>1:
        #Check collision for future time intervals with respect to current time-step velocity
        __, time_step = collision.check_collision(titan, options, time_step)
    
    # Propagate according to propagator function...
    new_state_vectors, new_derivs = options.dynamics.prop_func(current_state_vectors,state_vectors_prior,derivatives_prior,time_step,titan,options)

    # Update prior derivatives
    if new_derivs is not None: append_derivatives(titan,options,new_derivs)

    # Communicate new vectors to assemblies
    for i_assem, _assembly in enumerate(titan.assembly):
        update_dynamic_attributes(_assembly,new_state_vectors[i_assem],options)
        _assembly.state_vector = new_state_vectors[i_assem]
    
    # Writes the output data
    output.write_output_data(titan = titan, options = options)

    # Increment time step
    titan.time += time_step

def state_equation(titan,options,time,state_vectors):
    # This state equation will for each assembly compute the rate of change of a state vector (at that state),
    
    # State vectors can be passed flattened, need to account for this
    state_vectors = np.array(state_vectors)
    reshape_flat = False
    if len(state_vectors.shape)<2: 
        reshape_flat = True
        state_vectors = np.reshape(state_vectors,[-1,13])

    # First we communicate the state vector to assembly attributes
    for _assembly, state_vector in zip(titan.assembly,state_vectors):
        update_dynamic_attributes(_assembly,state_vector,options)
    
    # Then business as usual for computing forces...
    aerothermo.compute_aerothermo(titan, options)

    forces.compute_aerodynamic_forces(titan, options)
    forces.compute_aerodynamic_moments(titan, options)

    # Then determine the necessary derivatives to return the state vector(s)
    d_dt_state_vectors = []
    for _assembly in titan.assembly:
        angularDerivatives = dynamics.compute_angular_derivatives(_assembly)
        cartesianDerivatives = dynamics.compute_cartesian_derivatives(_assembly, options)

        # Use quaternion derivative equation 
        omega_q = [angularDerivatives.droll,angularDerivatives.dpitch,angularDerivatives.dyaw,0.0]
        d_q  = 0.5 *  quaternion_mult(_assembly.quaternion,omega_q)

        d_dt_state_vectors.append([cartesianDerivatives.dx,
                                   cartesianDerivatives.dy,
                                   cartesianDerivatives.dz,
                                   cartesianDerivatives.du,
                                   cartesianDerivatives.dv,
                                   cartesianDerivatives.dw,
                                   d_q[0],
                                   d_q[1],
                                   d_q[2],
                                   d_q[3],
                                   angularDerivatives.ddroll,
                                   angularDerivatives.ddpitch,
                                   angularDerivatives.ddyaw,])
        
    if reshape_flat: d_dt_state_vectors = np.array(d_dt_state_vectors).flatten()
    return d_dt_state_vectors

def update_dynamic_attributes(assembly,state_vector,options):
    # This function takes an ECEF/BODY state vector and applies it to all the necessary attributes of a TITAN assembly
    # This ensures the new dynamics code plays nicely with other parts of TITAN.

    if not np.array_equal(assembly.state_vector,state_vector): # Only need to update if state vector is different
        # Update ECEF state...
        assembly.position[0] = state_vector[0]
        assembly.position[1] = state_vector[1]
        assembly.position[2] = state_vector[2]

        assembly.velocity[0] = state_vector[3]
        assembly.velocity[1] = state_vector[4]
        assembly.velocity[2] = state_vector[5]

        # Update BODY state...
        assembly.quaternion[0] = state_vector[6]
        assembly.quaternion[1] = state_vector[7]
        assembly.quaternion[2] = state_vector[8]
        assembly.quaternion[3] = state_vector[9]
        assembly.quaternion = quaternion_normalize(assembly.quaternion)

        assembly.roll_vel = state_vector[10]
        assembly.pitch_vel = state_vector[11]
        assembly.yaw_vel = state_vector[12]

        # Communicate state to other assembly attributes...
        [latitude, longitude, altitude] = pymap3d.ecef2geodetic(assembly.position[0], assembly.position[1], assembly.position[2],
                                        ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                        deg = False);

        assembly.trajectory.latitude = latitude
        assembly.trajectory.longitude = longitude
        assembly.trajectory.altitude = altitude

        R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
        R_B_NED_quat = (R_NED_ECEF).inv()*Rot.from_quat(assembly.quaternion)
        [yaw,pitch,roll] = R_B_NED_quat.as_euler('ZYX')

        assembly.roll = roll
        assembly.yaw = yaw
        assembly.pitch = pitch

        [vEast, vNorth, vUp] = pymap3d.uvw2enu(assembly.velocity[0], assembly.velocity[1], assembly.velocity[2], latitude, longitude, deg=False)

        assembly.trajectory.gamma = np.arcsin(np.dot(assembly.position, assembly.velocity)/(np.linalg.norm(assembly.position)*np.linalg.norm(assembly.velocity)))
        assembly.trajectory.chi = np.arctan2(vEast,vNorth)

        #ECEF_2_B
        [Vx_B, Vy_B, Vz_B] =  Rot.from_quat(assembly.quaternion).inv().apply(assembly.velocity)
        assembly.trajectory.velocity = np.linalg.norm([Vx_B, Vy_B, Vz_B])

        assembly.aoa = np.arctan2(Vz_B,Vx_B)
        assembly.slip = np.arcsin(Vy_B/np.sqrt(Vx_B**2 + Vy_B**2 +  Vz_B**2))

    return assembly

def construct_state_vector(assembly):
    assembly.state_vector = [0 for _ in range(13)]

    assembly.state_vector[0]  = assembly.position[0]
    assembly.state_vector[1]  = assembly.position[1]
    assembly.state_vector[2]  = assembly.position[2]

    assembly.state_vector[3]  = assembly.velocity[0]
    assembly.state_vector[4]  = assembly.velocity[1]
    assembly.state_vector[5]  = assembly.velocity[2]

    assembly.state_vector[6]  = assembly.quaternion[0]
    assembly.state_vector[7]  = assembly.quaternion[1]
    assembly.state_vector[8]  = assembly.quaternion[2]
    assembly.state_vector[9]  = assembly.quaternion[3]

    assembly.state_vector[10]  = assembly.roll_vel
    assembly.state_vector[11]  = assembly.pitch_vel
    assembly.state_vector[12]  = assembly.yaw_vel    

    assembly.state_vector_prior = []
    assembly.derivs_prior = []

def collect_state_vectors(titan,options):
    # Collect state vectors

    current_state_vectors = []
    n_prior = titan.iter if titan.iter<options.dynamics.n_states_to_hold else options.dynamics.n_states_to_hold
    n_derivs = titan.iter if titan.iter<options.dynamics.n_derivs_to_hold else options.dynamics.n_derivs_to_hold
    state_vectors_prior = [[] for i_states in range(n_prior)] # It's easier to do this with lists as the shape of our priors can change
    derivatives_prior = [[] for i_derivs in range(n_derivs)] # Even if it is very ugly

    for _assembly in titan.assembly:

        if not hasattr(_assembly,'state_vector'): construct_state_vector(_assembly) 
        #^ Something of a hack but assemblies must be fully instantiated *before* the state vectors can be constructed
        current_state_vectors.append(_assembly.state_vector)

        # Collect prior states
        for i_states in range(n_prior):
            state_vectors_prior[i_states].append(_assembly.state_vector_prior[i_states])

        for i_states in range(n_derivs): 
            derivatives_prior[i_states].append(_assembly.derivs_prior[i_states])

    # Append states to prior
    for _assembly in titan.assembly: 
        if n_prior==options.dynamics.n_states_to_hold and len(_assembly.state_vector_prior)>0:
            _assembly.state_vector_prior.pop(0)
        _assembly.state_vector_prior.append(_assembly.state_vector)

    return current_state_vectors, state_vectors_prior, derivatives_prior

def append_derivatives(titan,options,new_derivs):
    n_derivs = titan.iter if titan.iter<options.dynamics.n_derivs_to_hold else options.dynamics.n_derivs_to_hold
    for i_assem, _assembly, in enumerate(titan.assembly):
        if n_derivs==options.dynamics.n_derivs_to_hold and len(_assembly.derivs_prior)>0:
            _assembly.derivs_prior.pop(0)
        _assembly.derivs_prior.append(new_derivs[i_assem])

def setup_integrator(options):

    choice = options.dynamics.propagator

    if choice == 'euler': 
        options.dynamics.prop_func = explicit_euler_propagate
    elif 'bwd' in choice and not 'legacy' in choice: 
        options.dynamics.prop_func = explicit_bwd_diff_propagate
        options.dynamics.n_states_hold = 1
    elif 'RK' in choice or 'DOP': 
        if '45' in choice: algo = integrate.RK45
        elif choice == 'DOP853': algo = integrate.DOP853
        elif '23' in choice: algo = integrate.RK23
        options.dynamics.prop_func = partial(explicit_rk_adapt_wrapper,algo)
    elif 'AB' in choice: 
        n = int(choice[2:])
        if n>5:
            print('Only Adams-Bashforth methods of order 2-5 are implemented! Setting order to 5...')
            n = 5
        options.dynamics.n_derivs_to_hold = n - 1
        options.dynamics.prop_func = partial(explicit_adams_bashforth_n,n)

#############################################################################################################################################
#############################################################################################################################################
###########################################################  ALGORITHMS  ####################################################################
#############################################################################################################################################
#############################################################################################################################################

def explicit_euler_propagate(state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    new_state_vectors = []
    d_dt_state_vectors = state_equation(titan,options,dt,state_vectors)
    for i_assem, _assembly in enumerate(titan.assembly):
        new_vector = []
        for element, d_dt_element in zip(state_vectors[i_assem],d_dt_state_vectors[i_assem]):
            new_vector.append(element+dt*d_dt_element)
        new_state_vectors.append(new_vector)
    return new_state_vectors, d_dt_state_vectors

def explicit_bwd_diff_propagate(state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    new_state_vectors = []
    if titan.iter==0: 
        new_state_vectors, d_dt_state_vectors = explicit_euler_propagate(state_vectors,state_vectors_prior,
                                                                         derivatives_prior,dt,titan,options)
    else:
        d_dt_state_vectors = state_equation(titan,options,dt,state_vectors)
        for i_assem, _assembly in enumerate(titan.assembly):
            new_vector = []
            for element_last, d_dt_element in zip(state_vectors_prior[i_assem],d_dt_state_vectors[i_assem]):
                new_vector.append(element_last+2*dt*d_dt_element)
            new_state_vectors.append(new_vector)
    return new_state_vectors, d_dt_state_vectors

def explicit_adams_bashforth_n(n,state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    coeffs = {2 : [-0.5,1.5], 3 : [5.0/12.0,-16.0/12.0,23.0/12.0], 4 : [-9.0/24.0,37.0/24.0,-59.0/24.0,55.0/24.0],
              5 : [251.0/720.0, -1274.0/720.0,2616.0/720.0,-2774.0/720.0,1901.0/720.0]}
    new_state_vectors = []
    if titan.iter<n-1: 
        if titan.iter==0: 
            new_state_vectors, d_dt_state_vectors = explicit_euler_propagate(state_vectors,state_vectors_prior,
                                                                             derivatives_prior,dt,titan,options)
        else:
            new_state_vectors, d_dt_state_vectors = explicit_adams_bashforth_n(titan.iter+1,state_vectors,state_vectors_prior,
                                                                               derivatives_prior,dt,titan,options)
    else:
        d_dt_state_vectors = state_equation(titan,options,dt,state_vectors)
        derivatives_prior.append(d_dt_state_vectors)
        for i_assem, _assembly in enumerate(titan.assembly):
            new_vector = []
            for i_elem, element in enumerate(state_vectors[i_assem]):
                new_element = element
                for i_prior in range(n):
                    new_element += dt*coeffs[n][i_prior] * derivatives_prior[i_prior][i_assem][i_elem]
                new_vector.append(new_element)
            new_state_vectors.append(new_vector)
        titan.previous_derivatives = d_dt_state_vectors
    return new_state_vectors, d_dt_state_vectors

def explicit_rk_adapt_wrapper(algorithm, state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    if not hasattr(titan, 'rk_fun'): titan.rk_fun=partial(state_equation,titan,options)
    if not hasattr(titan,'rk_params'):
        titan.rk_params = [titan.time, np.array(state_vectors).flatten(),
                     titan.time + dt*options.iters, 
                     dt]
    if not hasattr(titan, 'rk_adapt'): titan.rk_adapt = algorithm(fun=titan.rk_fun,
                                                                  t0=titan.rk_params[0],
                                                                  y0=titan.rk_params[1],
                                                                  t_bound=titan.rk_params[2],
                                                                  first_step=titan.rk_params[3])

    if titan.rk_adapt.status == 'running':
        titan.rk_adapt.step()
    else: 
        print('Simulation finished with propagation status {} ({} function evaluations)'.format(titan.rk45.status,titan.rk45.nfev))
    titan.time = titan.rk_adapt.t
    return np.reshape(titan.rk_adapt.y,[-1,13]), None

#############################################################################################################################################
#############################################################################################################################################
###################################################  QUATERNION HELPER FUNCTIONS  ###########################################################
#############################################################################################################################################
#############################################################################################################################################

def quaternion_mult(q1,q2):
    return np.array([q1[3]*q2[0]+q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1],
            q1[3]*q2[1]+q1[1]*q2[3]-q1[0]*q2[2]+q1[2]*q2[0],
            q1[3]*q2[2]+q1[2]*q2[3]+q1[0]*q2[1]-q1[1]*q2[0],
            q1[3]*q2[3]-q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]])

def quaternion_conjugate(q): return np.array([-q[0],-q[1],-q[2],q[3]])

def quaternion_normalize(q):
    norm = np.linalg.norm(q)
    return q/norm