from Dynamics import dynamics, frames, collision
from Aerothermo import aerothermo
from Forces import forces
import pymap3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy import integrate
from functools import partial
from Output import output
from warnings import warn

## Current implented integrators (define in cfg under [Time] as Time_integration='')...

## Constant time-step methods  
## - euler  : Euler method
## - bwd    : Backward difference, using information from 1 previous time step (2nd order)
## - AB[N]  : Adams-Bashford Nth order for N = 2-5 (AB2,...,AB5), using information from N-1 previous time step(s)
## - RK[N]  : Runge-Kutte Nth order for N = 2-5 (RK2,...,RK5), using information from N flow solves per time step

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

    n_prior = titan.post_event_iter if titan.post_event_iter<options.dynamics.n_states_to_hold else options.dynamics.n_states_to_hold
    n_derivs = titan.post_event_iter if titan.post_event_iter<options.dynamics.n_derivs_to_hold else options.dynamics.n_derivs_to_hold
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
    n_derivs = titan.post_event_iter if titan.post_event_iter<options.dynamics.n_derivs_to_hold else options.dynamics.n_derivs_to_hold
    for i_assem, _assembly, in enumerate(titan.assembly):
        if n_derivs==options.dynamics.n_derivs_to_hold and len(_assembly.derivs_prior)>0:
            _assembly.derivs_prior.pop(0)
        _assembly.derivs_prior.append(new_derivs[i_assem])

def get_integrator_func(options, choice):

    print('Selected...')
    if 'legacy' in choice:
        print('...legacy propagation, note these methods are deprecated.')
        return None
    if 'euler' in choice: 
        print('...Euler propagation')
        return explicit_euler_propagate
    if 'bwd' in choice and not 'legacy' in choice: 
        options.dynamics.n_states_to_hold = 1
        print('...backward difference propagation')
        return explicit_bwd_diff_propagate
    if 'rk' in choice or 'dop' in choice: 
        if len(choice.replace('rk',''))>1:
            if '45' in choice: 
                algo = integrate.RK45
                print('...scipy RK45 propagation')
            elif 'dop853' in choice: 
                algo = integrate.DOP853
                print('...scipy DOP853 propagation')
            elif '23' in choice: 
                algo = integrate.RK23
                print('...scipy RK23 propagation')
            return partial(explicit_rk_adapt_wrapper,algo)
        else:
            N = int(choice.replace('rk',''))
            print ('... RK{} propagation'.format(N))
            return partial(explicit_rk_N,N)
    if 'ab' in choice: 
        n = int(choice[2:])
        if n>5:
            print('NB: Only Adams-Bashforth methods of order 2-5 are implemented! Setting order to 5 to use...')
            n = 5
        options.dynamics.n_derivs_to_hold = n - 1
        print('...Adams-Bashforth {}th order propagation'.format(n))
        if n == 5 :
            warn_msg = 'The AB5 Method can exhibit strange behaviour near ground, this is being investigated. \nIf your simulation involves ground impacts consider another method'
            warn(warn_msg)
        return partial(explicit_adams_bashforth_n,n)

    print('...propagator Not recognised! See available options ↓')
    print('''
    ## Current implented time integrators (define in cfg under [Time] as Time_integration='')...

    ## Constant time-step methods  
    ## - euler      : Euler method
    ## - bwd        : Backward difference, using information from 1 previous time step (2nd order)
    ## - AB[N]      : Adams-Bashford Nth order for N = 2-5 (AB2,...,AB5), using information from N-1 previous time step(s)
    ## - RK[N]      : Runge-Kutte Nth order for N = 2-5 (RK2,...,RK5), using information from N flow solves per time step

    ## Adaptive time-step methods (via scipy.integrate)
    ## - RK23       : Time stepping according to 3rd order Runge-Kutta with 2nd order error control
    ## - RK45       : Time stepping according to 5th order Runge-Kutta with 4th order error control
    ## - DOP853     : The DOP8(5,3) adaptive algorithm 
    ## See docs.scipy.org/doc/scipy/reference/integrate.html for more info
        
    ## Legacy Dynamics Implementation (deprecated)
    ## legacy_euler : Prior Euler method
    ## legacy_bwd   : Prior backward difference method for position updates
    ''')
    raise Exception('Invalid propagator')
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
    if titan.post_event_iter==0:
        new_state_vectors, d_dt_state_vectors = explicit_euler_propagate(state_vectors,state_vectors_prior,
                                                                         derivatives_prior,dt,titan,options)
    else:
        d_dt_state_vectors = state_equation(titan,options,dt,state_vectors)
        for i_assem, _assembly in enumerate(titan.assembly):
            new_vector = []
            for element_last, d_dt_element in zip(state_vectors_prior[0][i_assem],d_dt_state_vectors[i_assem]):
                new_vector.append(element_last+2*dt*d_dt_element)
            new_state_vectors.append(new_vector)
    return new_state_vectors, d_dt_state_vectors

def explicit_adams_bashforth_n(n,state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    coeffs = {2 : [  -0.5,1.5], 
              3 : [   5.0/12.0,    -16.0/12.0,    23.0/12.0], 
              4 : [  -9.0/24.0,     37.0/24.0,   -59.0/24.0,     55.0/24.0],
              5 : [251.0/720.0, -1274.0/720.0, 2616.0/720.0, -2774.0/720.0, 1901.0/720.0]}
    new_state_vectors = []
    if titan.post_event_iter<n-1: 
        if titan.post_event_iter==0: 
            new_state_vectors, d_dt_state_vectors = explicit_euler_propagate(state_vectors,state_vectors_prior,
                                                                             derivatives_prior,dt,titan,options)
        else:
            new_state_vectors, d_dt_state_vectors = explicit_adams_bashforth_n(titan.post_event_iter+1,state_vectors,state_vectors_prior,
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
    if not hasattr(titan,'rk_params'): recompute_params = True
    elif not np.shape(titan.rk_params[1])==np.shape(np.array(state_vectors).flatten()): recompute_params = True
    else: recompute_params = False
    if recompute_params: titan.rk_params = [titan.time, 
                                            np.array(state_vectors).flatten(),
                                            titan.time + dt*options.iters, 
                                            0.01*dt] # Small initial timestep to combat discontinuities at fragmentation
        
    if not hasattr(titan, 'rk_fun')   or recompute_params: titan.rk_fun=partial(state_equation,titan,options)
    if not hasattr(titan, 'rk_adapt') or recompute_params: titan.rk_adapt=algorithm(fun=titan.rk_fun,
                                                                                    t0=titan.rk_params[0],
                                                                                    y0=titan.rk_params[1],
                                                                                    t_bound=titan.rk_params[2],
                                                                                    first_step=titan.rk_params[3])

    if titan.rk_adapt.status == 'running':
        titan.rk_adapt.step()
    else: 
        print('Propagator concluded with status {} ({} function evaluations)'.format(titan.rk_adapt.status,titan.rk_adapt.nfev))
        titan.end_trigger = True
    titan.time = titan.rk_adapt.t
    return np.reshape(titan.rk_adapt.y,[-1,13]), None


def explicit_rk_N(N,state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    if N==5: N+=1 # Fifth order method actually has 6 fevals
    ## Butcher tableaus, can be added to just make sure you correctly set N (e.g. Heun's method etc.)
    tableaus =  {2:[[0.0],
                    [2/3,  2/3]],
                 3:[[0.0],
                    [0.5,  0.5],
                    [1.0, -1.0,   -2.0]],
                 4:[[0.0],
                    [0.5,  0.5],
                    [0.5,  0.0,    0.5],
                    [1.0,  0.0,    0.0,   1.0]],
                 5:[[0,0],
                    [1/3,  1/3],
                    [2/5,  4/25,  6/25],
                    [1,    1/4,     -3,  15/4],
                    [4/5,  2/25, 12/15,  2/15,  8/75]]
                }
    ## Bottom row of Butcher tableaus
    k_factors = {2:[1/4,    3/4],
                 3:[1/6,    2/3,  1/6],
                 4:[1/6,    1/3,  1/3,     1/6],
                 5:[23/192, 0,    125/192, 0,   -27/64, 125/192]}
    new_state_vectors = []
    k_n = []
    for i_k in range(N):
        k_state_vectors = np.array(state_vectors)
        for i_coeff in range(i_k): k_state_vectors+= tableaus[N][i_k][i_coeff]*dt*k_n[i_coeff]
        k_n.append(np.array(state_equation(titan,options,dt + tableaus[N][i_k][0]*dt,k_state_vectors)))

    new_state_vectors = np.array(state_vectors)

    for i_k in range(N): new_state_vectors+= k_factors[N][i_k] * dt * k_n[i_k]
    return new_state_vectors, k_n[0]

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