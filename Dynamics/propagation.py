from Dynamics import dynamics, frames, collision
from Aerothermo import aerothermo
from Forces import forces
import pymap3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.special import erf
from scipy import integrate
from scipy.stats import uniform_direction
from functools import partial
from Output import output
from warnings import warn
from copy import copy

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

    #TODO: Manage collision handling properly
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
        angle_names = ['roll','pitch','yaw']
        if not hasattr(_assembly,'unmodded_angles'):
            _assembly.unmodded_angles  = np.array([getattr(_assembly,angle) for angle in angle_names])
            
        update_dynamic_attributes(_assembly,new_state_vectors[i_assem],options)
        _assembly.state_vector = new_state_vectors[i_assem]
        for i_angle, angle in enumerate(angle_names): 
            _assembly.unmodded_angles[i_angle]+=time_step*_assembly.state_vector[10+i_angle]
        from Output.output import write_to_series
        write_to_series([np.hstack([titan.time,_assembly.unmodded_angles])],[['Time','Roll','Pitch','Yaw']],options.output_folder+'/Angles_{}.csv'.format(_assembly.id))
    # Writes the output data
    output.write_output_data(titan = titan, options = options)

    # Increment time step
    if hasattr(titan,'rk_params'): time_step = titan.delta_t
    else: titan.delta_t = time_step
    
    titan.time += time_step
    

def state_equation(titan,options,time,state_vectors):
    if not hasattr(titan,'nfeval'): titan.nfeval = 1
    else: titan.nfeval +=1
    # This state equation will for each assembly compute the rate of change of a state vector (at that state),
    
    # State vectors can be passed flattened, need to account for this
    state_vectors = np.array(state_vectors)
    reshape_flat = False
    if len(state_vectors.shape)<2:
        reshape_flat = True
        state_vectors = np.reshape(state_vectors,[-1,13])

    # n_states = np.shape(state_vectors)[0]
    # from scipy.stats import uniform_direction
    # quat = uniform_direction(4).rvs(n_states)
    # state_vectors[:,6:10] = quat

    # First we communicate the state vector to assembly attributes
    for _assembly, state_vector in zip(titan.assembly,state_vectors):
        update_dynamic_attributes(_assembly,state_vector,options)
    
    # Then business as usual for computing forces...
    
    aerothermo.compute_aerothermo(titan, options)
    aero_states = [copy(_assembly.aerothermo) for _assembly in titan.assembly]
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
    return d_dt_state_vectors, aero_states

def cartesian_state_equation(titan,options,time,position_state_vectors):
    # This state equation will for each assembly compute the rate of change of a 3DOF state vector (at that state),
    
    # State vectors can be passed flattened, need to account for this
    position_state_vectors = np.array(position_state_vectors)
    reshape_flat = False
    if len(position_state_vectors.shape)<2: 
        reshape_flat = True
        position_state_vectors = np.reshape(position_state_vectors,[-1,6])

    # First we communicate the state vector to assembly attributes
    for _assembly, position_state_vector in zip(titan.assembly,position_state_vectors):
        update_state = _assembly.state_vector
        update_state[:6] = position_state_vector
        update_dynamic_attributes(_assembly,update_state,options)
    
    # Then business as usual for computing forces...
    aerothermo.compute_aerothermo(titan, options)

    forces.compute_aerodynamic_forces(titan, options)
    forces.compute_aerodynamic_moments(titan, options)

    # Then determine the necessary derivatives to return the state vector(s)
    d_dt_state_vectors = []
    for _assembly in titan.assembly:
        cartesianDerivatives = dynamics.compute_cartesian_derivatives(_assembly, options)

        d_dt_state_vectors.append([cartesianDerivatives.dx,
                                   cartesianDerivatives.dy,
                                   cartesianDerivatives.dz,
                                   cartesianDerivatives.du,
                                   cartesianDerivatives.dv,
                                   cartesianDerivatives.dw])
        
    if reshape_flat: d_dt_state_vectors = np.array(d_dt_state_vectors).flatten()
    return d_dt_state_vectors

def angular_state_equation(titan,options,time,angular_state_vectors):
    # This state equation will for each assembly compute the rate of change of a Body-Frame state vector (at that state),
    
    # State vectors can be passed flattened, need to account for this
    angular_state_vectors = np.array(angular_state_vectors)
    reshape_flat = False
    if len(angular_state_vectors.shape)<2: 
        reshape_flat = True
        angular_state_vectors = np.reshape(angular_state_vectors,[-1,7])

    # First we communicate the state vector to assembly attributes
    for _assembly, angular_state_vector in zip(titan.assembly,angular_state_vectors):
        update_state = _assembly.state_vector
        update_state[6:] = angular_state_vector
        update_dynamic_attributes(_assembly,update_state,options)
    
    # Then business as usual for computing forces...
    aerothermo.compute_aerothermo(titan, options)

    forces.compute_aerodynamic_forces(titan, options)
    forces.compute_aerodynamic_moments(titan, options)

    # Then determine the necessary derivatives to return the state vector(s)
    d_dt_state_vectors = []
    for _assembly in titan.assembly:
        angularDerivatives = dynamics.compute_angular_derivatives(_assembly)

        # Use quaternion derivative equation 
        omega_q = [angularDerivatives.droll,angularDerivatives.dpitch,angularDerivatives.dyaw,0.0]
        d_q  = 0.5 *  quaternion_mult(_assembly.quaternion,omega_q)

        d_dt_state_vectors.append([d_q[0],
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
    if 'area' in choice:
        options.dynamics.n_derivs_to_hold = 1
        return partial(proj_area_adapt_wrapper,23)
    if 'legacy' in choice:
        print('...legacy propagation, note these methods are deprecated.')
        return None
    if 'euler' in choice: 
        print('...Euler propagation')
        return explicit_euler
    if 'bwd' in choice and not 'legacy' in choice: 
        options.dynamics.n_states_to_hold = 1
        print('...backward difference propagation')
        return explicit_bwd_diff
    if 'adapt' in choice:
        n_rk = int(choice.partition('rk')[-1][0])
        n_ab = int(choice.partition('ab')[-1][0])
        if n_ab>5:
            print('NB: Only Adams-Bashforth methods of order 2-5 are implemented! Setting order to 5 to use...')
            n_ab = 5
        if n_rk>5:
            print('NB: Only Runge-Kutta methods of order 2-5 are implemented! Setting order to 5 to use...')
            n_rk = 5
        options.dynamics.n_derivs_to_hold = n_ab - 1
        print('...adaptive decoupling with AB{} and RK{}'.format(n_ab,n_rk))
        return partial(adaptive_integrator_selector,n_ab,n_rk)
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
    ## - euler : Euler method
    ## - bwd   : Backward difference, using information from 1 previous time step (2nd order)
    ## - AB[N] : Adams-Bashford Nth order for N = 2-5 (AB2,...,AB5), using information from N-1 previous time step(s)
    ## - RK[N] : Runge-Kutte Nth order for N = 2-5 (RK2,...,RK5), using information from N flow solves per time step
          
    ## Adaptive integration methods  
    ## - adapt_AB[N_AB]_RK[N_RK] : High angular accelerations trigger a switch from AB[N] to a determined RK method up to order N

    ## Adaptive time-step methods (via scipy.integrate)
    ## - RK23    : Time stepping according to 3rd order Runge-Kutta with 2nd order error control
    ## - RK45    : Time stepping according to 5th order Runge-Kutta with 4th order error control
    ## - DOP853  : The DOP8(5,3) adaptive algorithm 
    ## See docs.scipy.org/doc/scipy/reference/integrate.html for more info
    
    ## Legacy Dynamics Implementation (deprecated)
    ## legacy_euler : Prior Euler method
    ## legacy_bwd   : Prior backward difference method for position updates
    ''')
    raise Exception('Invalid propagator')
#############################################################################################################################################
#############################################################################################################################################
################################################## INTEGRATOR PARAMETERS  ###################################################################
#############################################################################################################################################
#############################################################################################################################################

## Butcher tableaus, can be added to just make sure you correctly set N (e.g. Heun's method etc.)
RK_tableaus =  {'1': [[0.0]],
                '2': [[0.0],
                      [2/3,  2/3]],
                '3': [[0.0],
                      [0.5,  0.5],
                      [1.0, -1.0,   -2.0]],
	            '23':[[0.5,  0.5],
                      [0.75, 0,      0.75],
                      [1.0,  2/9,    1/3,   4/9]],
                '4': [[0.0],
                      [0.5,  0.5],
                      [0.5,  0.0,    0.5],
                      [1.0,  0.0,    0.0,   1.0]],
                }
## Bottom row of Butcher tableaus
RK_k_factors = {'2' :[1/4,    3/4],
                '23':[7/24,   1/4,  1/3,     1/8],
                '3' :[1/6,    2/3,  1/6],
                '4' :[1/6,    1/3,  1/3,     1/6]
                }
## Number of fevals for each RK model (written like this for extensibility)
RK_N_actual = {'2':2, '23':3, '3':3, '4':4}
## Adaptive error coefficients for RK models
RK_Error = {'23':[2/9, 1/3, 4/9,0],
            '45':[]}
## Adaptive error orders for RK models
RK_order = {'23':2}
## Adaptive tolerances
Atol = 1e-3
Rtol = 0.1
## Adams - Bashforth Coefficients
AB_coeffs = {'1' : [       1.0],
             '2' : [  -0.5,1.5], 
             '3' : [   5.0/12.0,    -16.0/12.0,    23.0/12.0], 
             '4' : [  -9.0/24.0,     37.0/24.0,   -59.0/24.0,     55.0/24.0],
             '5' : [251.0/720.0, -1274.0/720.0, 2616.0/720.0, -2774.0/720.0, 1901.0/720.0]}

#############################################################################################################################################
#############################################################################################################################################
###########################################################  ALGORITHMS  ####################################################################
#############################################################################################################################################
#############################################################################################################################################
def explicit_euler(state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    new_state_vectors = []
    d_dt_state_vectors, aero_states = state_equation(titan,options,dt,state_vectors)
    for i_assem, _assembly in enumerate(titan.assembly):
        new_vector = []
        new_vector = np.array(state_vectors[i_assem])+dt*np.array(d_dt_state_vectors[i_assem])
        new_state_vectors.append(new_vector)
        _assembly.aerothermo = copy(aero_states[i_assem])
    return new_state_vectors, d_dt_state_vectors

def explicit_bwd_diff(state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    new_state_vectors = []
    if titan.post_event_iter==0:
        new_state_vectors, d_dt_state_vectors = explicit_euler(state_vectors,state_vectors_prior,
                                                                         derivatives_prior,dt,titan,options)
    else:
        d_dt_state_vectors, aero_states = state_equation(titan,options,dt,state_vectors)
        for i_assem, _assembly in enumerate(titan.assembly):
            new_vector = np.array(state_vectors_prior[0][i_assem])+2*dt*np.array(d_dt_state_vectors[i_assem])
            new_state_vectors.append(new_vector)
            _assembly.aerothermo = copy(aero_states[i_assem])
    return new_state_vectors, d_dt_state_vectors

def explicit_adams_bashforth_n(N,state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    new_state_vectors = []

    d_dt_state_vectors, aero_states = state_equation(titan,options,dt,state_vectors)
    derivatives_prior.append(d_dt_state_vectors)
    N_derivs = len(derivatives_prior) if len(derivatives_prior) < N else N

    for i_assem, _assembly in enumerate(titan.assembly):
        new_vector = np.array(state_vectors[i_assem])
        for i_prior in range(N_derivs):
            new_vector += dt*AB_coeffs[str(N_derivs)][i_prior] * np.array(derivatives_prior[i_prior][i_assem])
        new_state_vectors.append(new_vector)
        _assembly.aerothermo = copy(aero_states[i_assem])
    return new_state_vectors, d_dt_state_vectors

def explicit_rk_adapt_wrapper(algorithm, state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    if not hasattr(titan,'rk_params'): recompute_params = True
    elif not np.shape(titan.rk_params['state'])==np.shape(np.array(state_vectors).flatten()): recompute_params = True
    else: recompute_params = False
    if recompute_params: 
        if titan.time>0: titan.time-=dt
        titan.rk_params = {'time'    : titan.time, 
                           'state'   : np.array(state_vectors).flatten(),
                           't_end'   : titan.time + dt*options.iters, 
                           't_first' : 0.01*dt,  # Small initial timestep to combat discontinuities at fragmentation
                           't_max'   : dt}

    if not hasattr(titan, 'rk_fun')   or recompute_params: 
        def rk_wrapper(titan,options,time,vector): return state_equation(titan,options,time,vector)[0]
        titan.rk_fun=partial(rk_wrapper,titan,options)
    if not hasattr(titan, 'rk_adapt') or recompute_params: titan.rk_adapt=algorithm(fun=titan.rk_fun,
                                                                                    t0=titan.rk_params['t_first'],
                                                                                    y0=titan.rk_params['state'],
                                                                                    t_bound=titan.rk_params['t_end'],
                                                                                    first_step=titan.rk_params['t_first'],
                                                                                    max_step=titan.rk_params['t_max'])

    if titan.rk_adapt.status == 'running':
        titan.rk_adapt.step()
    else: 
        print('Propagator concluded with status {} ({} function evaluations)'.format(titan.rk_adapt.status,titan.rk_adapt.nfev))
        titan.end_trigger = True
    titan.delta_t = titan.rk_adapt.step_size
    return np.reshape(titan.rk_adapt.y,[-1,13]), None

def proj_area_adapt_wrapper(N, state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    if titan.post_event_iter==0:
        new_state_vectors, d_dt_state_vectors = explicit_euler(state_vectors,state_vectors_prior,
                                                                         derivatives_prior,dt,titan,options)
        titan.rk_params = 'Pip'
        return new_state_vectors, d_dt_state_vectors
    state_vectors = np.array(state_vectors)
    ## Should first recover the current timestep projected area from aerothermo class
    area_divergence = []
    for _assembly in titan.assembly:
        if not hasattr(_assembly.aerothermo,'proj_area'): _assembly.aerothermo.proj_area = 1
        area_divergence.append(1/_assembly.aerothermo.proj_area)
    ## Then we want to do get our derivatives based upon some RK model
    dt = 0.001*titan.delta_t if titan.iter==0 else titan.delta_t
    k_n = [np.array(derivatives_prior[-1])] ## Via FSAL
    for i_k in range(RK_N_actual[str(N)]):
        k_state_vectors = np.array(state_vectors)
        for i_coeff in range(i_k): k_state_vectors += RK_tableaus[str(N)][i_k][i_coeff] * dt * k_n[i_coeff]
        if i_k==0:
            d_dt_state_vectors, aero_states = state_equation(titan, options, dt + RK_tableaus[str(N)][i_k][0] * dt,k_state_vectors)
        else: d_dt_state_vectors, final_states = state_equation(titan, options, dt + RK_tableaus[str(N)][i_k][0] * dt,k_state_vectors)
        k_n.append(np.array(d_dt_state_vectors))

    derivs_high_order = np.zeros_like(derivatives_prior[-1])
    derivs_low_order  = np.zeros_like(derivatives_prior[-1])
    for i_factor, factor in enumerate(RK_k_factors[str(N)]):
        derivs_high_order += factor*k_n[i_factor]
        derivs_low_order  += RK_Error[str(N)][i_factor]*k_n[i_factor]
    new_state_vectors = state_vectors+dt*derivs_high_order
    new_derivatives = derivs_high_order

    ## Afterwards we can compare the errors of 3dof and 6dof propagation
    error_6DoF = np.max(abs(derivs_high_order-derivs_low_order),axis=0)
    error_3DoF = np.max(abs(derivs_high_order[:,:6]-derivs_low_order[:,:6]),axis=0)

    magnitude_previous = np.max(abs(state_vectors),axis=0)
    magnitude_current = np.max(abs(new_state_vectors),axis=0)
    
    tol_6dof = Atol+Rtol*dt*np.max([magnitude_current,magnitude_previous],axis=0)
    tol_3dof = Atol+Rtol*dt*np.max([magnitude_previous[:6],magnitude_current[:6]],axis=0)

    scaled_E_6dof = np.linalg.norm(error_6DoF/tol_6dof)
    scaled_E_3dof = np.linalg.norm(error_3DoF/tol_3dof)
    print('Errors of {} 6Dof, {} 3Dof'.format(scaled_E_6dof,scaled_E_3dof))
    ## Finally we can recover our candidate step sizes
    factor_3dof = 0.9*(1/scaled_E_6dof)**(1/(RK_order[str(N)]+1))
    factor_6dof = 0.9*(1/scaled_E_3dof)**(1/(RK_order[str(N)]+1))
    # h_6dof = dt*0.9*(1/scaled_E_6dof)**(1/(RK_order[str(N)]+1))
    # h_3dof = dt*0.9*(1/scaled_E_3dof)**(1/(RK_order[str(N)]+1))
    print('Factors are 6Dof = {}, 3Dof = {}'.format(factor_6dof,factor_3dof))
    ## Now to decide which one to use, compare the rate of change of projected areas...
    high_factor = max(factor_6dof,factor_3dof)
    low_factor = min(factor_6dof,factor_3dof)
    for i_area, aero in enumerate(final_states):
        area_divergence[i_area]=abs(area_divergence[i_area]*aero.proj_area-1)
    ## "Area divergence" represents the level of change in projected area the flow "sees", 0 is no change, higher values are bigger changes
    print('AD of {}'.format(area_divergence))
    AD = max(area_divergence)
    threshold = 0.0
    k=5.0
    ## For a high area divergence we need to accurately resolve the attitude motion
    bridged_h = high_factor*np.exp(k*(threshold-AD))
    print('Bridged dt of {}'.format(bridged_h))
    factor = min(high_factor,max(bridged_h,low_factor,0.2),10)
    titan.delta_t = min(titan.delta_t*factor,options.dynamics.time_step)
    print('Factor {} gives a dt of {}'.format(factor,titan.delta_t))
    new_state_vectors = state_vectors+titan.delta_t*derivs_high_order
    #if AD<threshold: new_state_vectors[:,:6] = state_vectors[:,:6]+titan.delta_t*k_n[0][:,:6]
    for i_assem, _assembly in enumerate(titan.assembly): _assembly.aerothermo = aero_states[i_assem]
    return new_state_vectors, new_derivatives


def explicit_rk_N(N,state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    new_state_vectors = []
    k_n = []
    for i_k in range(RK_N_actual[str(N)]):
        k_state_vectors = np.array(state_vectors)
        for i_coeff in range(i_k): k_state_vectors += RK_tableaus[str(N)][i_k][i_coeff] * dt * k_n[i_coeff]
        if i_k==0:
            d_dt_state_vectors, aero_states = state_equation(titan, options, dt + RK_tableaus[str(N)][i_k][0] * dt,k_state_vectors)
        else: d_dt_state_vectors, _ = state_equation(titan, options, dt + RK_tableaus[str(N)][i_k][0] * dt,k_state_vectors)
        k_n.append(np.array(d_dt_state_vectors))
    new_state_vectors = np.array(state_vectors)

    for i_k in range(N): new_state_vectors+= RK_k_factors[str(N)][i_k] * dt * k_n[i_k]
    for i_assem, _assembly in enumerate(titan.assembly): _assembly.aerothermo = aero_states[i_assem]
    return new_state_vectors, k_n[0]

def adaptive_integrator_selector(N_AB, N_RK,state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    ## This algorithm will use Adams Bashforth unless extreme angular accelerations are detected, in that case we use RK
    ## A somewhat experimental method that shouldn't be any less accurate than either of the methods supplied
    if titan.post_event_iter==0: return explicit_adams_bashforth_n(N_AB,state_vectors,state_vectors_prior,
                                                                   derivatives_prior,dt,titan,options)

    spins = []
    for _assembly in titan.assembly:
        angularDerivatives = dynamics.compute_angular_derivatives(_assembly)
        spins.append([angularDerivatives.ddroll,angularDerivatives.ddpitch,angularDerivatives.ddyaw])
    spin_max = np.max(np.abs(spins))

    AB_new_state_vectors, AB_d_dt_state_vectors = explicit_adams_bashforth_n(N_AB,state_vectors,state_vectors_prior,
                                                                             derivatives_prior,dt,titan,options)
    RK_new_state_vectors = np.zeros_like(AB_new_state_vectors)
    RK_d_dt_state_vectors = np.zeros_like(AB_d_dt_state_vectors)

    
    bridging = [(erf(150*(options.dynamics.acceleration_threshold-spin_max))+1)/2]
    bridging.append(1-bridging[0])
    thresholds = np.array([i/(N_RK-1) for i in range(N_RK-1)])

    if spin_max > options.dynamics.tumbling_criterion and options.dynamics.tumbling_criterion > 0:
        print('Performing random tumbling!!')
        angles = uniform_direction(4).rvs()
        accelerations = [0.0,0.0,0.0]
        new_state_vectors = AB_new_state_vectors
        d_dt_state_vectors = AB_d_dt_state_vectors
        new_state_vectors[6:10] = angles
        new_state_vectors[10:] = accelerations
        d_dt_state_vectors[6:] = np.zeros(7)
        return new_state_vectors, d_dt_state_vectors
    elif bridging[1]>0:
        print('Switched to RK{}'.format(np.where(bridging[1]>=thresholds)[0][-1]+2))
        RK_new_state_vectors, RK_d_dt_state_vectors = explicit_rk_N(np.where(bridging[1]>=thresholds)[0][-1]+2,
                                                                    state_vectors,state_vectors_prior,
                                                                    derivatives_prior,dt,titan,options)
    
    new_state_vectors  = bridging[0]*np.array(AB_new_state_vectors)  + bridging[1] * np.array(RK_new_state_vectors)
    d_dt_state_vectors = bridging[0]*np.array(AB_d_dt_state_vectors) + bridging[1] * np.array(RK_d_dt_state_vectors)
    return new_state_vectors, d_dt_state_vectors
    


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