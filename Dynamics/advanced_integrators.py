from Dynamics import dynamics, frames, collision
from Aerothermo import aerothermo
from Forces import forces
import pymap3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.integrate import RK45
from functools import partial
from Output import output
import pyquaternion
from Freestream import gram
import copy
def explicit_euler_propagate(state_vectors,state_vectors_last,dt,titan,options):
    new_state_vectors = []
    d_dt_state_vectors = state_equation(titan,options,dt,state_vectors)
    for i_assem, _assembly in enumerate(titan.assembly):
        new_vector = []
        for element, d_dt_element in zip(state_vectors[i_assem],d_dt_state_vectors[i_assem]):
            new_vector.append(element+dt*d_dt_element)
        new_state_vectors.append(new_vector)
    return new_state_vectors

def explicit_bwd_diff_propagate(state_vectors,state_vectors_last,dt,titan,options):
    new_state_vectors = []
    d_dt_state_vectors = state_equation(titan,options,dt,state_vectors)
    for i_assem, _assembly in enumerate(titan.assembly):
        new_vector = []
        for element_last, d_dt_element in zip(state_vectors_last[i_assem],d_dt_state_vectors[i_assem]):
            new_vector.append(element_last+2*dt*d_dt_element)
        new_state_vectors.append(new_vector)
    return new_state_vectors

def rk45_wrapper(state_vectors,state_vectors_last,dt,titan,options):
    if not hasattr(titan,'rk45'):
        rk_45_func = partial(state_equation,titan,options)
        t0 = 0.0
        y_0 = np.array(state_vectors).flatten()
        t_bound = dt*options.iters
        first_step = dt
        titan.rk45 = RK45(rk_45_func,t0,y_0,t_bound,first_step=first_step)
    titan.rk45.step()
    titan.time = titan.rk45.t
    return np.reshape(titan.rk45.y,[-1,13])


integrator = {'euler': explicit_euler_propagate, 
              'RK45' : rk45_wrapper,
              'bwd'  : explicit_bwd_diff_propagate}

def propagate(titan, options):
    # Updates the state of all assemblies according to a propagator specified in the cfg file

    if options.collision.flag and len(titan.assembly)>1:
        flag_collision, __ = collision.check_collision(titan, options, 0)
        if flag_collision: collision.collision_physics(titan, options)
        #if flag_collision: collision.collision_physics_simultaneous(titan, options)

    # If we go to switch.py or su2.py, Because we call deepcopy() function, we need to rebuild
    #the collision mesh
    if options.collision.flag and options.fidelity.lower() in ['multi','high']:
        for assembly in titan.assembly: collision.generate_collision_mesh(assembly, options)
        collision.generate_collision_handler(titan, options)

    # Collect our state vectors
    current_state_vectors = []
    previous_state_vectors = []
    for _assembly in titan.assembly:
        if not hasattr(_assembly,'state_vector'): construct_state_vector(_assembly)
        current_state_vectors.append(_assembly.state_vector)

        if titan.iter==0:
            previous_state_vectors.append(_assembly.state_vector)
        else: previous_state_vectors.append(_assembly.state_vector_prev)

    for _assembly in titan.assembly: _assembly.state_vector_prev = _assembly.state_vector

    ## NB: Highly unsure if I've correctly applied this collision implementation, needs a deeper look
    time_step = options.dynamics.time_step
    if options.collision.flag and len(titan.assembly)>1:

        #Check collision for future time intervals with respect to current time-step velocity
        __, time_step = collision.check_collision(titan, options, time_step)
    

    new_state_vectors = integrator[options.dynamics.integrator](current_state_vectors,previous_state_vectors,time_step,titan,options)

    for _assembly, new_state_vector in zip(titan.assembly,new_state_vectors): 
        update_dynamic_attributes(_assembly,new_state_vector,options)
        _assembly.state_vector = new_state_vector
        _assembly.quaternion_canonical = _assembly.quaternion
    # Writes the output data before
    output.write_output_data(titan = titan, options = options)


    titan.time += time_step

        
    

def state_equation(titan,options,time,state_vectors):
    # This state equation will compute cartesian and angular position, velocity and acceleration based upon a prior state vector,
    # In order to do this we must accept assembly states as an input (state_vectors)
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

        omega_q = [angularDerivatives.droll,angularDerivatives.dpitch,angularDerivatives.dyaw,0.0]
        alpha_q = [angularDerivatives.ddroll,angularDerivatives.ddpitch,angularDerivatives.ddyaw,0.0]

        d_q  = 0.5 *  quaternion_mult(_assembly.quaternion,omega_q)
        dd_q = 0.5 * (quaternion_mult(alpha_q,_assembly.quaternion)+quaternion_mult(omega_q,d_q))

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
                                #    dd_q[0],
                                #    dd_q[1],
                                #    dd_q[2],
                                #    dd_q[3]])
    if reshape_flat: d_dt_state_vectors = np.array(d_dt_state_vectors).flatten()
    return d_dt_state_vectors

def update_dynamic_attributes(assembly,state_vector,options):
    # This function takes an ECEF/BODY state vector and applies it to all the necessary attributes of a TITAN assembly
    # This ensures the new dynamics code plays nicely with other parts of TITAN.

    assembly.position[0] = state_vector[0]
    assembly.position[1] = state_vector[1]
    assembly.position[2] = state_vector[2]

    assembly.velocity[0] = state_vector[3]
    assembly.velocity[1] = state_vector[4]
    assembly.velocity[2] = state_vector[5]

    assembly.quaternion[0] = state_vector[6]
    assembly.quaternion[1] = state_vector[7]
    assembly.quaternion[2] = state_vector[8]
    assembly.quaternion[3] = state_vector[9]

    assembly.roll_vel = state_vector[10]
    assembly.pitch_vel = state_vector[11]
    assembly.yaw_vel = state_vector[12]
    
    # assembly.q_dot[0] = state_vector[10]
    # assembly.q_dot[1] = state_vector[11]
    # assembly.q_dot[2] = state_vector[12]
    # assembly.q_dot[3] = state_vector[13]

    omega = 2*quaternion_mult(quaternion_conjugate(assembly.quaternion),assembly.q_dot)

    # assembly.roll_vel = omega[0]
    # assembly.pitch_vel = omega[1]
    # assembly.yaw_vel = omega[2]
    # delta_rot = Rot.from_euler('ZYX',[delta_yaw,delta_pitch,delta_roll],degrees=False)
    # rot = Rot.from_quat(assembly.quaternion_canonical)
    # #delta_q = Rot.from_euler('ZYX',[delta_yaw,delta_pitch,delta_roll],degrees=False).as_quat()
    # # py_q = pyquaternion.Quaternion(q[3],q[0],q[1],q[2])
    # # py_q.integrate([delta_roll,delta_pitch,delta_yaw],1.0)
    # assembly.quaternion = (delta_rot * rot).as_quat()
    # #assembly.quaternion = np.append(py_q.vector, py_q.real)
    # #assembly.quaternion = quaternion_mult(delta_q,q)
    #assembly.quaternion = Rot.from_euler('ZYX',[assembly.yaw,assembly.pitch,assembly.roll],degrees=False).as_quat()
    # Get the new latitude, longitude and altitude
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
    # assembly.state_vector[10]  = assembly.q_dot[0]
    # assembly.state_vector[11]  = assembly.q_dot[1]
    # assembly.state_vector[12]  = assembly.q_dot[2]
    #assembly.state_vector[13]  = assembly.q_dot[3]

def quaternion_mult(q1,q2):
    return np.array([q1[3]*q2[0]+q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1],
            q1[3]*q2[1]+q1[1]*q2[3]-q1[0]*q2[2]+q1[2]*q2[0],
            q1[3]*q2[2]+q1[2]*q2[3]+q1[0]*q2[1]-q1[1]*q2[0],
            q1[3]*q2[3]-q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]])

def quaternion_conjugate(q): return np.array([-q[0],-q[1],-q[2],q[3]])

def quaternion_normalize(q):
    norm = np.linalg.norm(q)
    return q/norm