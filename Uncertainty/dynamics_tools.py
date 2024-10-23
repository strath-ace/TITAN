from Dynamics.frames import R_B_NED, R_W_B,R_B_W, R_W_NED
import numpy as np

def add_airspeed_velocity_NED(assembly, NED_vector):
    RBN = R_B_NED(assembly.pitch, assembly.roll,assembly.yaw)
    RWB = R_W_B(assembly.aoa,assembly.slip)
    RBW = R_B_W(assembly.aoa,assembly.slip)

    v = RWB.apply([assembly.trajectory.velocity,0,0])
    perturb = RBN.apply(NED_vector, inverse=True)
    v += perturb

    assembly.freestream.velocity = np.linalg.norm(RBW.apply(v))
    assembly.aoa = np.arctan2(v[2],v[0])
    assembly.slip = np.arcsin(v[1]/np.sqrt(v[0]**2 + v[1]**2 +  v[2]**2))
    return assembly

def apply_velocity_wind(trajectory, wind_vector):
    R = R_W_NED(trajectory.gamma, trajectory.chi)
    v = R.apply([trajectory.velocity,0,0])
    v += R.apply(wind_vector)
    trajectory.velocity = np.linalg.norm(v)
    trajectory.chi = np.arctan2(v[1], v[0])
    trajectory.gamma = -np.arcsin(v[2] / trajectory.velocity)

    return trajectory

def deterministicImpulse(cfg,burndata):
    from Configuration.configuration import Trajectory
    # First calculate magnitude and direction (retrograde) of deorbit burn
    # Tsiolkovsky rocket eq...
    v_e = burndata['specific_impulse']*9.80665
    mass_ratio= (burndata['propellant_mass']+burndata['vehicle_mass'])/burndata['vehicle_mass']
    delta_v = v_e * np.log(mass_ratio)

    velocity = cfg.getfloat('Trajectory','Velocity')
    gamma = cfg.getfloat('Trajectory','Flight_path_angle')
    chi = cfg.getfloat('Trajectory','Heading_angle')

    trajectory = Trajectory(velocity=velocity,gamma=np.radians(gamma),chi=np.radians(chi))

    manoeuvre = [-delta_v, 0, 0] # just retrograde for now

    trajectory = apply_velocity_wind(trajectory, manoeuvre)

    cfg.set('Trajectory', 'Velocity', str(trajectory.velocity))
    cfg.set('Trajectory', 'Flight_path_angle', str(np.degrees(trajectory.gamma)))
    cfg.set('Trajectory', 'Heading_angle', str(np.degrees(trajectory.chi)))

    print('Performed manoeuvre with a delta v of ', np.round(abs(velocity - trajectory.velocity), 4),
        'm/s actual (',np.round(delta_v, 4), ' m/s ideal)')

    print('State before || Velocity:', np.round(velocity,4), 'm/s | Flight Path Angle:',
        np.round(gamma,4), 'deg | Heading Angle:', np.round(chi,4),'deg')

    print('State after || Velocity:', np.round(trajectory.velocity,4), 'm/s | Flight Path Angle:',
        np.round(np.degrees(trajectory.gamma),4), 'deg | Heading Angle:',
        np.round(np.degrees(trajectory.chi),4),'deg')

