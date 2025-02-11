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
from Dynamics import dynamics, frames, collision
from Aerothermo import aerothermo
from Forces import forces
import pymap3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from Output import output
import pyquaternion
from Freestream import gram
from Model import drag_model
import copy

def compute_Euler(titan, options):
    """
    Euler integration

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options
    """

    if options.collision.flag and len(titan.assembly)>1:
        flag_collision, __ = collision.check_collision(titan, options, 0)
        if flag_collision: collision.collision_physics(titan, options)
        #if flag_collision: collision.collision_physics_simultaneous(titan, options)

    aerothermo.compute_aerothermo(titan, options)  

    # If we go to switch.py or su2.py, Because we call deepcopy() function, we need to rebuild
    #the collision mesh
    if options.collision.flag and options.fidelity.lower() in ['multi','high']:
        for assembly in titan.assembly: collision.generate_collision_mesh(assembly, options)
        collision.generate_collision_handler(titan, options)

    forces.compute_aerodynamic_forces(titan, options)
    forces.compute_aerodynamic_moments(titan, options)

    # Writes the output data before
    output.write_output_data(titan = titan, options = options)

    time_step = options.dynamics.time_step
    if options.collision.flag and len(titan.assembly)>1:

        #Check collision for future time intervals with respect to current time-step velocity
        __, time_step = collision.check_collision(titan, options, time_step)
    
    titan.time += time_step
    titan.time = round(titan.time, 5)

    # Loop over the assemblies and compute the dericatives
    for assembly in titan.assembly:
        angularDerivatives = dynamics.compute_angular_derivatives(assembly)
        cartesianDerivatives = dynamics.compute_cartesian_derivatives(assembly, options)     
        update_position_cartesian(assembly, cartesianDerivatives, angularDerivatives, options, time_step, titan.iter)
        
def update_position_cartesian(assembly, cartesianDerivatives, angularDerivatives, options, time_step, iter_num):
    """
    Update position and attitude of the assembly

    Parameters
    ----------
    assembly: Assembly
        Object of class Assembly
    cartesianDerivatives: DerivativesCartesian
        Object of class DerivativesCartesian
    angularDerivatives: DerivativesAngle
        Object of class DerivativesAngle
    options: Options
        Object of class Options
    """

    dt = time_step
    
    if iter_num == 0 or options.dynamics.integrator == 'legacy_euler':

        assembly.position[0] += dt*cartesianDerivatives.dx
        assembly.position[1] += dt*cartesianDerivatives.dy
        assembly.position[2] += dt*cartesianDerivatives.dz

        assembly.velocity[0] += dt*cartesianDerivatives.du
        assembly.velocity[1] += dt*cartesianDerivatives.dv
        assembly.velocity[2] += dt*cartesianDerivatives.dw

    elif options.dynamics.integrator == 'legacy_bwd':

        px = assembly.position_nlast[0] +  2*dt*cartesianDerivatives.dx
        py = assembly.position_nlast[1] +  2*dt*cartesianDerivatives.dy
        pz = assembly.position_nlast[2] +  2*dt*cartesianDerivatives.dz
        vx = assembly.velocity_nlast[0] + 2*dt*cartesianDerivatives.du
        vy = assembly.velocity_nlast[1] + 2*dt*cartesianDerivatives.dv
        vz = assembly.velocity_nlast[2] + 2*dt*cartesianDerivatives.dw

        assembly.position[0] = px
        assembly.position[1] = py
        assembly.position[2] = pz
        assembly.velocity[0] = vx
        assembly.velocity[1] = vy
        assembly.velocity[2] = vz

    assembly.position_nlast = copy.deepcopy(assembly.position)
    assembly.velocity_nlast = copy.deepcopy(assembly.velocity)

    assembly.distance_travelled += np.sqrt(dt*cartesianDerivatives.dx*dt*cartesianDerivatives.dx+dt*cartesianDerivatives.dy*dt*cartesianDerivatives.dy+dt*cartesianDerivatives.dz*dt*cartesianDerivatives.dz) 

    q = assembly.quaternion

    # Get the new latitude, longitude and altitude
    [latitude, longitude, altitude] = pymap3d.ecef2geodetic(assembly.position[0], assembly.position[1], assembly.position[2],
                                      ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                      deg = False);

    assembly.trajectory.latitude = latitude
    assembly.trajectory.longitude = longitude
    assembly.trajectory.altitude = altitude

    print('Altitude:', altitude)

    [vEast, vNorth, vUp] = pymap3d.uvw2enu(assembly.velocity[0], assembly.velocity[1], assembly.velocity[2], latitude, longitude, deg=False)

    gamma = np.arcsin(np.dot(assembly.position, assembly.velocity)/(np.linalg.norm(assembly.position)*np.linalg.norm(assembly.velocity)))
    assembly.trajectory.chi = np.arctan2(vEast,vNorth)
    
    R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)

    norm = np.linalg.norm(assembly.quaternion)

    #Should it be like this??
    R_B_NED_quat = (R_NED_ECEF).inv()*Rot.from_quat(assembly.quaternion)
    [yaw,pitch,roll] = R_B_NED_quat.as_euler('ZYX')

    assembly.yaw = yaw
    assembly.pitch = pitch
    assembly.roll = roll

    #ECEF_2_B
    [Vx_B, Vy_B, Vz_B] =  Rot.from_quat(assembly.quaternion).inv().apply(assembly.velocity)
    assembly.trajectory.velocity = np.linalg.norm([Vx_B, Vy_B, Vz_B])

    assembly.aoa = np.arctan2(Vz_B,Vx_B)
    assembly.slip = np.arcsin(Vy_B/np.sqrt(Vx_B**2 + Vy_B**2 +  Vz_B**2))

    # Integrates the quaternion with respect to the angular velocities and the time-steo
    py_quat = pyquaternion.Quaternion(q[3],q[0],q[1],q[2])
    
    py_quat.integrate([angularDerivatives.droll, angularDerivatives.dpitch,angularDerivatives.dyaw], dt)
    assembly.quaternion = np.append(py_quat.vector, py_quat.real)

    assembly.roll_vel  += dt*angularDerivatives.ddroll  #christie: p,q,r or d(euler)??
    assembly.pitch_vel += dt*angularDerivatives.ddpitch
    assembly.yaw_vel   += dt*angularDerivatives.ddyaw

    #Limiting the angular velocity to 100 rad/s.

    #christie: not good
    if assembly.roll_vel > 100:  assembly.roll_vel = 100
    if assembly.roll_vel < -100: assembly.roll_vel = -100

    if assembly.pitch_vel > 100:  assembly.pitch_vel = 100
    if assembly.pitch_vel < -100: assembly.pitch_vel = -100

    if assembly.yaw_vel > 100:  assembly.yaw_vel = 100
    if assembly.yaw_vel < -100: assembly.yaw_vel = -100

    # angle of attack is zero if a Drag model is specified, so pitch needs to follow the flght path angle
    if options.vehicle:# and options.vehicle.Cd:
        assembly.roll_vel  = 0
        assembly.pitch_vel = (gamma-assembly.pitch)/dt
        assembly.yaw_vel   = 0
    
    assembly.trajectory.gamma = gamma