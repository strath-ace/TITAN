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
from Dynamics import dynamics, frames
from Aerothermo import aerothermo
from Forces import forces
import pymap3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from Output import output
import pyquaternion
from Freestream import gram
from Model import drag_model

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

    aerothermo.compute_aerothermo(titan, options)

    forces.compute_aerodynamic_forces(titan, options)
    forces.compute_aerodynamic_moments(titan, options)

    # Writes the output data before
    output.write_output_data(titan = titan, options = options)

    # Loop over the assemblies and compute the dericatives
    for assembly in titan.assembly:
        angularDerivatives = dynamics.compute_angular_derivatives(assembly)
        cartesianDerivatives = dynamics.compute_cartesian_derivatives(assembly, options)
        update_position_cartesian(assembly, cartesianDerivatives, angularDerivatives, options)
        
def update_position_cartesian(assembly, cartesianDerivatives, angularDerivatives, options):
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

    dt = options.dynamics.time_step

    assembly.position[0] += dt*cartesianDerivatives.dx
    assembly.position[1] += dt*cartesianDerivatives.dy
    assembly.position[2] += dt*cartesianDerivatives.dz
    assembly.velocity[0] += dt*cartesianDerivatives.du
    assembly.velocity[1] += dt*cartesianDerivatives.dv
    assembly.velocity[2] += dt*cartesianDerivatives.dw

    q = assembly.quaternion

    # Get the new latitude, longitude and altitude
    [latitude, longitude, altitude] = pymap3d.ecef2geodetic(assembly.position[0], assembly.position[1], assembly.position[2],
                                      ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                      deg = False);

    assembly.trajectory.latitude = latitude
    assembly.trajectory.longitude = longitude
    assembly.trajectory.altitude = altitude

    [vEast, vNorth, vUp] = pymap3d.uvw2enu(assembly.velocity[0], assembly.velocity[1], assembly.velocity[2], latitude, longitude, deg=False)

    gamma = np.arcsin(np.dot(assembly.position, assembly.velocity)/(np.linalg.norm(assembly.position)*np.linalg.norm(assembly.velocity)))
    assembly.trajectory.chi = np.arctan2(vEast,vNorth)
    
    R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)

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

    assembly.roll_vel  += dt*angularDerivatives.ddroll
    assembly.pitch_vel += dt*angularDerivatives.ddpitch
    assembly.yaw_vel   += dt*angularDerivatives.ddyaw

    # angle of attack is zero if a Drag model is specified, so pitch needs to follow the flght path angle
    
    if options.vehicle:# and options.vehicle.Cd:
        assembly.roll_vel  = 0
        assembly.pitch_vel = (gamma-assembly.pitch)/dt
        assembly.yaw_vel   = 0
    
    assembly.trajectory.gamma = gamma