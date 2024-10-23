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
from Dynamics import euler, frames
from Freestream import gram
import pymap3d
from scipy.spatial.transform import Rotation as Rot
import pyquaternion

class DerivativesPointMass():
    def __init__(self, dh = 0, dv = 0, dchi = 0, dgamma = 0, dlat = 0, dlon = 0):
        self.dh = dh
        self.dv = dv
        self.dchi = dchi
        self.dgamma = dgamma
        self.dlat = dlat
        self.dlon = dlon

class DerivativesCartesian():
    """ Class DerivativesCartesian
    
        A class to store the derivatives information of position and velocity in the cartesian (ECEF) frame
    """
    def __init__(self, dx = 0, dy = 0, dz = 0, du = 0, dv = 0, dw = 0):

        #: [float] Derivative of the X-position
        self.dx = dx

        #: [float] Derivative of the Y-position
        self.dy = dy

        #: [float] Derivative of the Z-position
        self.dz = dz

        #: [float] Derivative of the X-velocity
        self.du = du

        #: [float] Derivative of the Y-velocity
        self.dv = dv

        #: [float] Derivative of the Z-velocity
        self.dw = dw

class DerivativesAngle():
    """ Class DerivativesAngle
    
        A class to store the derivatives information regarding the angular dynamics in the body frame
    """

    def __init__(self, droll = 0, dpitch = 0, dyaw = 0, ddroll = 0, ddpitch = 0, ddyaw = 0):

        #: [float] Derivative of the roll angle
        self.droll   = droll

        #: [float] Derivative of the pitch angle
        self.dpitch  = dpitch

        #: [float] Derivative of the yaw angle
        self.dyaw    = dyaw

        #: [float] Second derivative of the roll angle
        self.ddroll  = ddroll

        #: [float] Second derivative of the pitch angle
        self.ddpitch = ddpitch

        #: [float] Second derivative of the yaw angle
        self.ddyaw   = ddyaw

"""
def compute_gravity(h):
    
    Gravity function

    This function computes the gravity (radial only) with respect to the altitude

    Parameters
    ----------
    h: float
        Altitude in meters

    Returns
    -------
    gr: float
        Radial gravity value
    gt: float
        Tagential grvity value
    


    #TODO change to a model folder
    g0 = 9.80665
    rE = 6375253

    gr = g0*(rE/(h+rE))*(rE/(h+rE))
    gt = 0

    return gr,gt
"""

def compute_quaternion(assembly):
    """
    Computation of the quaternion

    This function computes the quaternion value of the body frame with respect to the ECEF frame
    The quaternion will give the rotation matrix that will allow to pass from Body to ECEF

    Parameters
    ----------
    assembly: Assembly
        Object of Assembly class
    """

    #Fix pitch and yaw values according to flight path angle, heading angle, slip and angle of attack
    assembly.pitch= assembly.trajectory.gamma+assembly.aoa
    assembly.yaw  = assembly.trajectory.chi - assembly.slip #christie: check sign of slip

    R_B_NED =   frames.R_B_NED(roll = assembly.roll, pitch = assembly.pitch, yaw = assembly.yaw) 
    R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)

    R_B_ECEF = (R_NED_ECEF*R_B_NED)

    assembly.quaternion = R_B_ECEF.as_quat()

    return


def compute_cartesian(assembly, options):
    '''
    Computation of the cartesian dynamics

    This function computes the cartesian position and velocity of the assembly

    Parameters
    ----------
    assembly: Assembly
        Object of class Assembly
    options: Options
        Object of class Options

    '''

    # The function assumes an ellipsoidal Earth
    #TODO 
    #This also needs to be considered in the gravity function

    [X,Y,Z] = pymap3d.geodetic2ecef(assembly.trajectory.latitude, assembly.trajectory.longitude, assembly.trajectory.altitude,
                        ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                        deg = False);

    uEast   = assembly.trajectory.velocity * np.cos(assembly.trajectory.gamma) * np.sin(assembly.trajectory.chi); #+ constants.omega_e * r * cos(latitude);
    vNorth  = assembly.trajectory.velocity * np.cos(assembly.trajectory.gamma) * np.cos(assembly.trajectory.chi);
    wUp     = assembly.trajectory.velocity * np.sin(assembly.trajectory.gamma);

    [U,V,W] = pymap3d.enu2uvw(uEast,vNorth,wUp,assembly.trajectory.latitude, assembly.trajectory.longitude,deg = False)

    assembly.position = np.array([X,Y,Z])
    assembly.velocity = np.array([U,V,W])

    #Account with the displacement of body COG and the origin of the body reference frame
    assembly.position += Rot.from_quat(assembly.quaternion).apply(assembly.COG)



def compute_cartesian_derivatives(assembly, options):
    """
    Computation of the cartesian derivatives

    This function computes the cartesian derivatives of the position and velocity
    It uses the gravity, aerodynamic, centrifugal and coriolis forces for the acceleration computation.

    Parameters
    ----------
    assembly: Assembly
        Object of class Assembly 
    options: Options
        Object of class Options

    """

    wE = options.planet.omega()
    
    r = np.linalg.norm(assembly.position)
    gr,gt = options.planet.gravitationalAcceleration(r, phi = np.pi/2 - assembly.trajectory.latitude)

    if options.freestream.method.upper() == "GRAM":
        data = gram.read_gram(assembly, options)
        gr = float(data['Gravity_ms2'])
        gt = 0

    #Delete
    #gr = -assembly.gravity

    [agrav_u,agrav_v,agrav_w] = pymap3d.enu2uvw(0,0, gr,assembly.trajectory.latitude, assembly.trajectory.longitude,deg = False)

    #R_W_NED = frames.R_W_NED(fpa = assembly.trajectory.gamma, ha = assembly.trajectory.chi)
    #R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
    #R_W_ECEF = R_W_NED* R_NED_ECEF

    q = assembly.quaternion

    #R_W_NED = frames.R_W_NED(fpa = assembly.trajectory.gamma, ha = assembly.trajectory.chi)
    R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
    #R_B_W_quat =  (R_NED_ECEF * R_W_NED).inv()*Rot.from_quat(q)
    R_B_ECEF = Rot.from_quat(q)

    Faero_I = R_B_ECEF.apply(np.array(assembly.body_force.force))

    Fgrav_I = np.array([agrav_u,agrav_v,agrav_w])
    # Fgrav_I = np.array([agrav_u, agrav_v, agrav_w])*assembly.mass
    Fcoreolis_I = -np.cross(np.array([0,0,wE]), np.cross(np.array([0,0,wE]), assembly.position))
    Fcentrif_I  = -2*np.cross(np.array([0,0,wE]), assembly.velocity)

    #For ECI, we need to work with the epochs to convert from ECEF to ECI -> To obtain Latitude, Longitude and Altitude
    #pymap3d has the functions we need

    dx = assembly.velocity
    # dv = (Faero_I + Fgrav_I + Fcoreolis_I + Fcentrif_I) / assembly.mass
    dv = ((Faero_I / assembly.mass) + Fgrav_I + Fcoreolis_I + Fcentrif_I)

    return DerivativesCartesian(dx = dx[0], dy = dx[1], dz = dx[2], du = dv[0], dv = dv[1], dw = dv[2])


def compute_angular_derivatives(assembly):
    """
    Computation of the angular derivatives in the Body frame

    This function computes the angular dericatives taking into consideration the euler and aerodynamic moments

    Parameters
    ----------
    assembly: Assembly
        Object of Assembly class

    """
    angle_vel = np.array([assembly.roll_vel, assembly.pitch_vel, assembly.yaw_vel])

    moment_euler = - np.cross(angle_vel, assembly.inertia@angle_vel)
    moment_body = assembly.body_force.moment

#christie: check d(euler) is correct from dM
    rotational_accel = np.linalg.solve(assembly.inertia, moment_body + moment_euler)

    droll  =  assembly.roll_vel
    dpitch =  assembly.pitch_vel
    dyaw   =  assembly.yaw_vel

    ddroll  = rotational_accel[0]
    ddpitch = rotational_accel[1]
    ddyaw   = rotational_accel[2]
    
    if (abs(ddroll) > 100) or (abs(ddpitch) > 100) or (abs(ddyaw) > 100):
        ddroll = np.sign(ddroll)*100
        ddpitch = np.sign(ddpitch)*100
        ddyaw = np.sign(ddyaw)*100
    
    return DerivativesAngle(droll = droll, dpitch = dpitch, dyaw = dyaw, ddroll = ddroll, ddpitch = ddpitch, ddyaw = ddyaw)

"""
def compute_point_mass_derivatives(assembly):

    #https://www.sciencedirect.com/science/article/pii/S1000936113002094
    #Using a spherical rotating Earth model

    rE = 6375253
    wE = 7.2921150e-5

    gamma = assembly.trajectory.gamma
    chi = assembly.trajectory.chi
    lat = assembly.trajectory.latitude
    lon = assembly.trajectory.longitude
    v = assembly.trajectory.velocity
    R = rE + assembly.trajectory.altitude
    mass = assembly.mass

    drag = assembly.wind_force.drag
    lift = assembly.wind_force.lift
    crosswind = assembly.wind_force.crosswind

    gr, gt = compute_gravity(assembly.trajectory.altitude)

    dh = v*np.sin(gamma)

    # Non-inertial forces // # Centrifugal force // # Coriolis contribution
    dv = -drag/mass - gr*np.sin(gamma)+ \
        gt*np.cos(gamma)*np.cos(chi) + \
        np.cos(lat) * (np.cos(lat)*np.sin(gamma) - np.cos(gamma)*np.cos(chi)*np.sin(lat)) * wE**2 * R;

    # Non-inertial forces // # Spherical geometry // # Centrifugal term // # Coriolis term
    dgamma = (lift/mass - gr*np.cos(gamma) - gt*np.sin(gamma)*np.cos(chi))/v + \
            (v/R)*np.cos(gamma) + \
            (np.cos(gamma)*np.cos(lat) + np.cos(chi)*np.sin(gamma)*np.sin(lat)) * (wE**2 * np.cos(gamma) * R) / v + \
            2 * np.cos(lat) * np.sin(chi) * wE; 

    # spherical geometry // # Non-inertial forces // # centripetal force from rotation of the Earth  // # coriolis force from rotation of the Earth
    dchi = (v / R) * np.cos(gamma) * np.sin(chi) * np.tan(lat) + \
           crosswind / (mass* v * np.cos(gamma)) + \
           wE**2 * R * (np.sin(chi) * np.sin(lat) * np.cos(lat)) / (v * np.cos(gamma)) + \
           2 * wE * ( np.sin(lat) - np.tan(gamma) * np.cos(chi) * np.cos(lat)) 
    
    dlat = v * np.cos(gamma) * np.cos(chi) / R
    dlon = v * np.cos(gamma) * np.sin(chi) / (R * np.cos(lat))

    return DerivativesPointMass(dh = dh, dv = dv, dchi = dchi, dgamma = dgamma, dlat = dlat, dlon = dlon)
"""
def integrate(titan, options):
    """
    Time integration

    This function calls a time integration scheme

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options

    """
    if options.wrap_propagator:
        from Uncertainty import UT
        UT.unscentedPropagation(titan,options)
    else: euler.compute_Euler(titan, options)
