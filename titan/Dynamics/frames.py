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
"""frames module."""
import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from numpy import cos, sin
import pymap3d

#REN to ECEF
def R_ECEF_from_NED(lat : float = 0., lon : float = 0.) -> Rotation:
    """Construct the rotation from local NED coordinates to ECEF.

    :param lat: Geodetic latitude in radians.
    :type lat: float
    :param lon: Longitude in radians.
    :type lon: float
    :return: Rotation mapping NED-frame vectors into ECEF.
    :rtype: scipy.spatial.transform.Rotation
"""
    latitude = lat   #Latitude
    longitude = lon  #Longitude

    R_ECEF_from_NED = Rotation.from_euler('ZY',
                                [longitude,-latitude-np.pi/2])  # converts from North East Down to ECEF  
                                                         

    return R_ECEF_from_NED

#WIND to REN 
#Wind frame should have the Z direction downwards to the body
def R_NED_from_W(fpa : float = 0., ha : float = 0.) -> Rotation:
    """Construct the rotation from the wind frame to local NED.

    Wind frame should have the Z direction downwards to the body

    :param fpa: Flight-path angle in radians.
    :type fpa: float
    :param ha: Heading angle in radians.
    :type ha: float
    :return: Rotation mapping wind-frame vectors into NED.
    :rtype: scipy.spatial.transform.Rotation
"""
    gamma = fpa #Flight Path Angle
    chi = ha    #Heading Angle
    
    R_NED_from_W = Rotation.from_euler('ZY', [chi, gamma])  # converts from wind frame to North East Down

    return R_NED_from_W

#WIND to BODY
def R_B_from_W(aoa : float = 0, slip : float = 0) -> Rotation:
    """Construct the rotation from wind axes to body axes.

    :param aoa: Angle of attack in radians.
    :type aoa: float
    :param slip: Sideslip angle in radians.
    :type slip: float
    :return: Rotation mapping wind-frame vectors into body axes.
    :rtype: scipy.spatial.transform.Rotation
"""
    a=aoa   #A = Angle of attack = Pitch
    b=slip  #B = Sideslip = Yaw

    R_B_from_W = Rotation.from_matrix(np.array([[cos(a)*cos(b), sin(b)*cos(a), -sin(a)],
                                    [-sin(b), cos(b), 0],
                                    [cos(b)*sin(a), sin(b)*sin(a), cos(a)]]))#* R_roll

    return R_B_from_W

#BODY to WIND
def R_W_from_B(aoa : float = 0., slip : float = 0.) -> Rotation:
    """Construct the inverse body-to-wind rotation.

    :param aoa: Angle of attack in radians.
    :type aoa: float
    :param slip: Sideslip angle in radians.
    :type slip: float
    :return: Rotation mapping body-frame vectors into wind axes.
    :rtype: scipy.spatial.transform.Rotation
"""

    R_W_from_B = R_B_from_W(aoa = aoa, slip = slip).inv().as_matrix()
    R_W_from_B[np.abs(R_W_from_B) < 1E-14] = 0
    R_W_from_B = Rotation.from_matrix(R_W_from_B)

    return R_W_from_B

#BODY to NED
def R_NED_from_B(roll : float = 0, pitch : float = 0, yaw : float = 0) -> Rotation:
    """Construct the rotation from body axes to local NED.

    :param roll: Roll angle in radians.
    :type roll: float
    :param pitch: Pitch angle in radians.
    :type pitch: float
    :param yaw: Yaw angle in radians.
    :type yaw: float
    :return: Rotation mapping body-frame vectors into NED.
    :rtype: scipy.spatial.transform.Rotation
"""

    R_NED_from_B = Rotation.from_euler('zyx', [yaw, pitch, roll])  # converts from Body frame to North East Down
    return(R_NED_from_B)

def R_ECEF_from_ECI(epoch) -> Rotation:
    """Construct the Earth-centred inertial to Earth-fixed rotation.

    :param epoch: Time at which the inertial-to-fixed transformation is evaluated.
    :type epoch: datetime.datetime
    :return: Rotation mapping ECI-frame vectors into ECEF.
    :rtype: scipy.spatial.transform.Rotation
    """
    rotMatrix = np.transpose([pymap3d.eci2ecef(1,0,0,epoch),
                              pymap3d.eci2ecef(0,1,0,epoch),
                              pymap3d.eci2ecef(0,0,1,epoch)])
    return Rotation.from_matrix(rotMatrix)


def Rig_iOTA_from_TITAN(assembly, epoch) -> RigidTransform:
    """Transform from a TITAN frame to an iOTA one, works dependent on settings in both codes

    :param assembly: Assembly of interest
    :type assembly: Assembly
    :param epoch: Time 
    :type epoch: datetime.datetime
    :return: Rigid transform from TITAN to iOTA
    :rtype: scipy.spatial.transform.RigidTransform
    """
    translate_titan_body_from_mesh = RigidTransform.from_translation(-assembly.COG)
    rotate_iota_body_from_titan_body = RigidTransform.from_rotation(
        Rotation.from_euler('x',0, degrees=True)
        )
    translate_ECEF_from_body = RigidTransform.from_translation(assembly.position)
    rotate_ECI_from_ECEF = RigidTransform.from_rotation(R_ECEF_from_ECI(epoch).inv())
    rotate_iota_ECI_from_titan_ECI = RigidTransform.from_rotation(
        Rotation.from_euler('y',0, degrees=True)
        )
    Rig_iOTA_from_TITAN =   rotate_iota_ECI_from_titan_ECI \
                          * rotate_ECI_from_ECEF \
                          * translate_ECEF_from_body \
                          * rotate_iota_body_from_titan_body \
                          * translate_ECEF_from_body \
                          * translate_titan_body_from_mesh
    
    return Rig_iOTA_from_TITAN

def R_ECEF_from_TVN(x : np.ndarray, v : np.ndarray) -> Rotation:
    """Transforms from Transverse-Velocity-Normal Frame (TVN/PTW/NTW) to ECEF

    The frame is defined as detailed in https://sanaregistry.org/r/orbit_relative_reference_frames/records/6

    :param x: Position vector
    :type x: np.ndarray
    :param v: Velocity vector
    :type v: np.ndarray
    :return: Rotation from TVN to ECEF
    :rtype: Rotation
    """

    r_hat = x / np.linalg.norm(x)
    v_hat = v / np.linalg.norm(v)
    
    # For TVN/NTW our T/N vector is not position aligned 
    x_track = np.cross(r_hat, v_hat)
    in_track = v_hat
    radial = np.cross(in_track, x_track) # i.e. this is not (necessarily) r_hat
    return Rotation.from_matrix(np.vstack([radial, in_track, x_track]))

def R_ECEF_from_RIC(x : np.ndarray, v : np.ndarray) -> Rotation:
    """Transforms from Radial, In-Track, Cross-Track (RIC/RTN) to RIC

    The frame is defined as detailed in https://sanaregistry.org/r/orbit_relative_reference_frames/records/9

    :param x: Position vector
    :type x: np.ndarray
    :param v: Velocity vector
    :type v: np.ndarray
    :return: Rotation from RIC to ECEF
    :rtype: Rotation
    """

    r_hat = x / np.linalg.norm(x)
    v_hat = v / np.linalg.norm(v)
    
    # For RTN/RIC our T/I vector is not velocity aligned 
    x_track = np.cross(r_hat, v_hat)
    in_track = np.cross(x_track, r_hat) # i.e. this is not (necessarily) v_hat
    radial = r_hat
    return Rotation.from_matrix(np.vstack([radial, in_track, x_track]))