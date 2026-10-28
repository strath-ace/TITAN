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
def R_ECEF_from_NED(lat = 0, lon = 0):
    """Construct the rotation from local NED coordinates to ECEF.

    :param lat: Geodetic latitude in radians.
    :type lat: Any
    :param lon: Longitude in radians.
    :type lon: Any
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
def R_NED_from_W(fpa = 0, ha = 0):
    """Construct the rotation from the wind frame to local NED.

    :param fpa: Flight-path angle in radians.
    :type fpa: Any
    :param ha: Heading angle in radians.
    :type ha: Any
    :return: Rotation mapping wind-frame vectors into NED.
    :rtype: scipy.spatial.transform.Rotation
"""
    gamma = fpa #Flight Path Angle
    chi = ha    #Heading Angle
    
    R_NED_from_W = Rotation.from_euler('ZY', [chi, gamma])  # converts from wind frame to North East Down

    return R_NED_from_W

#WIND to BODY
def R_B_from_W(aoa = 0, slip = 0):
    """Construct the rotation from wind axes to body axes.

    :param aoa: Angle of attack in radians.
    :type aoa: Any
    :param slip: Sideslip angle in radians.
    :type slip: Any
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
def R_W_from_B(aoa = 0, slip = 0):
    """Construct the inverse body-to-wind rotation.

    :param aoa: Angle of attack in radians.
    :type aoa: Any
    :param slip: Sideslip angle in radians.
    :type slip: Any
    :return: Rotation mapping body-frame vectors into wind axes.
    :rtype: scipy.spatial.transform.Rotation
"""

    R_W_from_B = R_B_from_W(aoa = aoa, slip = slip).inv().as_matrix()
    R_W_from_B[np.abs(R_W_from_B) < 1E-14] = 0
    R_W_from_B = Rotation.from_matrix(R_W_from_B)

    return R_W_from_B

#BODY to NED
def R_NED_from_B(roll = 0, pitch = 0, yaw = 0):
    """Construct the rotation from body axes to local NED.

    :param roll: Roll angle in radians.
    :type roll: Any
    :param pitch: Pitch angle in radians.
    :type pitch: Any
    :param yaw: Yaw angle in radians.
    :type yaw: Any
    :return: Rotation mapping body-frame vectors into NED.
    :rtype: scipy.spatial.transform.Rotation
"""

    R_NED_from_B = Rotation.from_euler('zyx', [yaw, pitch, roll])  # converts from Body frame to North East Down
    return(R_NED_from_B)

def R_ECEF_from_ECI(epoch):
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


def Rig_iOTA_from_TITAN(assembly, epoch):
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