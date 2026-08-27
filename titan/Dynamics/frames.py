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
from scipy.spatial.transform import Rotation as Rot
from numpy import cos, sin

#REN to ECEF
def R_ECEF_from_NED(lat = 0, lon = 0):
    """Documentation for the function.
:param lat: Value for lat.
:type lat: Any
:param lon: Value for lon.
:type lon: Any
:return: Return value.
:rtype: Any"""
    latitude = lat   #Latitude
    longitude = lon  #Longitude

    R_ECEF_from_NED = Rot.from_euler('ZY',
                                [longitude,-latitude-np.pi/2])  # converts from North East Down to ECEF  
                                                         

    return R_ECEF_from_NED

#WIND to REN 
#Wind frame should have the Z direction downwards to the body
def R_NED_from_W(fpa = 0, ha = 0):
    """Documentation for the function.
:param fpa: Value for fpa.
:type fpa: Any
:param ha: Value for ha.
:type ha: Any
:return: Return value.
:rtype: Any"""
    gamma = fpa #Flight Path Angle
    chi = ha    #Heading Angle
    
    R_NED_from_W = Rot.from_euler('ZY', [chi, gamma])  # converts from wind frame to North East Down

    return R_NED_from_W

#WIND to BODY
def R_B_from_W(aoa = 0, slip = 0):
    """Documentation for the function.
:param aoa: Value for aoa.
:type aoa: Any
:param slip: Value for slip.
:type slip: Any
:return: Return value.
:rtype: Any"""
    a=aoa   #A = Angle of attack = Pitch
    b=slip  #B = Sideslip = Yaw

    R_B_from_W = Rot.from_matrix(np.array([[cos(a)*cos(b), sin(b)*cos(a), -sin(a)],
                                    [-sin(b), cos(b), 0],
                                    [cos(b)*sin(a), sin(b)*sin(a), cos(a)]]))#* R_roll

    return R_B_from_W

#BODY to WIND
def R_W_from_B(aoa = 0, slip = 0):
    """Documentation for the function.
:param aoa: Value for aoa.
:type aoa: Any
:param slip: Value for slip.
:type slip: Any
:return: Return value.
:rtype: Any"""

    R_W_from_B = R_B_from_W(aoa = aoa, slip = slip).inv().as_matrix()
    R_W_from_B[np.abs(R_W_from_B) < 1E-14] = 0
    R_W_from_B = Rot.from_matrix(R_W_from_B)

    return R_W_from_B

#BODY to NED
def R_NED_from_B(roll = 0, pitch = 0, yaw = 0):
    """Documentation for the function.
:param roll: Value for roll.
:type roll: Any
:param pitch: Value for pitch.
:type pitch: Any
:param yaw: Value for yaw.
:type yaw: Any
:return: Return value.
:rtype: Any"""

    R_NED_from_B = Rot.from_euler('ZYX', [yaw, pitch, roll])  # converts from Body frame to North East Down
    return(R_NED_from_B)
