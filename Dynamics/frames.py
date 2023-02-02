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
from scipy.spatial.transform import Rotation as Rot
from numpy import cos, sin

#REN to ECEF
def R_NED_ECEF(lat = 0, lon = 0):
    latitude = lat   #Latitude
    longitude = lon  #Longitude

    R_NED_ECEF = Rot.from_euler('ZY',
                                [longitude,-latitude-np.pi/2])  # converts from North East Down to ECEF  
                                                         

    return R_NED_ECEF

#WIND to REN 
#Wind frame should have the Z direction downwards to the body
def R_W_NED(fpa = 0, ha = 0):
    gamma = fpa #Flight Path Angle
    chi = ha    #Heading Angle
    
    R_W_NED = Rot.from_euler('ZY', [chi, gamma])  # converts from wind frame to North East Down

    return R_W_NED

#WIND to BODY
def R_W_B(aoa = 0, slip = 0):
    a=aoa   #A = Angle of attack = Pitch
    b=slip  #B = Sideslip = Yaw

    R_W_B = Rot.from_matrix(np.array([[cos(a)*cos(b), sin(b)*cos(a), -sin(a)],
                                    [-sin(b), cos(b), 0],
                                    [cos(b)*sin(a), sin(b)*sin(a), cos(a)]]))#* R_roll

    return R_W_B

#BODY to WIND
def R_B_W(aoa = 0, slip = 0):

    R_B_W = R_W_B(aoa = aoa, slip = slip).inv().as_matrix()
    R_B_W[np.abs(R_B_W) < 1E-14] = 0
    R_B_W = Rot.from_matrix(R_B_W)

    return R_B_W

#BODY to NED
def R_B_NED(roll = 0, pitch = 0, yaw = 0):

    R_B_NED = Rot.from_euler('ZYX', [yaw, pitch, roll])  # converts from Body frame to North East Down
    return(R_B_NED)