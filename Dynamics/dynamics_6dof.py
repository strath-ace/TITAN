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
import pymap3d
from scipy.spatial.transform import Rotation

def dynamics_6DOF_quaternions_cartesian_ECI(t, states, controls, phase, const):
    '''
    Python transcription of  MODHOC/Problems/OCD/Cartesian_6dof_OCD_eq2/Cartesian_6dof_OCD_state_equations_ph1.m file
    :param t:
    :param states:
    :param controls:
    :param phases:
    :param params:
    :param const:
    :return:
    states: states(1) = x       inertial position x
            states(2) = y       inertial position y
            states(3) = z       inertial position z
            states(4) = vx      inertial velocity x
            states(5) = vy      inertial velocity y
            states(6) = vz      inertial velocity z
            states(7) = q0      inertial attitude quaternion component 0
            states(8) = q1      inertial attitude quaternion component 1
            states(9) = q2      inertial attitude quaternion component 2
            states(10) = q3     inertial attitude quaternion component 3
            states(11) = wx     angular rate x in body frame
            states(12) = wy     angular rate y in body frame
            states(13) = wz     angular rate z in body frame
            states(14) = m      vehicle mass change rate
    controls:
    '''


    """

    altitude = []
    latitude = []
    longitude = []
    for i in range(np.shape(trajectory)[1]):
        altitude.append(np.linalg.norm(trajectory[0:3,i])-const['rE'])
        latitude.append(np.arcsin(trajectory[2,i] / (np.linalg.norm(trajectory[0:3,i]))))
        longitude.append(np.arctan2(trajectory[1,i], trajectory[0,i]))
    """

    r = states[0:3]   # position of fixed points in rigid part of rocket (point 0) [inertial frame]

    v = states[3:6]   # velocity vector of fixed point 0 [intertial frame]

    q = states[6:10]   # attitude quaternion, transform a vector from the body frame to the inertial frame

    omega = states[10:13]  # angular rate [body frame]

    #import ipdb; ipdb.set_trace()

    q = q / (np.linalg.norm(q))  # ensure quaternion has unit norm, can drift due to rounding

    if phase['mStateFlag'] is True:
        mTotal = states[13]  # total mass
    else:
        mTotal = phase['vehicle']['m0']

    q0 = q[0]    # individual components of attitude quaternion
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    ## angles and rotation matrices

    R_BI = Rotation.from_quat([q1, q2, q3, q0]).as_matrix()  #  rotation matrix FROM body frame TO inertial frame, scalar has to be at end in function

    rECEF = pymap3d.eci2ecef(states[0], states[1], states[2], phase['dateRef'], use_astropy=True)  # conversion of position from ECI to ECEF
    #rECEF = r #temporary fix
    rECEF = np.array(rECEF).squeeze()

    ###  atmospheric model
    atmoData = phase['atmo'](rECEF, phase, const)


    ### wind models
    windECEF = phase['wind'](rECEF, const, phase, atmoData)  # output wind components in ECEF

    windECI = pymap3d.ecef2eci(windECEF[0], windECEF[1], windECEF[2], phase['dateRef'], use_astropy=True)  # convert wind into ECI from ECEF
    #windECI = windECEF
    windECI = np.array(windECI).squeeze()

    windECI = windECI + np.cross(np.array(([0,
                                            0,
                                            const['wE']])), r)  # wind velocity + Earth rotation

    TAS_ECI = v - windECI  # true airspeed vector

    TAS_norm = np.linalg.norm(TAS_ECI)  # magnitude of TAS vector

    environment = {'atmoData': atmoData, 'TAS_norm': TAS_norm}

    #import ipdb; ipdb.set_trace()

    TAS_B = R_BI.transpose() @ TAS_ECI  # velocity wrt wind [body frame]

    ### evaluate aero angles

    aoa = np.arctan(TAS_B[2]/TAS_B[0])

    sideslip = np.arcsin(TAS_B[1]/TAS_norm)

    aero_angles = {'aoa': aoa, 'sideslip': sideslip}
    
    ## change with euler function
    R_WB = np.array(([[np.cos(aoa) * np.cos(sideslip), -np.cos(aoa) * np.sin(sideslip), -np.sin(aoa)],   # rotation matrix FROM wind frame TO body frame
                      [np.sin(sideslip),                np.cos(sideslip),                          0],
                      [np.sin(aoa) * np.cos(sideslip), -np.sin(aoa) * np.sin(sideslip), np.cos(aoa)]]))

    ### aerothermal model
    Faero_W, Maero_W, dmA, dxExtendA = phase['aero_thermal'](states, controls, const, phase, environment, aero_angles)  # input: airspeed magnitude and attitude in wind axes, aero angles;
                                                                                                                      # output: forces and moments in wind axis, variation due to thermal and additional states if necessary
    Faero_I = R_BI @ R_WB @ Faero_W  # aero forces in inertial frame (obtained rotating forces from wind
                                                 # frame to body, and then from body to inertial frame using the
                                                 # TRANSPOSE of R_IB)

    Maero_B = R_WB @ Maero_W  # aerodynamic moments in body axis


    ## Thrust and thrust related forces and torques

    Fthrust_B, Mthrust_B, dmp, dxExtendT = phase['prop'](t, states, controls, const, phase, environment)   #  outputs forces and moments in body axes


    Fthrust_I = R_BI @ Fthrust_B  # total thrust vector, inertial frame   # check matrix multiplications

    ## Evaluation of changes in inertia matrix
    inertia_vars = phase['inertia'](states, controls, const, phase)  # function used to update the inertia model due to changes in the mass

    J_B = inertia_vars['J_B']

    ## Gravity model
    Fgrav_I, Mgrav_B = phase['gravity'](t, states, controls, phase, const, inertia_vars)

    Meuler = -np.cross(omega, J_B @ omega)

    ## Equations of motion

    Forces_I = Fgrav_I + Faero_I + Fthrust_I    # total of forces acting on reference point [inertial frame]
    Moments_B = Maero_B + Mthrust_B + Mgrav_B + Meuler # total moments acting on reference point [inertial frame]


    omega_matr = np.array(([[0,        -omega[0], -omega[1], -omega[2]],       # angular velocity matrix for quaternions
                            [omega[0], 0,         omega[2],  -omega[1]],
                            [omega[1], -omega[2], 0,          omega[0]],
                            [omega[2],  omega[1],  -omega[0],        0]]))

    dq = 0.5 * omega_matr @ q

    domega = np.linalg.solve(J_B, Moments_B)

    dv = Forces_I / mTotal

    dm = dmA + dmp   ### TODO: to check if correct

    dx = np.concatenate(([v,
                          dv,
                          dq,
                          domega,
                          dm]))

    dx = np.hstack((dx, dxExtendA, dxExtendT))
    return dx

    