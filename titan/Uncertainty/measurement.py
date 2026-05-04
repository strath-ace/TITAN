#
# Copyright (c) 2026 TITAN Contributors (cf. AUTHORS.md).
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

import numpy as np
import pymap3d
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
from ..Dynamics import frames
from ..Dynamics.quaternion_operations import quaternion_normalize
from scipy.interpolate import PchipInterpolator

def measurement_func(measurements, options, augmented_state=False, x=None):
    if x==None: raise Exception('Must provide state')
    output_measures = []
    if not np.all(['pass' in measurements]): data = measurement_from_state(options, augmented_state, x)
    for measurement in measurements:
        # Define pass-through measurement as 'pass_i' to pass the ith state variable
        if 'pass' in measurement:
            output_measures.append(x[int(measurement.strip('pass_'))])
        elif measurement in data.keys(): output_measures.append(data[measurement])
        else: raise Exception('Measurement {} not available from state!'.format(measurement))
    return np.array(output_measures)

def measurement_from_state(options, augmented=False, state_vector=None):
    if state_vector  is None: raise Exception('Need to provide state vector')
    if augmented: 
        n_components = (len(state_vector) -13)/2
        masses = np.sum(state_vector[np.arange(14,12+2*n_components,2)])
        temperatures = state_vector[np.arange(12,11+2*n_components,2)] 
    quaternion = quaternion_normalize(state_vector[6:10])

    # Communicate state to other assembly attributes...
    [latitude, longitude, altitude] = pymap3d.ecef2geodetic(state_vector[0], state_vector[1], state_vector[2],ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),deg = False);


    R_NED_ECEF = frames.R_NED_ECEF(lat = latitude, lon =longitude)
    R_B_NED_quat = (R_NED_ECEF).inv()*Rot.from_quat(quaternion)
    [yaw,pitch,roll] = R_B_NED_quat.as_euler('ZYX')

    [vEast, vNorth, vUp] = pymap3d.uvw2enu(state_vector[3], state_vector[4], state_vector[5], latitude, longitude, deg=False)

    gamma = np.arcsin(np.dot(state_vector[:3], state_vector[3:6])/(np.linalg.norm(state_vector[:3])*np.linalg.norm(state_vector[3:6])))
    chi = np.arctan2(vEast,vNorth)

    #ECEF_2_B
    [Vx_B, Vy_B, Vz_B] =  Rot.from_quat(quaternion).inv().apply(state_vector[3:6])
    velocity = np.linalg.norm([Vx_B, Vy_B, Vz_B])

    magnitude_omega = np.linalg.norm(state_vector[10:13])

    aoa = np.arctan2(Vz_B,Vx_B)
    slip = np.arcsin(Vy_B/np.sqrt(Vx_B**2 + Vy_B**2 +  Vz_B**2))
    
    measurement_dict = {'altitude'  : altitude,
                        'latitude'  : latitude,
                        'longitude' : longitude,
                        'velocity'  : velocity,
                        'flight_path_angle' : gamma,
                        'heading_angle' : chi,
                        'angle_attack' : aoa,
                        'angle_sideslip' : slip,
                        'roll' : roll,
                        'pitch' : pitch,
                        'yaw' : yaw,
                        'magnitude_omega' : magnitude_omega}
    if augmented:
        measurement_dict['temperatures'] = temperatures
        measurement_dict['masses'] = masses
        measurement_dict['mass'] = np.sum(masses)
        measurement_dict['kinetic_energy'] = 0.5*np.sum(masses)*velocity**2
    return measurement_dict

class StateObservation():
    # This is a class that can provide a measurement from an available data source
    def __init__(self, kind='LUT', source=None, independent_variable='time', measurements = []):
        self.kind = kind
        self.source = None
        self.independent_variable = independent_variable
        if self.kind=='LUT': 
            if self.source is None: raise Exception('Must provide a data source')
            data = pd.read_csv(self.source)
            self.measurements = list(set(data.columns) - self.independent_variable)
            x = data[self.independent_variable].to_numpy()
            ys = data[self.measurements].to_numpy()
            self.interpolator = PchipInterpolator(x=x, y=ys)

    def observe(self, x):
        match self.kind:
            case 'LUT': return self.interpolator(x)
            case 'callable': return self.source(x)
