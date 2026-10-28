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
"""atmosphere module."""
from scipy.interpolate import interp1d, RegularGridInterpolator
from functools import partial
from ..Freestream import gram
import numpy as  np
import pandas as pd
import os
import pymsis
from datetime import timedelta


## Chemical parameters
Avo = 6.022169e23       #Avogrados number 

mN2 = 28.01340/1E3;               #molar mass of nitrogen molecule, kg/mole
mO2 = 31.99880/1E3;               #molar mass of oxigen molecule,   kg/mole
mO = mO2/2.0;                     #molar mass of oxigen atom,       kg/mole
mN = mN2/2.0;                     #molar mass of Nitrogen atom,     kg/mole
mAr = 39.9480/1E3;                #molar mass of Argon molecule,    kg/mole
mHe = 4.0026020/1E3;              #molar mass of helium molecule,   kg/mole
mH= 1.007940/1E3;                 #molar mass of Hydrogen molecule, kg/mole

def convert_numberDensity_to_density(atm,species):
    """Convert species number densities in an atmosphere table to mass densities.

:param atm: Atmospheric data frame modified in place.
:type atm: Any
:param species: Species names whose number densities are converted.
:type species: list[str]
"""

    for specie in species:
        if specie == "N2":    atm[specie] *= mN2/Avo
        if specie == "O2":    atm[specie] *= mO2/Avo
        if specie == "O" :    atm[specie] *= mO/Avo
        if specie == "N" :    atm[specie] *= mN/Avo
        if specie == "Ar":    atm[specie] *= mAr/Avo
        if specie == "He":    atm[specie] *= mHe/Avo
        if specie == "H" :    atm[specie] *= mH/Avo


def load_atmosphere(name, options, epoch=None):
    """Load or select the configured atmospheric model.

:param name: Atmospheric model name (for example, ``'PYMSIS'`` or ``'GRAM'``).
:type name: str
:param options: Simulation options used to cache model interpolation data.
:type options: Options
:param epoch: Optional epoch for time-dependent atmospheric models.
:type epoch: datetime.datetime or None
:return: Atmospheric data provider and its species ordering.
:rtype: tuple[callable, list[str]]
"""

    dirname = os.path.dirname(os.path.abspath(__file__))

    if name.upper() == "NRLMSISE00" or name.upper() == "PYMSIS":
         species_index = ["N2","O2","O","He","Ar","N","H"]
#        if options.planet.name != "earth": raise Exception("The model NRLMSISE00 contains Earth atmopshere. Please choose the GRAM model")
         if not hasattr(options,'atmosphere_interp'):
            dirname = os.path.dirname(os.path.abspath(__file__))
            atm = pd.read_csv(dirname+'/Models/NRLMSISE00.csv')

            

            #Convert from 1/cm^3 to 1/m^3      
            atm[species_index] *= 1E6

            #Convert from number density (1/m^3) to (kg/(m^3))
            convert_numberDensity_to_density(atm,species_index)

            options.atmosphere_interp = interp1d(atm.iloc[:,0], atm, axis = 0, kind = 'cubic')
    elif name.upper() == 'PYMSIS_ONLINE':
        species_index = ["N2","O2","O","He","Ar","N","H"]
        return pymsis_wrapper, species_index
    return options.atmosphere_interp, species_index


def retrieve_atmosphere_data(name, altitude, assembly, options, epoch = None):
    """Retrieve atmospheric properties at one assembly altitude.

:param name: Atmospheric model name.
:type name: str
:param altitude: Geodetic altitude in metres.
:type altitude: Any
:param assembly: Assembly object to process.
:type assembly: object
:param options: Options or configuration object.
:type options: Options
:param epoch: Optional epoch for time-dependent model evaluation.
:type epoch: datetime.datetime or None
:return: Atmospheric state and species information for the requested altitude.
:rtype: Any
"""

    #This function only returns the data for a single altitude

    if name.upper() == "NRLMSISE00" or name.upper() == "PYMSIS":
        if options.planet.name != "earth": raise Exception("The model NRLMSISE00 contains Earth atmopshere. Please choose the GRAM model")
        species_index = ["N2","O2","O","He","Ar","N","H"]
        if not hasattr(options,'atmosphere_interp'):
            dirname = os.path.expandvars('$TITAN_PATH/titan/Freestream')
            atm = pd.read_csv(dirname+'/Models/NRLMSISE00.csv')

            

            #Convert from 1/cm^3 to 1/m^3      
            atm[species_index] *= 1E6

            #Convert from number density (1/m^3) to (kg/(m^3))
            convert_numberDensity_to_density(atm,species_index)

            options.atmosphere_interp = interp1d(atm.iloc[:,0], atm, axis = 0, kind = 'cubic')

        if altitude<=0: altitude = 0

        if isinstance(options.atmosphere_interp, interp1d):
            data = options.atmosphere_interp(altitude)
        elif isinstance(options.atmosphere_interp, RegularGridInterpolator):
            data = options.atmosphere_interp([assembly.trajectory.latitude*180/np.pi, assembly.trajectory.longitude*180/np.pi, altitude/1000]).flatten()
            data = np.hstack([altitude, data])
        

    elif name.upper() == "GRAM":
        gram.run_single_gram(assembly, options)
        data, species_index = gram.read_gram_species(altitude, options)
    elif name.upper() == 'PYMSIS_ONLINE':
        species_index = ["N2","O2","O","He","Ar","N","H"]
        atmos = pymsis.calculate(epoch,assembly.trajectory.latitude*180/np.pi,assembly.trajectory.longitude*180/np.pi,altitude/1000)
        data = np.hstack([altitude, process_pymsis_result(atmos).flatten()])
    data[2:] *= options.freestream.density_mult
    
    return data, species_index

def create_msis_atmosphere(titan, options, altitudes = None, latitudes = None, longitudes = None):
    """Builds msis grid at the epoch of the simulation with specified resolution.
    Creates an interpolant to be called 

    :param titan: _description_
    :type titan: _type_
    :param options: _description_
    :type options: _type_
    """
    if altitudes is None: altitudes  = np.arange(0,999,2.0)
    if latitudes is None: latitudes  = np.arange(-90,90,10.0)
    if longitudes is None: longitudes = np.arange(-180,180,10.)
    print('Building MSIS atmosphere grid, this may take some time...')
    msis_atmosphere = pymsis.calculate(titan.epoch+timedelta(seconds=titan.time),latitudes,longitudes,altitudes)

    msis_atmosphere = process_pymsis_result(msis_atmosphere)
    #print(atmosphere)

    options.atmosphere_interp = RegularGridInterpolator((latitudes,longitudes, altitudes),msis_atmosphere, method='linear', bounds_error=False, fill_value=None)
    print('Atmosphere grid completed!')

def process_pymsis_result(atmos : np.ndarray) -> np.ndarray:
    if atmos.ndim >2:
        atmos = atmos[0][:,:,:,[10,1,2,3,7,6,4,5]]

        atmos[:,:,:,1] *=mN2/Avo
        atmos[:,:,:,2] *=mO2/Avo
        atmos[:,:,:,3] *=mO/Avo
        atmos[:,:,:,4] *=mAr/Avo
        atmos[:,:,:,5] *=mHe/Avo
        atmos[:,:,:,6] *=mN/Avo
        atmos[:,:,:,7] *=mH/Avo    
    
    elif atmos.ndim == 2: 
        atmos = atmos[:,[10,1,2,3,7,6,4,5]]
        atmos[:,1] *=mN2/Avo
        atmos[:,2] *=mO2/Avo
        atmos[:,3] *=mO/Avo
        atmos[:,4] *=mAr/Avo
        atmos[:,5] *=mHe/Avo
        atmos[:,6] *=mN/Avo
        atmos[:,7] *=mH/Avo

    else:
        atmos = atmos[[10,1,2,3,7,6,4,5]]
        atmos[1] *=mN2/Avo
        atmos[2] *=mO2/Avo
        atmos[3] *=mO/Avo
        atmos[4] *=mAr/Avo
        atmos[5] *=mHe/Avo
        atmos[6] *=mN/Avo
        atmos[7] *=mH/Avo    

    atmos[np.isnan(atmos)]=0
    


    return atmos

def pymsis_wrapper(epoch, latitude, longitude, altitude):
    atmos = pymsis.calculate(epoch, latitude, longitude, altitude)
    return process_pymsis_result(atmos)