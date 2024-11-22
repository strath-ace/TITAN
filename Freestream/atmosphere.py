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
from scipy.interpolate import interp1d, PchipInterpolator
from Freestream import gram
import numpy as  np
import pandas as pd
import os

def convert_numberDensity_to_density(atm,species):
    Avo = 6.022169e23       #Avogrados number 

    mN2 = 28.01340/1E3;               #molar mass of nitrogen molecule, kg/mole
    mO2 = 31.99880/1E3;               #molar mass of oxigen molecule,   kg/mole
    mO = mO2/2.0;                   #molar mass of oxigen atom,       kg/mole
    mN = mN2/2.0;                   #molar mass of Nitrogen atom,     kg/mole
    mAr = 39.9480/1E3;                #molar mass of Argon molecule,    kg/mole
    mHe = 4.0026020/1E3;              #molar mass of helium molecule,   kg/mole
    mH= 1.007940/1E3;                 #molar mass of Hydrogen molecule, kg/mole

    for specie in species:
        if specie == "N2":    atm[specie] *= mN2/Avo
        if specie == "O2":    atm[specie] *= mO2/Avo
        if specie == "O" :    atm[specie] *= mO/Avo
        if specie == "N" :    atm[specie] *= mN/Avo
        if specie == "Ar":    atm[specie] *= mAr/Avo
        if specie == "He":    atm[specie] *= mHe/Avo
        if specie == "H" :    atm[specie] *= mH/Avo


def load_atmosphere(name):
    """
    This function loads the atmosphere model with respect to the user specification

    Parameters
    ----------
    name: str
        Name of the atmospheric model

    Returns
    -------
    f: scipy.interpolate.interp1d
        Function interpolation of the atmopshere atributes with respect to altitude
    spacies_index: array
        Array with the species used in the model
    """



    species_index = ["N2", "O2", "O", "He", "Ar", "N", "H"]

    if name.upper() == "NRLMSISE00":
#        if options.planet.name != "earth": raise Exception("The model NRLMSISE00 contains Earth atmopshere. Please choose the GRAM model")

        dirname = os.path.dirname(os.path.abspath(__file__))
        atm = pd.read_csv(dirname+'/Models/NRLMSISE00.csv')

        #Convert from 1/cm^3 to 1/m^3      
        atm[species_index] *= 1E6

        #Convert from number density (1/m^3) to (kg/(m^3))
        convert_numberDensity_to_density(atm,species_index)
        f = PchipInterpolator(atm.iloc[:,0], atm, axis = 0)
    
    return f, species_index


def retrieve_atmosphere_data(name, altitude, assembly, options):

    #This function only returns the data for a single altitude

    if name.upper() == "NRLMSISE00":
        if options.planet.name != "earth": raise Exception("The model NRLMSISE00 contains Earth atmopshere. Please choose the GRAM model")
        
        dirname = os.path.dirname(os.path.abspath(__file__))
        atm = pd.read_csv(dirname+'/Models/NRLMSISE00.csv')

        species_index = ["N2","O2","O","He","Ar","N","H"]

        #Convert from 1/cm^3 to 1/m^3      
        atm[species_index] *= 1E6

        #Convert from number density (1/m^3) to (kg/(m^3))
        convert_numberDensity_to_density(atm,species_index)

        f = PchipInterpolator(atm.iloc[:,0], atm, axis = 0)
        data = f(altitude)

    elif name.upper() == "GRAM":
        gram.run_single_gram(assembly, options)
        data, species_index = gram.read_gram_species(altitude, options)


    return data, species_index