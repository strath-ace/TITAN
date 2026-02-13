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

import pandas as pd
import os
import subprocess
import numpy as np

def generate_script(assembly, options):
    with open(options.output_folder + "/GRAM/gram_config_" + str(assembly.id), 'w') as f:

        f.write(" $INPUT \n")
        f.write("  SpicePath      = '" + str(options.gram.spicePath) + "'\n")

        if options.planet.name == "earth":
            f.write("  DataPath       = '" + str(options.gram.gramPath) + "/Earth/data'\n")

        elif options.planet.name == "mars":
            f.write("  DataPath       = '" + str(options.gram.gramPath) + "/Mars/data'\n")

        elif options.planet.name == "venus":
            f.write("  DataPath       = '" + str(options.gram.gramPath) + "/Venus/data'\n")

        elif options.planet.name == "titan":
            f.write("  DataPath       = '" + str(options.gram.gramPath) + "/Titan/data'\n")

        else:
            f.write("  DataPath       = '" + str(options.gram.gramPath) + "/Earth/data'\n")

        f.write("  ListFileName   = '" + str(options.output_folder) + "/GRAM/LIST'\n")
        f.write("  ColumnFileName = '" + str(options.output_folder) + "/GRAM/OUTPUT'\n")

        f.write("  Month = '" + str(options.gram.month) + "'\n")
        f.write("  Day = '" + str(options.gram.day) + "'\n")
        f.write("  Year = '" + str(options.gram.year) + "'\n")
        f.write("  Hour = '" + str(options.gram.hour) + "'\n")
        f.write("  Minute = '" + str(options.gram.minute) + "'\n")
        f.write("  Seconds = '" + str(options.gram.seconds) + "'\n")

        f.write("NumberOfPositions     = 1 \n")
        f.write("EastLongitudePositive = 1 \n")
        f.write("InitialHeight         = " + str(assembly.trajectory.altitude / 1000) + " \n")
        f.write("InitialLatitude       = " + str(assembly.trajectory.latitude * 180 / np.pi) + "\n")
        f.write("InitialLongitude      = " + str(assembly.trajectory.longitude * 180 / np.pi) + " \n")

        if options.planet.name != 'earth':
            f.write("MinMaxFactor = " + str(options.gram.MinMaxFactor) + " \n")
            f.write("ComputeMinMaxFactor = " + str(options.gram.ComputeMinMaxFactor) + " \n")

        f.write(" $END")

     
	
#  Month     = 3
#  Day       = 25
#  Year      = 2020
#  Hour      = 12
#  Minute    = 30
#  Seconds   = 0.0
#
#  InitialRandomSeed               = 1001    
#  RandomPerturbationScale         = 1.6   
#  HorizontalWindPerturbationScale = 1.75   
#  VerticalWindPerturbationScale   = 2.0   
#  NumberOfMonteCarloRuns          = 1
# 
#  AP         = 16.0
#  DailyF10   = 148.0
#  MeanF10    = 67.0
#  DailyS10   = 0.0
#  MeanS10    = 0.0
#  DailyXM10  = 0.0
#  MeanXM10   = 0.0
#  DailyY10   = 0.0
#  MeanY10    = 0.0
#  DSTTemperatureChange = 0.0
#  
#  ThermosphereModel = 1
#
#  NCEPYear = 9715
#  NCEPHour = 5
#
#  UseRRA  = 0
#  RRAYear = 2019
#  RRAOuterRadius = 2.0
#  RRAInnerRadius = 1.0
#  
#  Patchy = 0
#  SurfaceRoughness = -1
#
#  InitializePerturbations         = 0
#  InitialDensityPerturbation      = 0.0
#  InitialTemperaturePerturbation  = 0.0
#  InitialEWWindPerturbation       = 0.0
#  InitialNSWindPerturbation       = 0.0
#  InitialVerticalWindPerturbation = 0.0
#
#  UseTrajectoryFile     = 0
#  TrajectoryFileName    = 'null' 
		
#  DeltaHeight           = 40.0    
#  DeltaLatitude         = 0.3     
#  DeltaLongitude        = 0.5     
#  DeltaTime             = 500.0
#



#  MinMaxFactor           = Factor (-1. to +1. to vary between minimum and 
#                           maximuum allowed mean profiles
#  ComputeMinMaxFactor    = 0 to use Fminmax input value "as is"
#                           1 to automatically adjust input the factor for
#                             seasonal, latitude, and time-of-day effects
#  DinitrogenMoleFraction = N2 mole fraction (0.0 to 0.6)


#  UseAuxiliaryAtmosphere      = 0
#  AuxiliaryAtmosphereFileName = 'RRAanfAnn.txt'
#  OuterRadius = 0.0
#  InnerRadius = 0.0
#
#  FastModeOn        = 0
#  ExtraPrecision    = 0
#  UseLegacyOutputs  = 0 
#
		               

def read_gram_species(altitude, options):
    data = pd.read_csv(options.output_folder + "/GRAM/OUTPUT.csv")
    
    if options.planet.name == "earth":
        species_index = ["N2", "O2", "O", "He", "N", "H"]

    if options.planet.name == "neptune":
        species_index = ["H2", "He", "CH4"]

    if options.planet.name == "uranus":
        species_index = ["H2", "He", "CH4"]

    if options.planet.name == "mars":
        species_index = ["CO2", "N2", "Ar", "O2","CO"]


        composition = {
            "CO2": 0.961,
            "N2":  0.020,
            "Ar":  0.017,
            "O2":  0.001,
            "CO": 0.001
        }

        temperature = data["Temperature_K"].to_numpy()[0]
        density     = data["Density_kgm3"].to_numpy()[0]

        species_data = np.zeros(len(species_index) + 2)
        species_data[0] = altitude
        species_data[1] = temperature

        for i, sp in enumerate(species_index):
            species_data[i + 2] = density * composition[sp]

        return species_data, species_index
    
    if options.planet.name == "venus":
        species_index = ["CO2", "N2"]

        composition = {
            "CO2": 0.965,
            "N2":  0.035
        }

        temperature = data["Temperature_K"].to_numpy()[0]
        density     = data["Density_kgm3"].to_numpy()[0]

        species_data = np.zeros(len(species_index) + 2)
        species_data[0] = altitude
        species_data[1] = temperature

        for i, sp in enumerate(species_index):
            species_data[i + 2] = density * composition[sp]

        return species_data, species_index
    
    if options.planet.name == "titan":
        species_index = ["N2", "CH4"]

        composition = {
            "N2":  0.95,
            "CH4": 0.05
        }

        temperature = data["Temperature_K"].to_numpy()[0]
        density     = data["Density_kgm3"].to_numpy()[0]

        species_data = np.zeros(len(species_index) + 2)
        species_data[0] = altitude
        species_data[1] = temperature

        for i, sp in enumerate(species_index):
            species_data[i + 2] = density * composition[sp]

        return species_data, species_index



    temperature = data['Temperature_K'].to_numpy()[0]
    density = data['Density_kgm3'].to_numpy()[0]

    species_data = np.zeros(len(species_index) + 2)
    species_data[0] = altitude
    species_data[1] = temperature

    for index, specie in enumerate(species_index):
        species_data[index + 2] = data[specie + "_mass_pct"].to_numpy()[0] / 100

    species_data[2:] /= np.sum(species_data[2:])
    species_data[2:] *= density

    return species_data, species_index



def read_gram(assembly, options):
    data = pd.read_csv(options.output_folder + "/GRAM/OUTPUT.csv")
    return data


def run_single_gram(assembly, options):
    generate_script(assembly, options)
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #Run the GRAM model
    if options.planet.name == "earth":
        os.system("echo " + options.output_folder + "/GRAM/gram_config_" + str(assembly.id) +
                  " | " + path + "/Executables/EarthGRAM")

    if options.planet.name == "neptune":
        os.system("echo " + options.output_folder + "/GRAM/gram_config_" + str(assembly.id) +
                  " | " + path + "/Executables/NeptuneGRAM")

    if options.planet.name == "uranus":
        os.system("echo " + options.output_folder + "/GRAM/gram_config_" + str(assembly.id) +
                  " | " + path + "/Executables/UranusGRAM")

    if options.planet.name == "mars":
        os.system("echo " + options.output_folder + "/GRAM/gram_config_" + str(assembly.id) +
                  " | " + path + "/Executables/MarsGRAM")
        
    if options.planet.name == "venus":
        os.system("echo " + options.output_folder + "/GRAM/gram_config_" + str(assembly.id) +
                  " | " + path + "/Executables/VenusGRAM")

    if options.planet.name == "titan":
        os.system("echo " + options.output_folder + "/GRAM/gram_config_" + str(assembly.id) +
                  " | " + path + "/Executables/TitanGRAM") 




