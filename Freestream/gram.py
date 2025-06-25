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
	with open(options.output_folder+"/GRAM/gram_config_"+str(assembly.id), 'w') as f:

		f.write(" $INPUT \n")
		f.write("  SpicePath      = '"+options.gram.spicePath+"'\n")                                         
		f.write("  DataPath       = '"+options.gram.gramPath+"/Earth/data'\n")
		f.write("  ListFileName   = '"+options.output_folder+"/GRAM/LIST'\n")          
		f.write("  ColumnFileName = '"+options.output_folder+"/GRAM/OUTPUT'\n")       
		
		f.write("  Month = '"+options.gram.month+"\n")       
		f.write("  Day = '"+options.gram.day+"\n")       
		f.write("  Year = '"+options.gram.year+"\n")       
		f.write("  Hour = '"+options.gram.hour+"\n")       
		f.write("  Minute = '"+options.gram.minute+"\n")       
		f.write("  Seconds = '"+options.gram.seconds+"\n")       
	
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
		f.write("NumberOfPositions     = 1 \n")   
		f.write("EastLongitudePositive = 1 \n")  
		f.write("InitialHeight         = "+str(assembly.trajectory.altitude/1000)+" \n")    
		f.write("InitialLatitude       = "+str(assembly.trajectory.latitude*180/np.pi)+"\n")
		f.write("InitialLongitude      = "+str(assembly.trajectory.longitude*180/np.pi)+" \n") 
#  DeltaHeight           = 40.0    
#  DeltaLatitude         = 0.3     
#  DeltaLongitude        = 0.5     
#  DeltaTime             = 500.0
#

		if options.planet.name != 'earth':
			f.write("MinMaxFactor = " + options.gram.MinMaxFactor + " \n") 
			f.write("ComputeMinMaxFactor = " + options.gram.ComputeMinMaxFactor + " \n")


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
		f.write(" $END")                

def read_gram_species(altitude, options):
	data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")

	if options.planet.name == "earth":   species_index = ["N2","O2","O","He","N","H"]
	if options.planet.name == "neptune": species_index = ["H2","He","CH4"]
	if options.planet.name == "uranus":  species_index = ["H2","He","CH4"]

	temperature = data['Temperature_K'].to_numpy()[0]
	density = data['Density_kgm3'].to_numpy()[0]

	species_data = np.zeros(len(species_index)+2)

	species_data[0] = altitude
	species_data[1] = temperature

	for index, specie in enumerate(species_index):
		species_data[index+2] = data[specie+"mass_pct"].to_numpy()[0]/100

	species_data[2:] /= np.sum(species_data[2:])
	species_data[2:] *= density

	return species_data, species_index

def read_gram(assembly, options):
	data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")
	return data

def run_single_gram(assembly, options):
	generate_script(assembly, options)
	path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	#Run the GRAM model
	if options.planet.name == "earth": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/EarthGRAM")
	if options.planet.name == "neptune": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/NeptuneGRAM")
	if options.planet.name == "uranus": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/UranusGRAM")