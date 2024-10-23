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
import pickle
import pandas as pd
import os
import pathlib
import subprocess
import numpy as np
import datetime as dt
from scipy.interpolate import interp1d, PchipInterpolator

def generate_script(assembly, options):
	options.gram.isPerturbed=0

	if options.gram.Uncertain:
		from Uncertainty.atmosphere import setupGRAM
		setupGRAM(assembly,options)
	with open(options.output_folder+"/GRAM/gram_config_"+str(assembly.id), 'w') as f:

		f.write(" $INPUT \n")
		f.write("  SpicePath      = '"+options.gram.spicePath+"'\n")                                         
		f.write("  DataPath       = '"+options.gram.gramPath+"/Earth/data'\n")
		f.write("  ListFileName   = '"+options.output_folder+"/GRAM/LIST'\n")          
		f.write("  ColumnFileName = '"+options.output_folder+"/GRAM/OUTPUT'\n")       
		f.write("  EastLongitudePositive = 1 \n")

		if options.gram.isPerturbed:
			from Uncertainty.atmosphere import perturbGRAM
			f=perturbGRAM(f,options)
		else:
			f.write("  InitialHeight         = " + str(assembly.trajectory.altitude / 1000) + " \n")
			f.write("  InitialLatitude       = " + str(assembly.trajectory.latitude * 180 / np.pi) + "\n")
			f.write("  InitialLongitude      = " + str(assembly.trajectory.longitude * 180 / np.pi) + " \n")
			f.write("  NumberOfPositions     = 1 \n")
	
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
# f.write("NumberOfPositions     = 1 \n")
# f.write("EastLongitudePositive = 1 \n")
# f.write("InitialHeight         = "+str(assembly.trajectory.altitude/1000)+" \n")
# f.write("InitialLatitude       = "+str(assembly.trajectory.latitude*180/np.pi)+"\n")
# f.write("InitialLongitude      = "+str(assembly.trajectory.longitude*180/np.pi)+" \n")
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

	if options.planet.name == "earth":   species_index = ["N2", "O2", "O", "He", "N", "H"]
	if options.planet.name == "neptune": species_index = ["H2", "He", "CH4"]
	if options.planet.name == "uranus":  species_index = ["H2", "He", "CH4"]

	if not os.path.exists(options.output_folder+'/GRAM/gramSpecies.pkl') or not options.gram.isPerturbed:

		data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")


		temp_str = 'PerturbedTemperature_K' if options.gram.isPerturbed else 'Temperature_K'
		dens_str = 'PerturbedDensity_kgm3' if options.gram.isPerturbed else 'Density_kgm3'

		temperature = data[temp_str].to_numpy()
		density = data[dens_str].to_numpy()

		species_data = np.zeros((len(species_index)+2, len(temperature)))

		species_data[0, :] = data['Height_km'].to_numpy()*1000 if options.gram.isPerturbed else altitude
		species_data[1, :] = temperature

		for index, specie in enumerate(species_index):
			species_data[index+2, :] = data[specie+"mass_pct"].to_numpy()/100

		for i_alt in range(len(temperature)):
			species_data[2:, i_alt] /= np.sum(species_data[2:, i_alt])
			species_data[2:, i_alt] *= density[i_alt]

		if not options.gram.isPerturbed: return species_data[:, 0], species_index, 'NoInterpolator'

		_, unique_alts = np.unique(species_data[0, :], return_index=True)

		interp_data=species_data[:, unique_alts]

		with open(options.output_folder+'/GRAM/gramSpecies.pkl','wb') as file: pickle.dump(interp_data, file)
	else:
		with open(options.output_folder + '/GRAM/gramSpecies.pkl', 'rb') as file: interp_data = pickle.load(file)

	interp_data=np.transpose(interp_data)
	f = PchipInterpolator(interp_data[:, 0], interp_data, axis=0, extrapolate=False)
	species_data = f(altitude)
	if np.isnan(species_data).any(): species_data = interp_data[-1,:]
	# species_data = np.hstack((altitude, data))

	return species_data, species_index, f

def get_wind_vector(altitude, options):
	if not os.path.exists(options.output_folder+'/GRAM/gramWind.pkl'):

		if not os.path.exists(options.output_folder+"/GRAM/OUTPUT.csv"): return [0,0,0]
		data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")



		n_str = 'PerturbedNSWind_ms' if options.gram.isPerturbed else 'NSWind_ms'
		e_str = 'PerturbedEWWind_ms' if options.gram.isPerturbed else 'EWWind_ms'
		d_str = 'PerturbedVerticalWind_ms' if options.gram.isPerturbed else 'VerticalWind_ms'

		heights = data['Height_km'].to_numpy() * 1000 if options.gram.isPerturbed else altitude
		n_points = len(heights) if isinstance(heights,np.ndarray) else 1
		vectors = np.zeros((4, n_points))

		vectors[0, :] = heights
		vectors[1, :] = data[n_str].to_numpy()
		vectors[2, :] = data[e_str].to_numpy()
		vectors[3, :] = -1*data[d_str].to_numpy()

		if not options.gram.isPerturbed: return vectors[1:, 0]

		_, unique_alts = np.unique(vectors[0, :], return_index=True)

		interp_data=vectors[:, unique_alts]

		with open(options.output_folder+'/GRAM/gramWind.pkl','wb') as file: pickle.dump(interp_data, file)
	else:
		with open(options.output_folder + '/GRAM/gramWind.pkl', 'rb') as file: interp_data = pickle.load(file)

	interp_data=np.transpose(interp_data)
	f = PchipInterpolator(interp_data[:, 0], interp_data, axis=0, extrapolate=False)
	vector = f(altitude)
	if np.isnan(vector).any(): vector = interp_data[-1,:]
	return vector[1:]

def read_gram(assembly, options):
	data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")
	return data

def run_single_gram(assembly, options):
	generate_script(assembly, options)
	path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	#Run the GRAM model
	if options.planet.name == "earth":
		subprocess.run(
			args=[path + "/Executables/EarthGRAM", "-file", options.output_folder + "/GRAM/gram_config_" + str(assembly.id)], stdout=subprocess.DEVNULL
		)
	if options.planet.name == "neptune": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/NeptuneGRAM")
	if options.planet.name == "uranus": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/UranusGRAM")

