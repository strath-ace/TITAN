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
	with open(options.output_folder+"/GRAM/gram_config_"+str(assembly.id), 'w') as f:
		f.write(" $INPUT \n")
		f.write("  SpicePath      = '"+options.gram.spicePath+"'\n")                                         
		f.write("  DataPath       = '"+options.gram.gramPath+"/Earth/data'\n")
		f.write("  ListFileName   = '"+options.output_folder+"/GRAM/LIST'\n")          
		f.write("  ColumnFileName = '"+options.output_folder+"/GRAM/OUTPUT'\n")       
		f.write("  EastLongitudePositive = 1 \n")
		if options.gram.reference:
			from Uncertainty.atmosphere import construct_reference_traj
			f = construct_reference_traj(f,options)
		else:
			f.write("  InitialHeight         = " + str(assembly.trajectory.altitude / 1000) + " \n")
			f.write("  InitialLatitude       = " + str(assembly.trajectory.latitude * 180 / np.pi) + "\n")
			f.write("  InitialLongitude      = " + str(assembly.trajectory.longitude * 180 / np.pi) + " \n")
			f.write("  NumberOfPositions     = 1 \n")
		if options.gram.Uncertain:
			from Uncertainty.atmosphere import perturbGRAM
			f=perturbGRAM(f,options)

		# Removed additional GRAM options for readability, they can be found in the relevant GRAM users guide...
		#  e.g. https://ntrs.nasa.gov/api/citations/20210022157/downloads/Earth-GRAM%20User%20Guide_1.pdf

		if options.planet.name != 'earth':
			f.write("MinMaxFactor = " + options.gram.MinMaxFactor + " \n") 
			f.write("ComputeMinMaxFactor = " + options.gram.ComputeMinMaxFactor + " \n")

		f.write(" $END")                

def read_gram_species(altitude, options):

	if options.planet.name == "earth":   species_index = ["N2", "O2", "O", "He", "N", "H"]
	if options.planet.name == "neptune": species_index = ["H2", "He", "CH4"]
	if options.planet.name == "uranus":  species_index = ["H2", "He", "CH4"]

	if options.gram.reference:
		if not os.path.exists(options.output_folder + '/GRAM/gramSpecies.pkl'): 
			species_data=extract_species(options,species_index,altitude)
		with open(options.output_folder+'/GRAM/gramSpecies.pkl','rb') as file: interp_data=pickle.load(file)
		species_data = [np.interp(altitude,interp_data[0,:],interp_data[i_dataseries,:]) for i_dataseries in range(np.shape(interp_data)[0])]
		if np.isnan(species_data).any(): species_data = interp_data[-1,:]
	else:
		species_data=extract_species(options,species_index,altitude).flatten()
	return species_data, species_index

def extract_species(options,species_index,altitude):
	data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")

	temp_str = 'PerturbedTemperature_K' if options.gram.Uncertain else 'Temperature_K'
	dens_str = 'PerturbedDensity_kgm3' if options.gram.Uncertain else 'Density_kgm3'

	temperature = data[temp_str].to_numpy()
	density = data[dens_str].to_numpy()

	species_data = np.zeros((len(species_index)+2, len(temperature)))

	species_data[0, :] = data['Height_km'].to_numpy()*1000 if options.gram.Uncertain else altitude
	species_data[1, :] = temperature

	for index, specie in enumerate(species_index):
		species_data[index+2, :] = data[specie+"mass_pct"].to_numpy()/100

	for i_alt in range(len(temperature)):
		species_data[2:, i_alt] /= np.sum(species_data[2:, i_alt])
		species_data[2:, i_alt] *= density[i_alt]

	if options.gram.reference:
		_, unique_alts = np.unique(species_data[0, :], return_index=True)
		interp_data=species_data[:, unique_alts]
		interp_data = interp_data[:,interp_data[0,:].argsort()]
		with open(options.output_folder+'/GRAM/gramSpecies.pkl','wb') as file: pickle.dump(interp_data, file)

	return species_data

def read_gram(assembly, options):
	data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")
	return data

def run_single_gram(assembly, options):
	generate_script(assembly, options)
	path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	if (not os.path.exists(options.output_folder+'/GRAM/gramSpecies.pkl')) or (options.gram.reference==False):
	#Run the GRAM model
		if options.planet.name == "earth":
			subprocess.run(
				args=[path + "/Executables/EarthGRAM", "-file", options.output_folder + "/GRAM/gram_config_" + str(assembly.id)], stdout=subprocess.DEVNULL
			)
		if options.planet.name == "neptune": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/NeptuneGRAM")
		if options.planet.name == "uranus": os.system("echo "+options.output_folder+"/GRAM/gram_config_"+str(assembly.id)+" | "+path+"/Executables/UranusGRAM")

