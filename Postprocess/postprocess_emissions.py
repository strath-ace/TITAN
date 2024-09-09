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

from Thermal import thermal
from Output import output
import numpy as np
import pandas as pd
import pickle
import os
import glob

def postprocess_emissions(options):

	path = options.output_folder+'/Data/*'
	search_string = 'thermal'
	
	# Get a list of all files in the folder
	files = glob.glob(path)
	
	# Iterate and delete each file that contains the search_string
	for file in files:
		print(os.path.basename(file))
		if os.path.isfile(file) and search_string in os.path.basename(file):
			os.remove(file)

	data = pd.read_csv(options.output_folder+'/Data/data.csv', index_col = False)

	iter_interval = np.unique(data['Iter'].to_numpy())
	
	for iter_value in range(0, max(iter_interval)+1, options.save_freq):
		iter_value = int(iter_value)
		print('iter:', iter_value)	
		titan = read_state(options, iter_value)
		#emissions(titan, options, iter_value, data)
		output.generate_surface_solution(titan = titan, options = options, folder = 'Postprocess_emissions')

def read_state(options, i = 0):
    """
    Load state of the TITAN object for the given iteration

    Returns
    -------
    titan: Assembly_list
        Object of class Assembly_list
    """

    infile = open(options.output_folder + '/Restart/'+ 'Assembly_State_'+str(i)+'_.p','rb')
    titan = pickle.load(infile)
    infile.close()

    return titan

def emissions(titan, options, iter_value, data):

	index = data['Iter'] == iter_value

	#Quaternions Body to ECEF
	qw = data[index]['Quat_prev_w'].to_numpy()
	qx = data[index]['Quat_prev_x'].to_numpy()
	qy = data[index]['Quat_prev_y'].to_numpy()
	qz = data[index]['Quat_prev_z'].to_numpy()

	q = np.array([qx,qy,qz,qw]).transpose()

	print('iter:', iter_value)	
	print('options.radiation.spectral:', options.radiation.spectral)
	print('options.radiation.spectral_freq:', options.radiation.spectral_freq)
	print('/:', iter_value%options.radiation.spectral_freq)

	if options.radiation.black_body_emissions and (iter_value%options.radiation.black_body_emissions_freq == 0):
		thermal.compute_black_body_emissions(titan, options, q)
	if options.radiation.spectral and (iter_value%options.radiation.spectral_freq == 0):
		print('0')
		thermal.compute_black_body_spectral_emissions(titan, options, q)
	if options.thermal.ablation and options.radiation.particle_emissions and options.thermal.pato and (iter_value%options.radiation.black_body_emissions_freq == 0):
		thermal.compute_particle_emissions(titan, options, q)