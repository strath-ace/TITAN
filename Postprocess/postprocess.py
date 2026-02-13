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
import os
import pandas as pd
import numpy as np
import meshio
from Dynamics import frames
from scipy.spatial.transform import Rotation as Rot

def postprocess(options, postprocess = "wind", filter_name = None):
	if not os.path.exists(options.output_folder+'/Dense_surface_solution'):
		data = pd.read_csv(options.output_folder+'/Data/data.csv', index_col = False)
		data_obj = pd.read_csv(options.output_folder+'/Data/data_assembly.csv', index_col = False)

		iter_interval = np.unique(data['Iter'].to_numpy())
		
		for iter_value in range(0, max(iter_interval)+1, options.output_freq):
			generate_visualization(options, data, iter_value, postprocess, filter_name, data_obj)
	else:
		data_smooth = pd.read_csv(options.output_folder+'/Data/data_smooth.csv')

		times = np.unique(data_smooth['Time'].to_numpy())

		for i_time, time in enumerate(times):
			generate_visualization(options, data_smooth, np.round(time,6), postprocess,is_dense=True, iter_override=i_time)

def generate_visualization(options, data, iter_value, postprocess = "wind", filter_name = None, data_obj = None, is_dense=False, iter_override=None):

	if not is_dense:
		index = data['Iter']==iter_value
	else: index = data['Time']==iter_value

	assembly_ID = data[index]['Assembly_ID'].to_numpy()

	if filter_name:
		if is_dense: raise NotImplementedError
		assembly_obj = data_obj[(data_obj['Iter'] == iter_value)*(data_obj['Parent_part'].str.contains(filter_name))]['Assembly_ID'].to_numpy()
		index = (data['Iter']==iter_value)*(data['Assembly_ID']==assembly_obj[0])
		assembly_ID = data[index]['Assembly_ID'].to_numpy()
	print(data[index])
	latitude    = data[index]['Latitude'].to_numpy()/180*np.pi
	altitude    = data[index]['Altitude'].to_numpy()
	longitude   = data[index]['Longitude'].to_numpy()/180*np.pi
	chi         = data[index]['HeadingAngle'].to_numpy()/180*np.pi
	gamma       = data[index]['FlightPathAngle'].to_numpy()/180*np.pi
	aoa         = data[index]['AngleAttack'].to_numpy()/180*np.pi
	slip        = data[index]['AngleSideslip'].to_numpy()/180*np.pi
	mass        = data[index]['Mass'].to_numpy()

	roll         = data[index]['Roll'].to_numpy()/180*np.pi
	pitch        = data[index]['Pitch'].to_numpy()/180*np.pi
	yaw        = data[index]['Yaw'].to_numpy()/180*np.pi

	#Quaternions Body to ECEF
	qw = data[index]['Quat_w'].to_numpy()
	qx = data[index]['Quat_x'].to_numpy()
	qy = data[index]['Quat_y'].to_numpy()
	qz = data[index]['Quat_z'].to_numpy()

	#Position ECEF
	X = data[index]['ECEF_X'].to_numpy()
	Y = data[index]['ECEF_Y'].to_numpy()
	Z = data[index]['ECEF_Z'].to_numpy()

	body_X = data[index]['BODY_COM_X'].to_numpy()
	body_Y = data[index]['BODY_COM_Y'].to_numpy()
	body_Z = data[index]['BODY_COM_Z'].to_numpy()

	q = np.array([qx,qy,qz,qw]).transpose()

	#Retrieve index of maximum mass to lock the assembly on point (0,0,0)
	index_mass = np.argmax(mass)

	#CHANGE LATER
	#index_mass = np.argmin(altitude)

	#R_NED_W = frames.R_W_NED(fpa = gamma[index_mass], ha = chi[index_mass]).inv()
	#R_ECEF_NED = frames.R_NED_ECEF(lat = latitude[index_mass], lon = longitude[index_mass]).inv()
	#R_ECEF_W_0 = R_NED_W*R_ECEF_NED

	#Read mesh information and surface quantities, place them on the ECEF
	mesh = []
	for i, _id in enumerate(assembly_ID):
		if not is_dense:
			mesh.append(meshio.read(options.output_folder+'/Surface_solution/ID_'+str(int(_id))+'/solution_iter_'+str(iter_value).zfill(3)+'.xdmf'))
		else:
			mesh.append(meshio.read(options.output_folder+'/Dense_surface_solution/ID_'+str(int(_id))+'/solution_iter_'+str(iter_value).zfill(3)+'.xdmf'))
		
		R_B_ECEF = Rot.from_quat(q[i])

		#Apply Displacement due to the use of FEniCS
		mesh[i].points += mesh[i].point_data['displacement']

		#Translate the assembly to (0,0,0)
		mesh[i].points -= np.array([body_X[i],body_Y[i],body_Z[i]])
		mesh[i].points = R_B_ECEF.apply(mesh[i].points)

		#Translate to the ECEF position
		mesh[i].points += np.array([X[i],Y[i],Z[i]])		

	#Place the heaviest object in the origin
	for i, _id in enumerate(assembly_ID):
		mesh[i].points -= np.array([X[index_mass],Y[index_mass],Z[index_mass]])
	
	if postprocess.lower() == "wind": 
		#Rotate ECEF -> wind_frame
		for i, _id in enumerate(assembly_ID):

			R_ECEF_NED = frames.R_NED_ECEF(lat = latitude[i], lon = longitude[i]).inv()
			R_NED_W = frames.R_W_NED(ha = chi[i], fpa = gamma[i]).inv()

			#R_ECEF_B = Rot.from_quat(q[i]).inv()
			#R_B_NED =   frames.R_B_NED(roll = roll[i], pitch = pitch[i], yaw = yaw[i]) 
			#R_NED_W = frames.R_W_NED(ha = chi[i], fpa = gamma[i]).inv()
			
			R_ECEF_W = R_NED_W*R_ECEF_NED
			mesh[i].points = (R_ECEF_W).apply(mesh[i].points)
	elif postprocess.lower() == "int":
		#Rotate ECEF -> largest object frame
		R_ECEF_B = Rot.from_quat(q[index_mass]).inv().as_matrix()
		for i, _id in enumerate(assembly_ID):
			
			
			mesh[i].points = mesh[i].points @ R_ECEF_B.T


	#Create new mesh
	points = mesh[0].points
	facets = mesh[0].cells[0].data
	pressure = mesh[0].cell_data['pressure']
	heatflux = mesh[0].cell_data['heatflux']
	temperature  = mesh[0].cell_data['temperature']
	debug_alpha = mesh[0].cell_data['debug_alpha']

	facet_dev = len(points)

	for i, _id in enumerate(assembly_ID):		
		if i == 0: continue
		points = np.append(points, mesh[i].points, axis = 0)
		facets = np.append(facets, mesh[i].cells[-1].data+facet_dev, axis = 0)
		pressure = np.append(pressure,mesh[i].cell_data['pressure'])
		heatflux = np.append(heatflux,mesh[i].cell_data['heatflux'])
		temperature = np.append(temperature, mesh[i].cell_data['temperature'])
		debug_alpha = np.append(debug_alpha, mesh[i].cell_data['debug_alpha'])

		facet_dev = len(points)

	cells = {"triangle": facets}

	cell_data = {"pressure": pressure,
                  "heatflux": heatflux,
                  "temperature": temperature,
				  "debug_alpha" : debug_alpha
				 }

	if len(assembly_ID) > 1:
		cell_data = {"pressure": [pressure],
                  "heatflux": [heatflux],
                  "temperature": [temperature],
				  "debug_alpha": [debug_alpha]
				 }	

	trimesh = meshio.Mesh(
        points,
        cells=cells,
        cell_data = cell_data)
	iter_write = iter_override if iter_override is not None else iter_value
	if not os.path.exists(options.output_folder+'/Postprocess_{}'.format(postprocess.lower())): os.mkdir(options.output_folder+'/Postprocess_{}'.format(postprocess.lower()))
	trimesh.write(options.output_folder+'/Postprocess_{}/'.format(postprocess.lower())+ 'solution_iter_' + str(iter_write).zfill(3)+'.xdmf')
