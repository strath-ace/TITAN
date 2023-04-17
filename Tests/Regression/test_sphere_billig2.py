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
import pytest
import sys
import numpy as np
from TITAN import main
import meshio

options, titan = main("Tests/Configs/1m_sphere_CFD.txt")


TITAN_mesh = meshio.read("Tests/Simulation/Surface_solution/ID_1/solution_iter_000.vtk")
SU2_mesh = meshio.read("Tests/Simulation/CFD_sol/surface_flow_0_1_cluster_0.vtk")

def test_pressure():
	assert all(np.isclose(np.round(np.sort(TITAN_mesh.point_data["Pressure"]),2),np.round(np.sort(SU2_mesh.point_data["Pressure"].reshape(-1)),2), 0.1))

def test_heatflux():
	assert all(np.isclose(np.round(np.sort(TITAN_mesh.point_data["Heatflux"]),2),np.round(np.sort(SU2_mesh.point_data["Heat_Flux"].reshape(-1)),2), 0.1))

def test_shear():
	assert all(np.isclose(np.round(np.sort(TITAN_mesh.point_data["Shear"][:,0]),2),np.round(0.5*titan.assembly[0].freestream.density*titan.assembly[0].freestream.velocity**2*np.sort(SU2_mesh.point_data["Skin_Friction_Coefficient"][:,0]),2), 0.1))
	assert all(np.isclose(np.round(np.sort(TITAN_mesh.point_data["Shear"][:,1]),2),np.round(0.5*titan.assembly[0].freestream.density*titan.assembly[0].freestream.velocity**2*np.sort(SU2_mesh.point_data["Skin_Friction_Coefficient"][:,1]),2), 0.1))
	assert all(np.isclose(np.round(np.sort(TITAN_mesh.point_data["Shear"][:,2]),2),np.round(0.5*titan.assembly[0].freestream.density*titan.assembly[0].freestream.velocity**2*np.sort(SU2_mesh.point_data["Skin_Friction_Coefficient"][:,2]),2), 0.1))