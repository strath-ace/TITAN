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
import os
import meshio

def test_fenics():
	options, titan1 = main("Tests/Configs/ATV_FENICS.txt")

#def test_pressure():
#	assert all(np.isclose(np.round(np.sort(np.append(titan1.assembly[0].aerothermo_cfd.pressure,titan1.assembly[1].aerothermo_cfd.pressure)),2),np.round(np.sort(SU2_mesh.point_data["Pressure"].reshape(-1)),2), 0.1))