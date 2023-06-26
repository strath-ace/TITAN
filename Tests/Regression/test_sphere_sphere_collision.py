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
sys.path.append('../')
from TITAN import main

options, titan = main("Tests/Configs/sphere-sphere.txt")

print(titan.assembly[0].position, titan.assembly[1].position)

def test_position_post_collision():
	assert (np.round(titan.assembly[0].position[0],2) == np.round(7.07812896E+06,2))
	assert (np.round(titan.assembly[0].position[1],5) == np.round(-7.64165177E-01,5))
	assert (np.round(titan.assembly[0].position[2],5) == np.round(-3.29717107E+00,5))


	assert (np.round(titan.assembly[1].position[0],2) == np.round(7.07812890E+06,2))
	assert (np.round(titan.assembly[1].position[1],5) == np.round(6.32614291E-01,5))
	assert (np.round(titan.assembly[1].position[2],5) == np.round(2.78064246E+00,5))

#def test_pressure(): assert np.round(np.max(titan.assembly[0].aerothermo.pressure),5) == np.round(353018.08191,5)
#def test_heatflux(): assert np.round(np.max(titan.assembly[0].aerothermo.heatflux),5) == np.round(74730.60448,5)
