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

options, titan = main("Tests/Configs/2cube_mass.txt")

#Test assembly mass
def test_assembly_mass(): assert np.round(titan.assembly[0].mass,5) == np.round(2097.55900582,5)

#Test assembly inertia
def test_assembly_inertia_0(): assert np.round(titan.assembly[0].inertia[0,0],5) ==  np.round(3.36362957e+02,5)
def test_assembly_inertia_1(): assert np.round(titan.assembly[0].inertia[1,1],5) ==  np.round(1.46188072e+03,5)
def test_assembly_inertia_2(): assert np.round(titan.assembly[0].inertia[2,2],5) ==  np.round(1.46188030e+03,5)
