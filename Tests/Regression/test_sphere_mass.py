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
#sys.path.append('../')
from TITAN import main

options, titan = main("Tests/Configs/1m_sphere.txt")

#Test assembly mass
def test_assembly_mass(): assert np.round(titan.assembly[0].mass,5) == np.round(4174.100699031239,5)

#Test assembly inertia
def test_assembly_inertia_0(): assert np.round(titan.assembly[0].inertia[0,0],5) ==  np.round(1.66572056e+03,5)
def test_assembly_inertia_1(): assert np.round(titan.assembly[0].inertia[1,1],5) ==  np.round(1.66571801e+03,5)
def test_assembly_inertia_2(): assert np.round(titan.assembly[0].inertia[2,2],5) ==  np.round(1.66576779e+03,5)

#Test object mass
def test_object_mass(): assert np.round(titan.assembly[0].objects[0].mass,5) == np.round(4174.100699031239,5)

#Test object inertia
def test_object_inertia_0(): assert np.round(titan.assembly[0].objects[0].inertia[0,0],5) ==  np.round(1.66572056e+03,5)
def test_object_inertia_1(): assert np.round(titan.assembly[0].objects[0].inertia[1,1],5) ==  np.round(1.66571801e+03,5)
def test_object_inertia_2(): assert np.round(titan.assembly[0].objects[0].inertia[2,2],5) ==  np.round(1.66576779e+03,5)

#Test assembly and object COG
def test_assembly_COG_0(): assert np.round(titan.assembly[0].COG[0],5) == np.round(2.33553067e-06,5)
def test_assembly_COG_1(): assert np.round(titan.assembly[0].COG[1],5) == np.round(2.17407358e-06,5)
def test_assembly_COG_2(): assert np.round(titan.assembly[0].COG[2],5) == np.round(-2.55888820e-08,5)
def test_object_COG_0(): assert np.round(titan.assembly[0].objects[0].COG[0],5) == np.round(2.33553067e-06,5)
def test_object_COG_1(): assert np.round(titan.assembly[0].objects[0].COG[1],5) == np.round(2.17407358e-06,5)
def test_object_COG_2(): assert np.round(titan.assembly[0].objects[0].COG[2],5) == np.round(-2.55888820e-08,5)
