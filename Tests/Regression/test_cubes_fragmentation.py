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

options, titan1 = main("Tests/Configs/2cube_frag_altitude.txt")

#ALtitude fragmentation
def test_frag_altitude(): assert len(titan1.assembly) == 2

options, titan2 = main("Tests/Configs/2cube_frag_temperature.txt")

#Temperature fragmentation
def test_frag_temperature(): assert len(titan2.assembly) == 2

#Testing temperature in addition to fragmentation

#### Using old backface culling algo
#def test_temperature(): assert np.round(titan2.assembly[0].objects[0].temperature,5) == np.round(382.09268330658256,5)

### Using new backface culling algo
def test_temperature(): assert np.round(titan2.assembly[0].objects[0].temperature,5) == np.round(400.87135,5)

options, titan3 = main("Tests/Configs/cube_demise.txt")

#Test demise
def test_demise(): assert len(titan3.assembly) == 0
