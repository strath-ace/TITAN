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

options, titan = main("Tests/Configs/1m_sphere_rarefied.txt")

def test_pressure(): assert np.round(np.max(titan.assembly[0].aerothermo.pressure),5) == np.round(0.00048,5)
def test_heatflux(): assert np.round(np.max(titan.assembly[0].aerothermo.heatflux),5) == np.round(0.33274,5)
def test_shear(): assert np.round(np.max(titan.assembly[0].aerothermo.shear),7) == np.round(0.0001159,7)