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

#X_X_X_X Stands for Latitude, Longitude, AoA, Sideslip
 
options, titan_0 = main("Tests/Configs/1m_sphere_ground_0_0_0_0.txt")
def test_latitude0(): assert np.round(np.max(titan_0.assembly[0].trajectory.latitude),5) == np.round(0.27161,5)
def test_longitude0(): assert np.round(np.max(titan_0.assembly[0].trajectory.longitude),5) == np.round(4.89426e-07,5)

options, titan_1 = main("Tests/Configs/1m_sphere_ground_45_45_0_0.txt")
def test_latitude1(): assert np.round(np.max(titan_1.assembly[0].trajectory.latitude),5) == np.round(1.05166,5)
def test_longitude1(): assert np.round(np.max(titan_1.assembly[0].trajectory.longitude),5) == np.round(0.7854,5)

options, titan_2 = main("Tests/Configs/1m_sphere_ground_0_0_45_45.txt")
def test_latitude2(): assert np.round(np.max(titan_2.assembly[0].trajectory.latitude),5) == np.round(0.17249,5)
def test_longitude2(): assert np.round(np.max(titan_2.assembly[0].trajectory.longitude),5) == np.round(-0.20775,5)
