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
sys.path.append('Material')
from material import Material

material=Material("Unittest")

def test_density(): assert material.density == 1000.0, "Density of the material is different"
def test_specificHeatCapacity(): assert material.specificHeatCapacity(293) == 877.5, " specificHeatCapacity is different "
def test_meltingHeat(): assert material.meltingHeat == 361500.0, "Melting heat is different"
def test_meltingTemperature(): assert material.meltingTemperature == 906.15, "melting Temperature is different"
def test_emissivity(): assert material.emissivity(250) == 0.105, "melting Temperature is different"
def test_heatConductivity(): assert material.heatConductivity(293) == 163.89
def test_oxideActivationTemperature(): assert material.oxideActivationTemperature == 0.0
def test_oxideEmissivity(): assert material.oxideEmissivity(300) == 0.0
def test_oxideHeatofFormation(): assert material.oxideHeatOfFormation == 0.0
def test_oxideReactionProbabilty(): assert material.oxideReactionProbability == 0.0
