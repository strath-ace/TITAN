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
"""mix_mpp module."""
try:
    import mutationpp as mpp
except:
    print("Mutationpp library not set up")


def mixture_mpp(species, density, temperature):
    """Retrieve the mixture object of the Mutation++ library
:param species: Value for species.
:type species: Any
:param density: Numeric value for density.
:type density: float
:param temperature: Numeric value for temperature.
:type temperature: float
:return: Return value.
:rtype: Any"""

    opts = mpp.MixtureOptions()
    
    opts.setSpeciesDescriptor(' '.join(species))
    opts.setThermodynamicDatabase("RRHO");
    opts.setStateModel("ChemNonEq1T");

    opts.setViscosityAlgorithm("Wilke");

    mix = mpp.Mixture(opts)
    mix.setState(density, temperature, 1);

    return mix
