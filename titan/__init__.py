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

"""
TransatmospherIc flighT simulAtioN (TITAN) - A python code for multi-fidelity 
and multi-physics simulations of access-to-space and re-entry.
"""

__version__ = "0.1.0"
__author__ = "TITAN Contributors"

# Import subpackages to make them available at package level
from . import (
    Aerothermo,
    Configuration,
    Dynamics,
    Explosion,
    Forces,
    Fragmentation,
    Freestream,
    Geometry,
    Material,
    Model,
    Output,
    Postprocess,
    Structural,
    Thermal,
    Uncertainty,
)

# Convenience imports - commonly used modules
from .Configuration import configuration
from .Output import output, dynamic_plots
from .Dynamics import dynamics, propagation
from .Fragmentation import fragmentation
from .Postprocess import postprocess
from .Postprocess import postprocess_emissions
from .Thermal import thermal
from .Structural import structural

__all__ = [
    'Aerothermo',
    'Configuration',
    'Dynamics',
    'Explosion',
    'Forces',
    'Fragmentation',
    'Freestream',
    'Geometry',
    'Material',
    'Model',
    'Output',
    'Postprocess',
    'Structural',
    'Thermal',
    'Uncertainty',
    'configuration',
    'output',
    'dynamic_plots',
    'dynamics',
    'propagation',
    'fragmentation',
    'postprocess',
    'postprocess_emissions',
    'thermal',
    'structural',
]


def main(filename="", postprocess="", filter_name=None, emissions=""):
    """Run the TITAN command-line workflow through the package API.

:param filename: Configuration file used to initialize the simulation.
:type filename: str
:param postprocess: Optional post-processing mode to run after simulation.
:type postprocess: str
:param filter_name: Optional component-name filter for post-processing.
:type filter_name: str or None
:param emissions: Optional emissions-processing mode.
:type emissions: str
:return: Result returned by the command-line entry point.
:rtype: Any
"""
    from .__main__ import main as run_main
    return run_main(filename, postprocess, filter_name, emissions)
