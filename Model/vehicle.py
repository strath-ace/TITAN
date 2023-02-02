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
import numpy as np
from Model import drag_model

"""
vehicle_data = {"galileo":{'mass': 341, 'Rn': 1.25/(2)*0.352, 'D': 1.25,  'Aref':(1.25/2)**2*np.pi , 'Cd': lambda x: drag_model.drag_galileo(x)},
				"galileo_nasa":{'mass': 325, 'Rn': 1.20/(2)*0.352, 'D': 1.20,  'Aref':(1.20/2)**2*np.pi , 'Cd': lambda x: drag_model.drag_galileo(x)},
				"galileo-120":{'mass': 308, 'Rn': 1.20/(2)*0.352, 'D': 1.20,  'Aref':(1.20/2)**2*np.pi , 'Cd': lambda x: drag_model.drag_galileo(x)}}
"""

class ModelVehicle():
	def __init__(self):
		self.mass = 0
		self.noseRadius = 0
		self.Aref = 0
		self.Cd = None