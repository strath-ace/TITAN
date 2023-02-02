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

planet_data = {'earth':{'G': 6.6743E-11,'M': 5.972E24, 'P': 86164, 'w': 7.2921150E-5},
			   'uranus':{'G': 6.6743E-11,'M': 86.8127E24, 'P': 62064, 'a0': 25559, 'J2': 3510.7E-6, 'J4': -34.2E-6, 'w':1.0123882E-4 },
			   'neptune':{'G': 6.6743E-11,'M': 102.4126E24, 'P': 57996, 'a0': 24764, 'J2': 3536.5E-6, 'J4': -36.0E-6, 'w':1.0833825E-4}}

#Values retrieved from pymap3d github: https://github.com/geospace-code/pymap3d/blob/803407d063f1ead2b9f93f3c1c74767c8dda5c3e/src/pymap3d/ellipsoid.py
ellipsoid = {'earth':{"name": "WGS-84 (1984)", "a": 6378137.0, "b": 6356752.31424518},
			 'uranus':{"name": "uranus", "a": 25559000.0, "b": 24973000.0},
			 'neptune':{"name": "Neptune", "a": 24764000.0, "b": 24341000.0}}

def Legendre(n, x):
	L = 1
	if n == 1: L = x
	if n == 2: L = 0.5*(3*x**2-1)
	if n == 3: L = 0.5*(5*x**3-3*x)
	if n == 4: L = 1/8*(35*x**4-30*x**2+3)
	return L

class ModelPlanet():
	def __init__(self, name = "Earth"):
		self.name = name.lower()

	def mass(self):
		return planet_data[self.name]['M']

	def constant(self):
		return planet_data[self.name]['G']

	def period(self):
		return planet_data[self.name]['P']

	def J2(self):
		return planet_data[self.name]['J2']

	def J4(self):
		return planet_data[self.name]['J4']

	def a0(self):
		return planet_data[self.name]['a0']

	def omega(self):
		return planet_data[self.name]['w']

	def gravitationalAcceleration(self,r, phi):
		
		gr = -self.mass()*self.constant()/(r**2)

		if self.name != "earth":
			gr += self.mass()*self.constant()*3/(r**4)*(self.a0()**2)*self.J2()*Legendre(2,np.cos(phi))
			gr += self.mass()*self.constant()*5/(r**6)*(self.a0()**4)*self.J4()*Legendre(4,np.cos(phi))

		gt = 0
		return gr,gt

	def ellipsoid(self):
		return ellipsoid[self.name]