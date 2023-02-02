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
from bs4 import BeautifulSoup
import numpy as np
from scipy import interpolate
import os

#### LIST OF MATERIALS

# 'drama-AA2195 (Al-Li)', 'drama-AA7075', 'drama-A316', 'drama-Bat-Li', 'drama-Bat-NiCd', 'drama-Beryllium', 'drama-Brass',
# 'drama-Carbon-Carbon', 'drama-Copper', 'drama-El-Mat', 'drama-HC-AA7075', 'drama-HC-CFRP-4ply', 'drama-HC-CFRP-8ply',
# 'drama-Inconel', 'drama-Inermet', 'drama-Invar', 'drama-Iron', 'drama-SiC', 'drama-SolarPanel-Mat', 'drama-TiAl6v4',
# 'drama-Tungsten', 'drama-CFRP'

### LIST of atributes inside MetaMaterial class

# density
# specificHeatCapacity
# meltingHeat
# meltingTemperature
# heatConductivity
# heatConductivityVirgin
# emissivity
# oxideActivationTemperature
# oxideEmmisivity
# oxideHeatOfFormation
# oxideReactionProbability

class Material():
	""" Class Material
	
	A class to store the material properties for each user-defined component
	"""
	
	def __init__ (self, name):
		with open(os.path.dirname(os.path.abspath(__file__))+'/database_material.xml', 'r') as f:
			data = f.read()
		 
		# Passing the stored data inside
		# the beautifulsoup parser, storing
		# the returned object

		Bs_data = BeautifulSoup(data, "xml")
		 
		# Finding all instances of tag 'name'
		names_bs4 = Bs_data.find_all('name')

		names = []
		index = -1

		for _index,_name in enumerate(names_bs4):
			if name in _name.get_text():
				index=_index
				break

		if index == -1: print("Material Name does not exist"); exit()

		#TODO
		#Check if more materials are missing
		#Change metalMaterial for other class name
		self.metalMaterial = Bs_data.find_all('metalMaterial')[index]

		#: [str] Name of the material
		self.name = self.material_name(index)
		
		#: [float] Density of the material
		self.density = self.material_density(index)

		#: [float] Specific Heat Capacity value of the material
		self.specificHeatCapacity = self.material_specificHeatCapacity(index)  #function of Temperature
		
		#: [float] Melting Heat value of the material
		self.meltingHeat = self.material_meltingHeat(index)

		#: [float] Melting Temperature value
		self.meltingTemperature = self.material_meltingTemperature(index)

		#: [float] Emissivity value
		self.emissivity = self.material_emissivity(index)                      #function of Temperature

		#: [float] Heat conductivity value
		self.heatConductivity = self.material_heatConductivity(index)          #function of Temperature

		#: [float] Oxidation activation temperature value
		self.oxideActivationTemperature = self.material_oxideActivationTemperature(index)

		#: [float] Oxidation emissivity value
		self.oxideEmissivity = self.material_oxideEmissivity(index)            #function of Temperature

		#: [float] Oxidation Formation Heat value
		self.oxideHeatOfFormation = self.material_oxideHeatOfFormation(index)

		#: [float] Oxidation reaction probability value
		self.oxideReactionProbability = self.material_oxideReactionProbability(index)
		try:
			#: [float] Young Modulus value
			self.youngModulus = self.material_youngModulus(index) #function of Temperature
			
			#: [float] Yield Stress value
			self.yieldStress = self.material_yieldStress(index) #function of Temperature
		except:
			self.youngModulus = None
			self.yieldStress = None


	def material_name(self,index):
		"""
		Function to retrieve the material name

		Returns
		-------
		name: str
			Return material name
		"""

		return self.metalMaterial.find('name').get_text()

	def material_density(self,index):
		"""
		Function to retrieve the material density

		Returns
		-------
		density: float
			Return material density
		"""

		return float(self.metalMaterial.find('density').get_text())

	def material_specificHeatCapacity(self,index):
		"""
		Function to retrieve the material specific heat capacity

		Returns
		-------
		specificHeatCapacity : scipy.interpolate.interp1d
			Return interpolation function for the specific heat capacity
		"""

		values = np.array(self.metalMaterial.find('specificHeatCapacity').find('values').get_text().replace(',',';').split(';'))[:-1].astype(float)
		values.shape = (-1,2)

		values_T = values[:,0]
		values_Y = values[:,1]

		return interpolate.interp1d(values_T, values_Y, fill_value='extrapolate')

	def material_meltingHeat(self,index):
		"""
		Function to retrieve the melting Heat value
		
		Returns
		-------
		meltingHeat: float
			Return melting heat value
		"""

		return float(self.metalMaterial.find('meltingHeat').get_text())

	def material_meltingTemperature(self,index):
		"""
		Function to retrieve the melting temperature value
		
		Returns
		-------
		meltingTemperature: float
			Return melting temperature value
		"""

		return float(self.metalMaterial.find('meltingTemperature').get_text())

	def material_emissivity(self,index):
		"""
		Function to retrieve the emissivity value
		
		Returns
		-------
		emissivity: float
			Return emissivity value
		"""

		values = np.array(self.metalMaterial.find('emissivity').find('values').get_text().replace(',',';').split(';'))[:-1].astype(float)
		values.shape = (-1,2)

		values_T = values[:,0]
		values_Y = values[:,1]

		if len(values_T) == 1:
			values_T = np.array([values_T[0],10000.0])
			values_Y = np.array([values_Y[0],values_Y[0]])
		return interpolate.interp1d(values_T, values_Y, fill_value='extrapolate')

	def material_heatConductivity(self,index):
		"""
		Function to retrieve the material heat conductivity

		Returns
		-------
		heatConductivity : scipy.interpolate.interp1d
			Return interpolation function for the heat conductivity
		"""

		values = np.array(self.metalMaterial.find('heatConductivity').find('values').get_text().replace(',',';').split(';'))[:-1].astype(float)
		values.shape = (-1,2)

		values_T = values[:,0]
		values_Y = values[:,1]

		return interpolate.interp1d(values_T, values_Y, fill_value='extrapolate')	

	def material_oxideActivationTemperature(self,index):
		"""
		Function to retrieve the oxide activation Temperatire
		
		Returns
		-------
		oxideActivationTemperature: float
			Return oxide activation temperature value
		"""

		return float(self.metalMaterial.find('oxideActivationTemperature').get_text())

	def material_oxideEmissivity(self,index):
		"""
		Function to retrieve the material oxide emissivity

		Returns
		-------
		oxideEmissivity: scipy.interpolate.interp1d
			Return interpolation function for the oxide emissivity
		"""

		try:
			values = np.array(self.metalMaterial.find('oxideEmissivity').find('values').get_text().replace(',',';').split(';'))[:-1].astype(float)
				
			values.shape = (-1,2)

			values_T = values[:,0]
			values_Y = values[:,1]

		except:

			values_T = np.array([0.0,10000.0])
			values_Y = np.array([0.0,0.0])

		return interpolate.interp1d(values_T, values_Y, fill_value='extrapolate')


	def material_oxideHeatOfFormation(self,index):
		"""
		Function to retrieve the oxide heat of formation
		
		Returns
		-------
		oxideHeatofFormation: float
			Return oxide heat of formation value
		"""

		return float(self.metalMaterial.find('oxideHeatOfFormation').get_text())

	def material_oxideReactionProbability(self,index):
		"""
		Function to retrieve the oxide reaction probability
		
		Returns
		-------
		oxideReactionProbability: float
			Return oxide reaction probability
		"""
		
		return float(self.metalMaterial.find('oxideReactionProbability').get_text())

	def material_youngModulus(self,index):
		"""
		Function to retrieve the young Modulus

		Returns
		-------
		youngModulus: scipy.interpolate.interp1d
			Return interpolation function for the young Modulus
		"""

		values = np.array(self.metalMaterial.find('youngModulus').find('values').get_text().replace(',',';').split(';'))[:-1].astype(float)
		values.shape = (-1,2)

		values_T = values[:,0]
		values_Y = values[:,1]

		return interpolate.interp1d(values_T, values_Y, fill_value='extrapolate')

	def material_yieldStress(self,index):
		"""
		Function to retrieve the material yield stress

		Returns
		-------
		yieldStress: scipy.interpolate.interp1d
			Return interpolation function for the yield Stress
		"""

		values = np.array(self.metalMaterial.find('yieldStress').find('values').get_text().replace(',',';').split(';'))[:-1].astype(float)
		values.shape = (-1,2)

		values_T = values[:,0]
		values_Y = values[:,1]

		return interpolate.interp1d(values_T, values_Y, fill_value='extrapolate')	