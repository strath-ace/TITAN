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
from scipy.interpolate import interp1d
import pandas as pd
import os

def read_csv(filename = ""):
	path = os.path.dirname(os.path.abspath(__file__))+'/Drag/'+filename
	g_cd = pd.read_csv(path)
	f = interp1d(g_cd.iloc[:,0], g_cd.iloc[:,1], kind = 'linear', fill_value = 'extrapolate')
	return f

def drag_galileo(Mach):
	path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	g_cd = pd.read_csv(path+'/Model/Drag/Galileo_CD.csv')
	f = interp1d(g_cd.iloc[:,0], g_cd.iloc[:,1], kind = 'linear', fill_value = 'extrapolate')
	CD = f(Mach)

	return CD.item()


