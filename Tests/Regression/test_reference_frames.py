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
sys.path.append('../../Dynamics')
from frames import *

x_vector = np.array([1,0,0])
y_vector = np.array([0,1,0])
z_vector = np.array([0,0,1])

#Test ECEF to REN
"""
def test_REN_to_ECEF_lat_0_lon_0():
	#North East Down
	#
	lat = 0
	lon = 0
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.array([1,0,0])).all()
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.array([0,1,0])).all()
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.array([0,0,1])).all()
"""

def test_NED_to_ECEF_lat_0_lon_0():
	#North East Down
	#
	lat = 0
	lon = 0
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.array([0,0,1])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.array([0,1,0])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.array([-1,0,0])).all()


"""
def test_REN_to_ECEF_lat_0_lon_90():
	lat = 0
	lon = np.pi/2
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.array([0,1,0])).all()
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.array([-1,0,0])).all()
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.array([0,0,1])).all()
"""

def test_NED_to_ECEF_lat_0_lon_90():
	lat = 0
	lon = np.pi/2
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.array([0,0,1])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.array([-1,0,0])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.array([0,-1,0])).all()

"""
def test_REN_to_ECEF_lat_0_lon_minus90():
	lat = 0
	lon = -np.pi/2
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.array([0,-1,0])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.array([1,0,0])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.array([0,0,1])).all()
"""

def test_NED_to_ECEF_lat_0_lon_minus90():
	lat = 0
	lon = -np.pi/2
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.array([0,0,1])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.array([1,0,0])).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.array([0,1,0])).all()

"""
def test_REN_to_ECEF_lat_45_lon_0():
	lat = np.pi/4
	lon = 0
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.round(np.array([np.sqrt(2)/2,0,np.sqrt(2)/2]),5)).all()
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_REN_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.round(np.array([-np.sqrt(2)/2,0,np.sqrt(2)/2]),5)).all()
"""

def test_NED_to_ECEF_lat_45_lon_0():
	lat = np.pi/4
	lon = 0
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(x_vector),5) == np.round(np.array([-np.sqrt(2)/2,0,np.sqrt(2)/2]),5)).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_NED_ECEF(lat=lat, lon=lon).apply(z_vector),5) == np.round(np.array([-np.sqrt(2)/2,0,-np.sqrt(2)/2]),5)).all()


#Test Wind to REN
#Z_Wind is pointing downwards
"""
def test_WIND_to_REN_fpa_0_ha_0():
	fpa = 0 #flight path angle
	ha = 0  #Heading angle

	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([0,0,1]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([-1,0,0]),5)).all()
"""

def test_WIND_to_NED_fpa_0_ha_0():
	fpa = 0 #flight path angle
	ha = 0  #Heading angle

	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([1,0,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()

"""
def test_WIND_to_REN_fpa_0_ha_90():
	fpa = 0 #flight path angle
	ha = np.pi/2  #Heading angle

	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,0,-1]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([-1,0,0]),5)).all()
"""

def test_WIND_to_NED_fpa_0_ha_90():
	fpa = 0 #flight path angle
	ha = np.pi/2  #Heading angle

	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([-1,0,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()

"""
def test_WIND_to_REN_fpa_90_ha_0():
	fpa = np.pi/2 #flight path angle
	ha = 0 #Heading angle

	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([1,0,0]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()
"""

def test_WIND_to_NED_fpa_90_ha_0():
	fpa = np.pi/2 #flight path angle
	ha = 0 #Heading angle

	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([0,0,-1]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([1,0,0]),5)).all()

"""
def test_WIND_to_REN_fpa_0_ha_45():
	fpa = 0 #flight path angle
	ha = np.pi/4 #Heading angle

	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([0,np.sqrt(2)/2,np.sqrt(2)/2]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,np.sqrt(2)/2,-np.sqrt(2)/2]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([-1,0,0]),5)).all()
"""

def test_WIND_to_NED_fpa_0_ha_45():
	fpa = 0 #flight path angle
	ha = np.pi/4 #Heading angle

	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([np.sqrt(2)/2,np.sqrt(2)/2,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([-np.sqrt(2)/2,np.sqrt(2)/2,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()

"""
def test_WIND_to_REN_fpa_minus45_ha_45():
	fpa = -np.pi/4 #flight path angle
	ha = np.pi/4 #Heading angle

	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([-np.sqrt(2)/2,0.5,0.5]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([0,np.sqrt(2)/2,-np.sqrt(2)/2]),5)).all()
	assert (np.round(R_W_REN(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([-np.sqrt(2)/2,-0.5,-0.5]),5)).all()
"""

def test_WIND_to_NED_fpa_minus45_ha_45():
	fpa = -np.pi/4 #flight path angle
	ha = np.pi/4 #Heading angle

	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(x_vector),5) == np.round(np.array([0.5,0.5,np.sqrt(2)/2]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(y_vector),5) == np.round(np.array([-np.sqrt(2)/2,np.sqrt(2)/2,0]),5)).all()
	assert (np.round(R_W_NED(fpa = fpa, ha = ha).apply(z_vector),5) == np.round(np.array([-0.5,-0.5,np.sqrt(2)/2]),5)).all()

#Test Body to Wind
def test_BODY_to_WIND_aoa_0_slip_0():
	aoa = 0
	slip = 0

	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(x_vector),5) == np.round(np.array([1,0,0]),5)).all()
	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()

def test_BODY_to_WIND_aoa_45_slip_0():
	aoa = np.pi/4
	slip = 0

	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(x_vector),5) == np.round(np.array([np.sqrt(2)/2,0,-np.sqrt(2)/2]),5)).all()
	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(y_vector),5) == np.round(np.array([0,1,0]),5)).all()
	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(z_vector),5) == np.round(np.array([np.sqrt(2)/2,0,np.sqrt(2)/2]),5)).all()

def test_BODY_to_WIND_aoa_0_slip_45():
	aoa = 0
	slip = np.pi/4

	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(x_vector),5) == np.round(np.array([np.sqrt(2)/2,np.sqrt(2)/2,0]),5)).all()
	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(y_vector),5) == np.round(np.array([-np.sqrt(2)/2, np.sqrt(2)/2,0]),5)).all()
	assert (np.round(R_B_W(aoa = aoa, slip = slip).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()


def test_WIND_to_BODY_aoa_0_slip_90():
	aoa = 0
	slip = np.pi/2

	assert (np.round(R_W_B(aoa = aoa, slip = slip).apply(x_vector),5) == np.round(np.array([0,-1,0]),5)).all()
	assert (np.round(R_W_B(aoa = aoa, slip = slip).apply(y_vector),5) == np.round(np.array([1,0,0]),5)).all()
	assert (np.round(R_W_B(aoa = aoa, slip = slip).apply(z_vector),5) == np.round(np.array([0,0,1]),5)).all()

"""
def test_BODY_to_NED_0_0_0():
	roll = np.pi
	pitch = 0
	yaw = np.pi
	
	print(R_B_NED(roll = roll, pitch = pitch, yaw = yaw))
"""

