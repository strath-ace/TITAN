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
from TITAN import main

options, titan = main("Tests/Configs/1m_sphere.txt")

#Compare Assembly and obj mesh
def test_assembly_mesh_v0(): assert (titan.assembly[0].mesh.v0 == titan.assembly[0].objects[0].mesh.v0).all(), "  Mesh v0 is not the same"
def test_assembly_mesh_v1(): assert (titan.assembly[0].mesh.v1 == titan.assembly[0].objects[0].mesh.v1).all(), "  Mesh v1 is not the same"
def test_assembly_mesh_v2(): assert (titan.assembly[0].mesh.v2 == titan.assembly[0].objects[0].mesh.v2).all(), "  Mesh v2 is not the same"

def test_assembly_mesh_facet(): assert (titan.assembly[0].mesh.facets == titan.assembly[0].objects[0].mesh.facets).all(), " Facets are not the same"
def test_assembly_mesh_facet_normal(): assert (titan.assembly[0].mesh.facet_normal == titan.assembly[0].objects[0].mesh.facet_normal).all(), " Facet normals are not the same"
def test_assembly_mesh_facet_edges(): assert (titan.assembly[0].mesh.facet_edges== titan.assembly[0].objects[0].mesh.facet_edges).all(), " Facet edges are not the same"
def test_assembly_mesh_facet_area(): assert (titan.assembly[0].mesh.facet_area == titan.assembly[0].objects[0].mesh.facet_area).all(), " Facet areas are not the same"
def test_assembly_mesh_facet_COG(): assert (titan.assembly[0].mesh.facet_COG == titan.assembly[0].objects[0].mesh.facet_COG).all(), " Facet COG are not the same"

def test_assembly_mesh_nodes(): assert (titan.assembly[0].mesh.nodes == titan.assembly[0].objects[0].mesh.nodes).all(), " Nodes are not the same"
def test_assembly_mesh_nodes_normal(): assert (titan.assembly[0].mesh.nodes_normal == titan.assembly[0].objects[0].mesh.nodes_normal).all(), " Nodes normal are not the same"
def test_assembly_mesh_nodes_radius(): assert (titan.assembly[0].mesh.nodes_radius == titan.assembly[0].objects[0].mesh.nodes_radius).all(), " Nodes radius are not the same"

def test_assembly_mesh_edges(): assert (titan.assembly[0].mesh.edges == titan.assembly[0].objects[0].mesh.edges).all(), " Nodes radius are not the same"

###-------------------------------------------------------------------------------------------------------------------------------

def test_assembly_mesh_v0_0(): assert (np.round(titan.assembly[0].mesh.v0[0],5) == np.round(np.array([ 6.12323400e-17, -1.49975978e-32,  1.00000000e+00]),5)).all(), "  Mesh v0[0] is wrong"
def test_assembly_mesh_v0_1(): assert (np.round(titan.assembly[0].mesh.v0[1],5) == np.round(np.array([ 5.18588318e-01,  1.02593241e-01, -8.48846737e-01]),5)).all(), "  Mesh v0[1] is wrong"
def test_assembly_mesh_v0_2(): assert (np.round(titan.assembly[0].mesh.v0[2],5) == np.round(np.array([ 2.17011309e-01,  9.76019836e-01,  1.70695867e-02]),5)).all(), "  Mesh v0[2] is wrong"

def test_assembly_mesh_v1_0(): assert (np.round(titan.assembly[0].mesh.v1[0],5) == np.round(np.array([-0.08220775, -0.09373043,  0.99219781]),5)).all(), "  Mesh v1[0] is wrong"
def test_assembly_mesh_v1_1(): assert (np.round(titan.assembly[0].mesh.v1[1],5) == np.round(np.array([ 0.43457881,  0.18293856, -0.88185869]),5)).all(), "  Mesh v1[1] is wrong"
def test_assembly_mesh_v1_2(): assert (np.round(titan.assembly[0].mesh.v1[2],5) == np.round(np.array([ 0.13186685,  0.98985423, -0.05291252]),5)).all(), "  Mesh v1[2] is wrong"

def test_assembly_mesh_v2_0(): assert (np.round(titan.assembly[0].mesh.v2[0],5) == np.round(np.array([ 0.04889029, -0.08495351,  0.99518473]),5)).all(), "  Mesh v2[0] is wrong"
def test_assembly_mesh_v2_1(): assert (np.round(titan.assembly[0].mesh.v2[1],5) == np.round(np.array([ 0.51421239,  0.21033004, -0.83147272]),5)).all(), "  Mesh v2[1] is wrong"
def test_assembly_mesh_v2_2(): assert (np.round(titan.assembly[0].mesh.v2[2],5) == np.round(np.array([ 0.13245758,  0.98975144,  0.05335796]),5)).all(), "  Mesh v2[2] is wrong"

def test_assembly_mesh_facets_0(): assert (titan.assembly[0].mesh.facets[0] == [ 794,  727,  838]).all(), " Mesh facets are wrong"
def test_assembly_mesh_facets_1(): assert (titan.assembly[0].mesh.facets[1] == [1199, 1133, 1195]).all(), " Mesh facets are wrong"
def test_assembly_mesh_facets_2(): assert (titan.assembly[0].mesh.facets[2] == [ 970,  901,  903]).all(), " Mesh facets are wrong"

def test_assembly_mesh_facet_normal_0(): assert (np.round(titan.assembly[0].mesh.facet_normal[0],5) ==  np.round(np.array([-1.82404047e-02, -6.70415554e-02,  9.97583439e-01]),5)).all(), " Mesh facet normals are wrong"
def test_assembly_mesh_facet_normal_1(): assert (np.round(titan.assembly[0].mesh.facet_normal[1],5) ==  np.round(np.array([ 4.88511432e-01,  1.58220786e-01, -8.58092515e-01]),5)).all(), " Mesh facet normals are wrong"
def test_assembly_mesh_facet_normal_2(): assert (np.round(titan.assembly[0].mesh.facet_normal[2],5) ==  np.round(np.array([ 1.60327372e-01,  9.87063893e-01,  6.35085564e-05]),5)).all(), " Mesh facet normals are wrong"

def test_assembly_mesh_facet_edges_0(): assert (titan.assembly[0].mesh.facet_edges[0] ==  [-2250,  2252, -2451]).all(), " Mesh facet edges are wrong"
def test_assembly_mesh_facet_edges_1(): assert (titan.assembly[0].mesh.facet_edges[1] ==  [-3462,  3461,  3643]).all(), " Mesh facet edges are wrong"
def test_assembly_mesh_facet_edges_2(): assert (titan.assembly[0].mesh.facet_edges[2] ==  [-2773,  2771,  2779]).all(), " Mesh facet edges are wrong"

def test_assembly_mesh_facet_area_0(): assert np.round(titan.assembly[0].mesh.facet_area[0],5) == np.round(0.00579718,5), " Mesh facet areas are wrong" 
def test_assembly_mesh_facet_area_1(): assert np.round(titan.assembly[0].mesh.facet_area[1],5) == np.round(0.00506899,5), " Mesh facet areas are wrong"
def test_assembly_mesh_facet_area_2(): assert np.round(titan.assembly[0].mesh.facet_area[2],5) == np.round(0.00456252,5), " Mesh facet areas are wrong"

def test_assembly_mesh_facet_COG_0(): assert (np.round(titan.assembly[0].mesh.facet_COG[0],5) ==  np.round(np.array([-0.01110582, -0.05956131,  0.99579418]),5)).all(), " Mesh facet COG are wrong"
def test_assembly_mesh_facet_COG_1(): assert (np.round(titan.assembly[0].mesh.facet_COG[1],5) ==  np.round(np.array([ 0.48912651,  0.16528728, -0.85405938]),5)).all(), " Mesh facet COG are wrong"
def test_assembly_mesh_facet_COG_2(): assert (np.round(titan.assembly[0].mesh.facet_COG[2],5) ==  np.round(np.array([ 0.16044525,  0.9852085 ,  0.00583834]),5)).all(), " Mesh facet COG are wrong"

def test_assembly_mesh_nodes_0(): assert (np.round(titan.assembly[0].mesh.nodes[0],5) ==  np.round(np.array([-0.99944292, -0.01591856, -0.02933342]),5)).all(), " Mesh nodes are wrong"
def test_assembly_mesh_nodes_1(): assert (np.round(titan.assembly[0].mesh.nodes[1],5) ==  np.round(np.array([-0.99723318,  0.0284138 ,  0.06869239]),5)).all(), " Mesh nodes are wrong"
def test_assembly_mesh_nodes_2(): assert (np.round(titan.assembly[0].mesh.nodes[2],5) ==  np.round(np.array([-0.99609326, -0.06815389,  0.05615399]),5)).all(), " Mesh nodes are wrong"

def test_assembly_mesh_nodes_normal_0(): assert (np.round(titan.assembly[0].mesh.nodes_normal[0],5) ==  np.round(np.array([-9.63996430e-03, -9.19181309e-05, -2.83637954e-04]),5)).all(), " Mesh nodes normals are wrong"
def test_assembly_mesh_nodes_normal_1(): assert (np.round(titan.assembly[0].mesh.nodes_normal[1],5) ==  np.round(np.array([-8.61936882e-03,  2.46926035e-04,  5.68409836e-04]),5)).all(), " Mesh nodes normals are wrong"
def test_assembly_mesh_nodes_normal_2(): assert (np.round(titan.assembly[0].mesh.nodes_normal[2],5) ==  np.round(np.array([-7.31633426e-03, -4.65852176e-04,  4.29543309e-04]),5)).all(), " Mesh nodes normals are wrong"

def test_assembly_mesh_nodes_radius_0(): assert (np.round(titan.assembly[0].mesh.nodes_radius[0],5) ==  np.round(1.,5)).all(), " Mesh nodes radius are wrong"
def test_assembly_mesh_nodes_radius_1(): assert (np.round(titan.assembly[0].mesh.nodes_radius[1],5) ==  np.round(1.,5)).all(), " Mesh nodes radius are wrong"
def test_assembly_mesh_nodes_radius_2(): assert (np.round(titan.assembly[0].mesh.nodes_radius[2],5) ==  np.round(1.,5)).all(), " Mesh nodes radius are wrong"

def test_assembly_mesh_edges_0(): assert (titan.assembly[0].mesh.edges[0] == np.array([0, 1])).all(), " Mesh edges are wrong"
def test_assembly_mesh_edges_1(): assert (titan.assembly[0].mesh.edges[1] == np.array([0, 2])).all(), " Mesh edges are wrong"
def test_assembly_mesh_edges_2(): assert (titan.assembly[0].mesh.edges[2] == np.array([0, 3])).all(), " Mesh edges are wrong"
