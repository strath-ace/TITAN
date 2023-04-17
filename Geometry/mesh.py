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
import meshio
import numpy as np
import scipy.special as special


def read_mesh(filename):

    mesh = meshio.read(filename)
    facets = mesh.points[mesh.cells_dict['triangle']]

    v0 = facets[:,0]
    v1 = facets[:,1]
    v2 = facets[:,2]

    return v0,v1,v2

class Mesh():
    def __init__(self, filename = []):

        if filename == []:
            self.v0 = np.array([])
            self.v1 = np.array([])
            self.v2 = np.array([])
        else:
            self.v0, self.v1, self.v2 = read_mesh(filename)

        self.v0 = self.v0.astype(np.double)
        self.v1 = self.v1.astype(np.double)
        self.v2 = self.v2.astype(np.double)

        self.facet_normal = np.array([])
        self.facet_area = np.array([])
        
        self.min = np.zeros(3)
        self.max = np.zeros(3)

        self.facet_COG=np.array([])
        self.COG= np.array([])

        self.nodes = np.array([],dtype = np.double)
        self.nodes_normal = np.array([])

        self.edges= np.array([], dtype = int)

        self.facets = np.array([], dtype = int)
        self.facet_edges = np.zeros((len(self.v0), 3), dtype = int)
        self.nodes_radius = np.ones([]) 

        self.vol_coords = np.array([])


def append(mesh_assembly, mesh_obj):

    if  mesh_assembly.v0.size == 0:
        mesh_assembly.v0 = np.copy(mesh_obj.v0)
        mesh_assembly.v1 = np.copy(mesh_obj.v1)
        mesh_assembly.v2 = np.copy(mesh_obj.v2)
        mesh_assembly.facet_normal = np.copy(mesh_obj.facet_normal)
        mesh_assembly.facet_area = np.copy(mesh_obj.facet_area)
        mesh_assembly.facet_COG = np.copy(mesh_obj.facet_COG)

    else:
        mesh_assembly.v0 = np.append(mesh_assembly.v0, mesh_obj.v0, axis=0)
        mesh_assembly.v1 = np.append(mesh_assembly.v1, mesh_obj.v1, axis=0)
        mesh_assembly.v2 = np.append(mesh_assembly.v2, mesh_obj.v2, axis=0)
        mesh_assembly.facet_normal = np.append(mesh_assembly.facet_normal, mesh_obj.facet_normal, axis=0)
        mesh_assembly.facet_area = np.append(mesh_assembly.facet_area, mesh_obj.facet_area, axis=0)
        mesh_assembly.facet_COG = np.append(mesh_assembly.facet_COG, mesh_obj.facet_COG, axis=0)

    return mesh_assembly


def compute_mesh(mesh, compute_radius = True):
    mesh.facet_area = compute_facet_area(mesh.v0, mesh.v1, mesh.v2)
    mesh.facet_COG = compute_facet_COG(mesh.v0, mesh.v1, mesh.v2)
    mesh.COG = compute_geometrical_COG(mesh.facet_COG, mesh.facet_area)
    mesh.facet_normal = compute_facet_normal(mesh.COG, mesh.facet_COG, mesh.v0, mesh.v1, mesh.v2, mesh.facet_area)
    mesh.nodes, mesh.facets = map_facets_connectivity(mesh.v0, mesh.v1, mesh.v2)
    mesh.nodes_normal = compute_nodes_normals(len(mesh.nodes), mesh.facets ,mesh.facet_COG, mesh.v0,mesh.v1,mesh.v2)
    mesh.min, mesh.max = compute_min_max(mesh.nodes)
    mesh.edges, mesh.facet_edges = map_edges_connectivity(mesh.facets)
    
    if compute_radius: mesh.nodes_radius = compute_curvature(mesh.nodes, mesh.facets, mesh.nodes_normal, mesh.facet_normal, mesh.facet_area, mesh.v0,mesh.v1,mesh.v2)
    else: mesh.node_radius = np.ones((len(mesh.nodes)))

    mesh.surface_displacement = np.zeros((len(mesh.nodes),3))
    
    return mesh

def update_surface_displacement(mesh, surface_displacement_vector):
    mesh.nodes += surface_displacement_vector
    mesh.v0 = mesh.nodes[mesh.facets[:,0]]   
    mesh.v1 = mesh.nodes[mesh.facets[:,1]]    
    mesh.v2 = mesh.nodes[mesh.facets[:,2]] 

    mesh.facet_area = compute_facet_area(mesh.v0, mesh.v1, mesh.v2)
    mesh.facet_COG = compute_facet_COG(mesh.v0, mesh.v1, mesh.v2)
    mesh.COG = compute_geometrical_COG(mesh.facet_COG, mesh.facet_area)
    mesh.facet_normal = compute_facet_normal(mesh.COG, mesh.facet_COG, mesh.v0, mesh.v1, mesh.v2)
    mesh.nodes_normal = compute_nodes_normals(len(mesh.nodes), mesh.facets ,mesh.facet_COG, mesh.v0, mesh.v1, mesh.v2, mesh.facet_area)
    mesh.min, mesh.max = compute_min_max(mesh.nodes)


def update_volume_displacement(mesh, volume_displacement_vector):
    mesh.vol_coords += volume_displacement_vector

def compute_facet_area(v0,v1,v2):
    #Compute Area of the Mesh Facets

    v1_v0 = v1-v0
    v2_v0 = v2-v0
    v2_v1 = v2-v1

    a=np.sqrt(np.einsum('ij,ij->i',v1_v0,v1_v0))
    b=np.sqrt(np.einsum('ij,ij->i',v2_v0,v2_v0))
    c=np.sqrt(np.einsum('ij,ij->i',v2_v1,v2_v1))
    
    s = 0.5*(a+b+c)
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))

    return area

def compute_facet_COG(v0,v1,v2):
    #Compute the center of mass of each vertex

    facet_COG = (v0+v1+v2)/3.0
    return facet_COG

def compute_geometrical_COG(facet_COG, facet_area):
    #Compute Geometrical center of the Mesh

    COG = np.sum(facet_COG*facet_area[:,None], axis = 0)/np.sum(facet_area)
    return COG

def compute_facet_normal(COG, facet_COG, v0,v1,v2, area):
    #Compute Facet Normals

    facet_normal = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(facet_normal, axis = 1, ord=2)
    facet_normal /= norms[:,None]
    facet_normal *= area[:,None]

    return facet_normal

def map_facets_connectivity(v0,v1,v2):
    #Map the Facets with respect to the Mesh Nodes

    nodes = np.hstack((v0,v1,v2))
    nodes.shape=(3*len(v0),3)
    unique , idx, inv =np.unique(nodes, axis = 0, return_index = True, return_inverse = True)
    nodes = unique

    facets=np.copy(inv)
    facets.shape = (len(v0),3)
    
    return nodes, facets

def map_edges_connectivity(facets):
    #Map the Edges with respect to the Mesh Nodes
    #Additionally, maps the Facets with respect to the Edges

    facet_edges = np.zeros((len(facets)*3), dtype = int)
    edges = np.array([], dtype = int)

    edge1 = facets[:,0:2]
    edge2 = facets[:,1:3]
    edge3 = np.stack((facets[:,2],facets[:,0]), axis = -1)

    edges = np.hstack((edge1,edge2,edge3))
    edges.shape=(len(facets)*3,2)

    edges_original = np.copy(edges)
    edges_sorted = np.copy(np.sort(edges)) #Sort Edges from the lower node ID to higher node ID for each edge

    unique, idx, inv = np.unique(edges_sorted, axis = 0, return_index = True, return_inverse = True)

    edges= unique
    num_edges=len(edges)

    # Create Matrix of Elems_connects wrt vector of Edges (The +1 is for Ease the use in GMSH 
    facet_edges = np.copy(inv) + 1

    edges_inv = edges[inv]

    mask = (edges_inv[:,0] == edges_original[:,0]) * (edges_inv[:,1] == edges_original[:,1]) 
    facet_edges[~mask] *=-1
    facet_edges.shape=(len(facets),3)

    return edges, facet_edges

def compute_nodes_normals(num_nodes, facets ,facet_COG, v0,v1,v2):
    #Compute the normal at the Vertex

    nodes_normal = np.zeros((num_nodes,3))

    CG = facet_COG
    CE_01 = (v0+v1)/2.0
    CE_12 = (v1+v2)/2.0
    CE_20 = (v2+v0)/2.0

    cross_CE_01_v0 = np.cross(CE_01 - v0, CG - v0, axis = 1)
    cross_v0_CE_20 = np.cross(CG - v0, CE_20 - v0, axis = 1)
    cross_CE_12_v1 = np.cross(CE_12 - v1, CG - v1, axis = 1)
    cross_v1_CE_01 = np.cross(CG - v1, CE_01 - v1, axis = 1)
    cross_CE_20_v2 = np.cross(CE_20 - v2, CG - v2, axis = 1)
    cross_v2_CE_12 = np.cross(CG - v2, CE_12 - v2, axis = 1)
    
    np.add.at(nodes_normal, facets[:,0], 0.5* (cross_CE_01_v0 + cross_v0_CE_20))
    np.add.at(nodes_normal, facets[:,1], 0.5* (cross_CE_12_v1 + cross_v1_CE_01))
    np.add.at(nodes_normal, facets[:,2], 0.5* (cross_CE_20_v2 + cross_v2_CE_12))

    return nodes_normal

def compute_min_max(nodes):
    #Computes the minimum and maximum coordinates of the mesh

    _min = np.zeros((3))
    _max = np.zeros((3))
    
    _min=np.min(nodes, axis = 0)
    _max=np.max(nodes, axis = 0)

    return _min, _max


def compute_curvature(nodes, facets,nodes_normal, facet_normals, facets_area, v0, v1, v2):

    avType = 'e'
    Nsmooth = int(np.round(len(facets)/100, decimals = 0))
    M2M_RR = 10.0
    flatEdge = 0.9
    flatWeightFlag = 2

    free_vector = np.array([1,0,0])
    p1 = np.dot(facet_normals, free_vector)
    p1 = p1<0    

    Theta = np.pi/2 - np.arccos(np.clip(np.sum(- free_vector * facet_normals[p1] , axis = 1), -1.0, 1.0))
    
    area = np.sum(facets_area[p1]*np.sin(Theta))

    Rmax = np.sqrt(area/np.pi)*2
    Rmin = Rmax/M2M_RR 
    
    # Based on Fostrad module -> based on the work of Itzik Ben Shabat

    VertexNormals,Avertex,Acorner,up,vp, avEdge = calculate_vertex_normals(nodes, facets, facet_normals, facets_area, v0,v1,v2)
    VertexSFM=calculate_curvature(VertexNormals,Avertex,Acorner,up,vp,nodes,facets,facet_normals,v0,v1,v2)
    CurvUxVy,PrincipalDir1,PrincipalDir2=getPrincipalCurvatures(nodes,VertexSFM,up,vp)

    CurvUxVy = np.abs(CurvUxVy)

    radiiOnVerts = np.zeros((len(nodes)))

    SearchRadius=avEdge

    ParFlag = len(nodes)/50000.0

    with np.errstate(divide='ignore'):
        radiiOnVerts[:] = np.sqrt(1/(CurvUxVy[:,0]*CurvUxVy[:,1]))
 
    radiiOnVerts[(CurvUxVy[:,0] > 1/Rmax) * (CurvUxVy[:,1]< 1/Rmax)] = 1./CurvUxVy[(CurvUxVy[:,0] > 1/Rmax) * (CurvUxVy[:,1]< 1/Rmax),0]
    radiiOnVerts[(CurvUxVy[:,1] > 1/Rmax) * (CurvUxVy[:,0]< 1/Rmax)] = 1./CurvUxVy[(CurvUxVy[:,1] > 1/Rmax) * (CurvUxVy[:,0]< 1/Rmax),1]
    
    radiiOnVerts[radiiOnVerts > Rmax] = Rmax
    radiiOnVerts[radiiOnVerts < Rmin] = Rmin

    for i in range(len(nodes)):
        InnerPointsInd = searchableRadius(nodes[i],nodes,SearchRadius,Nsmooth)
        if len(InnerPointsInd) <= 1: continue
        radiiOnVerts[i]= sphVolSmoothing(InnerPointsInd, radiiOnVerts, avType, Nsmooth, Rmax, flatEdge, flatWeightFlag)

    node_radius = radiiOnVerts

    node_radius[node_radius <= 0] = Rmin

    return node_radius


def sphVolSmoothing(InnerPointsInd, propOnVerts, avType, Nsmooth, MaxRefRadius, flatEdge, flatWeightFlag):
    
    propOnVerts = propOnVerts[InnerPointsInd]
    MaxRefRadius*= 0.99

    Npoints = len(propOnVerts)
    if Npoints<Nsmooth+1: Nsmooth = Npoints-2

    propOnVerts=propOnVerts[-Nsmooth::]
    NpointsNonInf = np.sum(propOnVerts<MaxRefRadius)
    NpointsInf = Nsmooth+1-NpointsNonInf

    flatRatio = NpointsInf/Nsmooth
    propBaseline = np.mean(propOnVerts[propOnVerts<MaxRefRadius])

    if flatRatio >= flatEdge: flatFlag = 1
    else:
        flatFlag = 0
        if flatWeightFlag ==2:
            propOnVerts[propOnVerts>=MaxRefRadius] = propBaseline + (MaxRefRadius - propBaseline)*((1+special.erf(flatRatio*4-2))/2)

    if flatFlag == 1: smoothSphProp = MaxRefRadius
    else:
        if avType == 'e': 
            smoothSphProp = exponential_moving_average(propOnVerts[propOnVerts<MaxRefRadius], Nsmooth-NpointsInf , Nsmooth/32.0)#Nsmooth/2.0)
            smoothSphProp = smoothSphProp[-1]

    return smoothSphProp
    pass

def searchableRadius(center, nodes, SearchRadius, Nsmooth):

    dist = nodes - center
    dist = np.sqrt(dist[:,0]**2 + dist[:,1]**2 + dist[:,2]**2)

    SearchPointsInd = np.where((dist <=  SearchRadius))[0] # selected searchable indexes on original V reference
    SearchPointsXYZ = nodes[SearchPointsInd] # selected searchable points

    if len(SearchPointsXYZ) <= Nsmooth:
        SearchPointsInd = np.where((dist <= SearchRadius*10))[0] # selected searchable indexes on original V reference
        SearchPointsXYZ = nodes[SearchPointsInd] # selected searchable points

    Dxyz = center - SearchPointsXYZ
    SearchPointsDist = np.sqrt(Dxyz[:,0]**2 + Dxyz[:,1]**2 + Dxyz[:,2]**2)

    ind = np.argsort(SearchPointsDist )[::-1]
    SearchPointsDist = SearchPointsDist[ind]
    SearchPointsInd = SearchPointsInd[ind]

    SearchPosInd = SearchPointsInd[SearchPointsDist<=SearchRadius*10]
    Nind = np.min((Nsmooth+3, len(SearchPosInd)-1))
    SearchPosInd = SearchPosInd[-Nind::]

    return SearchPosInd

def calculate_vertex_normals(nodes, facets, facet_normals, facets_area, v0,v1,v2):
    VertexNormals = np.zeros((len(nodes),3))
    up = np.zeros((len(nodes),3))
    vp = np.zeros((len(nodes),3))

    Acorner = np.zeros((len(facets),3))
    Avertex = np.zeros((len(nodes)))

    e0=v2-v1
    e1=v0-v2
    e2=v1-v0

    e0_norm=np.linalg.norm(e0 , ord = 2, axis = 1)
    e1_norm=np.linalg.norm(e1 , ord = 2, axis = 1)
    e2_norm=np.linalg.norm(e2 , ord = 2, axis = 1)

    #edge lengths
    de0=np.sqrt(e0[:,0]**2+e0[:,1]**2+e0[:,2]**2)
    de1=np.sqrt(e1[:,0]**2+e1[:,1]**2+e1[:,2]**2)
    de2=np.sqrt(e2[:,0]**2+e2[:,1]**2+e2[:,2]**2)

    De0 = np.mean(de0[de0>1e-6])
    De1 = np.mean(de1[de1>1e-6])
    De2 = np.mean(de2[de2>1e-6])

    avEdge = np.mean([De0,De1,De2])

    # Change to numpy.dot (should be faster)
    l0_2 = np.linalg.norm(e0,axis=1)*np.linalg.norm(e0,axis=1)
    l1_2 = np.linalg.norm(e1,axis=1)*np.linalg.norm(e1,axis=1)
    l2_2 = np.linalg.norm(e2,axis=1)*np.linalg.norm(e2,axis=1)
    ew = np.stack( (l0_2*(l1_2+l2_2-l0_2), l1_2*(l0_2+l2_2-l1_2), l2_2*(l0_2+l1_2-l2_2)), axis = -1)

    wfv0 = facets_area/(l1_2*l2_2)
    wfv1 = facets_area/(l0_2*l2_2)
    wfv2 = facets_area/(l1_2*l0_2)

    #Has to be like this to not find simultaneous acess to the same memory
    np.add.at(VertexNormals,facets[:,0], (wfv0[:,None]*facet_normals))
    np.add.at(VertexNormals,facets[:,1], (wfv1[:,None]*facet_normals))
    np.add.at(VertexNormals,facets[:,2], (wfv2[:,None]*facet_normals))

    ew_0 = (ew[:,0]<=0)
    ew_1 = (ew[:,1]<=0)
    ew_2 = (ew[:,2]<=0)
    ew_t = np.ones(len(ew_0), dtype = bool)*(~ew_0)*(~ew_1)*(~ew_2)

    if ew_0.any():
        Acorner[ew_0, 1] = -0.25*l2_2[ew_0]*facets_area[ew_0]/(np.sum(e0[ew_0]*e2[ew_0],axis=1))
        Acorner[ew_0, 2] = -0.25*l1_2[ew_0]*facets_area[ew_0]/(np.sum(e0[ew_0]*e1[ew_0],axis=1))
        Acorner[ew_0, 0] = facets_area[ew_0] - Acorner[ew_0, 1] - Acorner[ew_0, 2]

    if ew_1.any():
        Acorner[ew_1, 0] = -0.25*l2_2[ew_1]*facets_area[ew_1]/(np.sum(e1[ew_1]*e2[ew_1],axis=1))
        Acorner[ew_1, 2] = -0.25*l0_2[ew_1]*facets_area[ew_1]/(np.sum(e1[ew_1]*e0[ew_1],axis=1))
        Acorner[ew_1, 1] = facets_area[ew_1] - Acorner[ew_1, 0] - Acorner[ew_1, 2]

    if ew_2.any():
        Acorner[ew_2, 0] = -0.25*l1_2[ew_2]*facets_area[ew_2]/(np.sum(e2[ew_2]*e1[ew_2],axis=1))
        Acorner[ew_2, 1] = -0.25*l0_2[ew_2]*facets_area[ew_2]/(np.sum(e2[ew_2]*e0[ew_2],axis=1))
        Acorner[ew_2, 2] = facets_area[ew_2] - Acorner[ew_2, 0] - Acorner[ew_2, 1]

    if ew_t.any():
        ewscale=0.5*facets_area[ew_t]/(ew[ew_t,0]+ew[ew_t,1]+ew[ew_t,2])
        Acorner[ew_t, 0] = ewscale*(ew[ew_t,1]+ew[ew_t,2])
        Acorner[ew_t, 1] = ewscale*(ew[ew_t,0]+ew[ew_t,2])
        Acorner[ew_t, 2] = ewscale*(ew[ew_t,0]+ew[ew_t,1])

    np.add.at(Avertex,facets, Acorner)
    up[facets] = np.stack((e2/e2_norm[:,None],e0/e0_norm[:,None],e1/e1_norm[:,None]), axis = 1)

    VertexNormals=VertexNormals/np.linalg.norm(VertexNormals, axis=1, ord=2)[:,None]
    up = np.cross(up, VertexNormals, axis = 1)/np.linalg.norm(np.cross(up,VertexNormals, axis = 1), axis=1, ord = 2)[:,None]
    vp = np.cross(VertexNormals, up, axis = 1)
    
    return VertexNormals,Avertex,Acorner,up,vp, avEdge

def calculate_curvature(VertexNormals,Avertex,Acorner,up,vp,nodes,facets,facet_normals,v0,v1,v2):
    
    e0=v2-v1
    e1=v0-v2
    e2=v1-v0

    e0_norm=e0/np.linalg.norm(e0 , ord = 2, axis = 1)[:,None]
    #e1_norm=e1/numpy.linalg.norm(e1 , ord = 2, axis = 1)[:,None]
    #e2_norm=e2/numpy.linalg.norm(e2 , ord = 2, axis = 1)[:,None]

    B = np.cross(facet_normals,e0_norm,axis=1)
    B/= np.linalg.norm(B, axis = 1)[:,None]
    
    A = np.zeros((len(facet_normals),6,3))
    b = np.zeros((len(facet_normals),6))

    A[:,0,0] = np.sum(e0*e0_norm, axis = 1);         A[:,0,1] = np.sum(e0*B, axis = 1) ;              A[:,0,2] = 0; 
    A[:,1,0] = 0;                                    A[:,1,1] = np.sum(e0*e0_norm, axis = 1);         A[:,1,2] = np.sum(e0*B, axis = 1) ; 
    A[:,2,0] = np.sum(e1*e0_norm, axis = 1);         A[:,2,1] = np.sum(e1*B, axis = 1) ;              A[:,2,2] = 0; 
    A[:,3,0] = 0;                                    A[:,3,1] = np.sum(e1*e0_norm, axis = 1) ;        A[:,3,2] = np.sum(e1*B, axis = 1) ; 
    A[:,4,0] = np.sum(e2*e0_norm, axis = 1);         A[:,4,1] = np.sum(e2*B, axis = 1) ;              A[:,4,2] = 0; 
    A[:,5,0] = 0;                                    A[:,5,1] = np.sum(e2*e0_norm, axis = 1) ;        A[:,5,2] = np.sum(e2*B, axis = 1) ; 

    n0 = VertexNormals[facets[:,0]]
    n1 = VertexNormals[facets[:,1]]
    n2 = VertexNormals[facets[:,2]]

    b[:,0] = np.sum((n2-n1)*e0_norm , axis = 1)
    b[:,1] = np.sum((n2-n1)*B  , axis = 1)
    b[:,2] = np.sum((n0-n2)*e0_norm , axis = 1)
    b[:,3] = np.sum((n0-n2)*B  , axis = 1)
    b[:,4] = np.sum((n1-n0)*e0_norm , axis = 1)
    b[:,5] = np.sum((n1-n0)*B  , axis = 1)

    x=np.zeros((int(len(b)),3))

    #BOTTLENECK
    for i in range(len(b)):
        x[i]=np.linalg.lstsq(A[i],b[i], rcond=None)[0]

    wfp = Acorner[:,:]/Avertex[facets[:]]

    VertexSFM = np.zeros((len(nodes),4))

    new_ku,new_kuv,new_kv = ProjectCurvatureTensor(e0_norm,B,facet_normals,x[:,0],x[:,1],x[:,2],up[facets],vp[facets])

    horizontal_stack_0 = np.stack((new_ku[:,0],new_kuv[:,0],new_kuv[:,0],new_kv[:,0]), axis = -1)
    horizontal_stack_1 = np.stack((new_ku[:,1],new_kuv[:,1],new_kuv[:,1],new_kv[:,1]), axis = -1)
    horizontal_stack_2 = np.stack((new_ku[:,2],new_kuv[:,2],new_kuv[:,2],new_kv[:,2]), axis = -1)
    
    np.add.at(VertexSFM, facets[:,0], wfp[:,0][:,None]*horizontal_stack_0)
    np.add.at(VertexSFM, facets[:,1], wfp[:,1][:,None]*horizontal_stack_1)
    np.add.at(VertexSFM, facets[:,2], wfp[:,2][:,None]*horizontal_stack_2)
 
    VertexSFM.shape = (-1,2,2)
    
    return VertexSFM


def RotateCoordinateSystem_nd(up,vp,nf, axis = 2):    
    r_new_u=up
    r_new_v=vp

    if axis == 2:
        _np=np.cross(up,vp, axis = -1)
        _np=_np/np.linalg.norm(_np, axis = -1)[:,:,None]

        arrays_temp = nf
        nf = np.repeat(nf[:,:,np.newaxis], repeats = 3, axis = -1)
        nf[:] = arrays_temp[:,None]

        ndot=np.sum(nf*_np, axis = -1)

        p = ndot<=-1
        r_new_u[p]=-r_new_u[p]
        r_new_v[p]=-r_new_v[p]  

        p = ~p

        arrays_temp = ndot
        ndot = np.repeat(ndot[:,:,np.newaxis], repeats = 3, axis = -1)
        perp = nf-ndot*_np
        dperp=(_np+nf)/(1+ndot)

        aux_1 = np.sum(perp*r_new_u, axis = -1)
        aux_1 = np.repeat(aux_1[:,:,np.newaxis], repeats = 3, axis = -1)

        aux_2 = np.sum(perp*r_new_v, axis = -1)
        aux_2 = np.repeat(aux_2[:,:,np.newaxis], repeats = 3, axis = -1)

        r_new_u[p]=(r_new_u-dperp*aux_1)[p]
        r_new_v[p]=(r_new_v-dperp*aux_2)[p]

    if axis == 1:

        _np=np.cross(up,vp, axis = -1)
        _np=_np/np.linalg.norm(_np, axis = -1)[:,None]

        arrays_temp = nf
        nf[:] = arrays_temp[None]
        ndot=np.sum(nf*_np, axis = -1)

        p = ndot<=-1

        r_new_u[p]=-r_new_u[p]
        r_new_v[p]=-r_new_v[p]  

        p = ~p

        arrays_temp = ndot
        perp = nf-ndot[:,None]*_np
        dperp=(_np+nf)/(1+ndot[:,None])

        aux_1 = np.sum(perp*r_new_u, axis = 1)
        aux_2 = np.sum(perp*r_new_v, axis = -1)

        r_new_u[p]=(r_new_u-dperp*aux_1[:,None])[p]
        r_new_v[p]=(r_new_v-dperp*aux_2[:,None])[p]

    return r_new_u,r_new_v

def RotateCoordinateSystem_1d(up,vp,nf):
    r_new_u=up;
    r_new_v=vp;
    _np=np.cross(up,vp);
    _np=_np/np.linalg.norm(_np);
    ndot=np.sum(nf*_np);

    if ndot<=-1:
        r_new_u=-r_new_u;
        r_new_v=-r_new_v;  
        return r_new_u,r_new_v
    perp=nf-ndot*_np;
    dperp=(_np+nf)/(1+ndot);

    r_new_u=r_new_u-dperp*(np.sum(perp*r_new_u));
    r_new_v=r_new_v-dperp*(np.sum(perp*r_new_v));
    return r_new_u,r_new_v

def ProjectCurvatureTensor(uf,vf,nf,old_ku,old_kuv,old_kv,up,vp):
    r_new_u,r_new_v=RotateCoordinateSystem_nd(up,vp,nf)
    OldTensor=np.transpose(np.array([[old_ku, old_kuv],[old_kuv, old_kv]]),(2,1,0))

    arrays_temp = uf
    uf = np.repeat(uf[:,:,np.newaxis], repeats = 3, axis = -1)
    uf[:] = arrays_temp[:,None]

    arrays_temp = vf
    vf = np.repeat(vf[:,:,np.newaxis], repeats = 3, axis = -1)
    vf[:] = arrays_temp[:,None]

    u1=np.sum(r_new_u*uf, axis = -1)
    v1=np.sum(r_new_u*vf, axis = -1)
    u2=np.sum(r_new_v*uf, axis = -1)
    v2=np.sum(r_new_v*vf, axis = -1)

    A1= np.transpose(np.array([u1,v1]),(1,2,0))
    A2= np.transpose(np.array([u2,v2]),(1,2,0))

    diagonal = np.array([[True,False,False],[False,True,False],[False,False,True]])

    new_ku  = np.matmul(np.matmul(A1,OldTensor),A1.transpose((0, 2, 1)))[:, diagonal]
    new_kuv = np.matmul(np.matmul(A1,OldTensor),A2.transpose((0, 2, 1)))[:, diagonal]
    new_kv  = np.matmul(np.matmul(A2,OldTensor),A2.transpose((0, 2, 1)))[:, diagonal]

    return new_ku, new_kuv, new_kv

def getPrincipalCurvatures(nodes,VertexSFM,up,vp):
    PrincipalCurvatures=np.empty((len(nodes),2))
    PrincipalDir1=np.empty((len(nodes),3))
    PrincipalDir2=np.empty((len(nodes),3))

    c = np.ones((len(up)))
    s = np.zeros((len(up)))
    tt = np.zeros((len(up)))
    nf = np.cross(up,vp)
    r_old_u,r_old_v=RotateCoordinateSystem_nd(up,vp,nf, axis = 1)

    ku = VertexSFM[:,0,0]
    kuv = VertexSFM[:,0,1]
    kv = VertexSFM[:,1,1]

    mask = (kuv != 0)
    h = np.zeros((len(kuv)))
    h[mask] = 0.5 * (kv[mask]-ku[mask])/kuv[mask]

    mask2 = (h < 0)

    tt[mask*mask2] = 1/(h-np.sqrt(1+h*h))[mask*mask2]
    tt[mask*~mask2] = 1/(h+np.sqrt(1+h*h))[mask*~mask2]

    c[mask] = (1/np.sqrt(1+tt*tt))[mask]
    s[mask] = (tt*c)[mask]

    k1 = ku - tt * kuv
    k2 = kv + tt * kuv

    mask = (np.abs(k1) >= np.abs(k2))
    PrincipalDir1[mask]= (c[:,None]*r_old_u - s[:,None]*r_old_v)[mask]
    PrincipalDir1[~mask]= (c[:,None]*r_old_u + s[:,None]*r_old_v)[~mask]
    PrincipalDir2 = np.cross(nf, PrincipalDir1)

    PrincipalCurvatures[:,0][mask] = k1[mask]
    PrincipalCurvatures[:,1][mask] = k2[mask]

    PrincipalCurvatures[:,0][~mask] = k2[~mask]
    PrincipalCurvatures[:,1][~mask] = k1[~mask]

    return PrincipalCurvatures,PrincipalDir1,PrincipalDir2

def exponential_moving_average(signal, points, smoothing=2):
    """
    Calculate the N-point exponential moving average of a signal

    Inputs:
        signal: numpy array -   A sequence of price points in time
        points:      int    -   The size of the moving average
        smoothing: float    -   The smoothing factor

    Outputs:
        ma:     numpy array -   The moving average at each point in the signal
    """

    weight = smoothing / (points + 1)
    ema = np.zeros(len(signal))
    ema[0] = signal[0]

    for i in range(1, len(signal)):
        ema[i] = (signal[i] * weight) + (ema[i - 1] * (1 - weight))

    return ema

def remove_repeated_facets(assembly_mesh):

    facet_COG = np.around(assembly_mesh.facet_COG, decimals = 6)

    ___, idx, count = np.unique(facet_COG, axis = 0, return_index = True, return_counts = True)

    p = idx.argsort()
    idx=idx[p]
    count=count[p]

    i = 0
    
    assembly_mesh.facet_COG = assembly_mesh.facet_COG[idx]
    assembly_mesh.v0 =       assembly_mesh.v0[idx]
    assembly_mesh.v1 =       assembly_mesh.v1[idx]
    assembly_mesh.v2 =       assembly_mesh.v2[idx]
    assembly_mesh.facet_normal =  assembly_mesh.facet_normal[idx]
    assembly_mesh.facet_area =     assembly_mesh.facet_area[idx]

    
    #idx is the CFD Grid

    while i < len(idx):
        if count[i] > 1:
            idx = np.delete(idx,i)
            count = np.delete(count,i)
        else: i+=1

    return idx
    

def create_index(body_atr, obj_atr):

    cond_obj = np.char.add(np.char.add(obj_atr[:,0].astype(str),obj_atr[:,1].astype(str)),obj_atr[:,2].astype(str) )
    cond_body = np.char.add(np.char.add(body_atr[:,0].astype(str),body_atr[:,1].astype(str)), body_atr[:,2].astype(str))

    mask = np.in1d(cond_body,cond_obj)
    index = np.where(mask)[0]

    return index, mask

def compute_new_volume(assembly, old_nodes):

    assembly.mesh.vol_elements = np.array([], dtype = int)
    assembly.mesh.vol_density = np.array([])
    assembly.mesh.vol_tag = np.array([], dtype = int)

    for obj in assembly.objects:
        size = len(obj.mesh.vol_elements)
        assembly.mesh.vol_elements = np.append(assembly.mesh.vol_elements,obj.mesh.vol_elements)
        assembly.mesh.vol_density = np.append(assembly.mesh.vol_density, np.ones(size)*obj.material.density)
        assembly.mesh.vol_tag = np.append(assembly.mesh.vol_tag, np.ones(size)*obj.id)

    assembly.mesh.vol_elements.shape = (-1,4)
    old_elements = np.copy(assembly.mesh.vol_elements)

    #Creation of the new nodes list:
    nodes = old_nodes[assembly.mesh.vol_elements]
    nodes.shape = (-1,3)
    
    unique, index, idx, count = np.unique(nodes, axis = 0, return_index= True, return_inverse = True, return_counts = True)
    idx.shape = (-1,4)

    assembly.mesh.vol_coords = unique
    assembly.mesh.vol_elements = idx

    #Order elements according to surface mesh
    surf_nodes = np.copy(assembly.mesh.nodes)

    condition = np.append(unique, surf_nodes,axis = 0)
    unique, idx, count = np.unique(condition, axis = 0, return_inverse = True, return_counts = True)
    unique2 = np.copy(unique)

    #Switch the nodes:
    #Change the elements connectivity #TODO better way
    index_surf_1 = idx[len(unique):]
    index_surf_2 = range(len(surf_nodes))    
    
    index_old = np.arange(len(unique))
    for i in range(len(surf_nodes)):

        temp = np.copy(unique[index_surf_2[i]])
        unique[index_surf_2[i]] = unique[index_surf_1[i]] # surf_nodes
        unique[index_surf_1[i]] = temp
                
        temp = np.copy(index_old[index_surf_2[i]])
        index_old[index_surf_2[i]] = index_old[index_surf_1[i]]
        index_old[index_surf_1[i]] = temp
        
        temp1 = (assembly.mesh.vol_elements == index_surf_1[i])
        temp2 = (assembly.mesh.vol_elements == index_surf_2[i])
    
        assembly.mesh.vol_elements[temp1] = index_surf_2[i] 
        assembly.mesh.vol_elements[temp2] = index_surf_1[i]
    
    assembly.mesh.vol_coords = unique

    start = 0
    for obj in assembly.objects:
        obj.mesh.vol_elements = np.copy(assembly.mesh.vol_elements[start:start+len(obj.mesh.vol_elements)])
        start += len(obj.mesh.vol_elements)

    return index,old_elements,index_old