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
import gmsh
import numpy as np

def mesh_Settings(gmsh):
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10) # for parallel 3D meshing
    #gmsh.option.setNumber("General.NumThreads", 1);
    #self.gmsh.option.setNumber("Mesh.Algorithm", 8);
    #self.gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1);
    #self.gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2);
    #self.gmsh.option.setNumber("Mesh.Optimize",1)
    #self.gmsh.option.setNumber("Mesh.QualityType",2);

def generate_inner_domain(mesh, assembly = [], write = False, output_folder = '', output_filename = '', bc_ids = []):
    gmsh.initialize()
    mesh_Settings(gmsh)

    ref_objects = 0.15
    ref_joint = 0.5
    ref_panel = 0.05
    density_elem = []
    tag_elem = []

    init_ref_surf = 1
    surf_ref_init = 1
    ref_phys_surface = 1

    #map_objects = dict()
    gmsh.model.mesh.createGeometry()

    #Change refs to refine joints as well
    ref = np.ones(len(mesh.nodes))*ref_objects

    for obj in assembly.objects:
        if obj.type.lower() == 'joint':
            ref[obj.node_index] = ref_joint
        if "panel" in obj.name:
            ref[obj.node_index] = ref_panel

    node_ref_init, edge_ref_init, surf_ref_init = object_grid(gmsh, mesh.nodes, mesh.edges, mesh.facet_edges, ref)
    init_ref_surf, ref_phys_surface = object_physical(gmsh, init_ref_surf, surf_ref_init, ref_phys_surface, 'top')

    if assembly:
        for i in range(len(assembly.objects)):
            out = gmsh.model.geo.addSurfaceLoop(np.array(assembly.objects[i].facet_index)+1)

            if assembly.objects[i].inner_mesh:
                node_ref_end , edge_ref_end, surf_ref_end = object_grid(gmsh,assembly.objects[i].inner_mesh.nodes, assembly.objects[i].inner_mesh.edges, assembly.objects[i].inner_mesh.facet_edges, ref,node_ref_init, edge_ref_init, surf_ref_init)
                #assembly.objects[i].inner_node_index = np.array(range(node_ref_init-1, node_ref_end-1))
                hole = gmsh.model.geo.addSurfaceLoop(range(surf_ref_init, surf_ref_end))
                vol_tag = gmsh.model.geo.addVolume([out,hole])
                
                node_ref_init = node_ref_end
                edge_ref_init = edge_ref_end
                surf_ref_init = surf_ref_end

            else:
                vol_tag = gmsh.model.geo.addVolume([out])
            
            assembly.objects[i].vol_tag = vol_tag 
            ref_phys_volume = gmsh.model.geo.addPhysicalGroup(3, [vol_tag])
            gmsh.model.setPhysicalName(3, ref_phys_volume, "body")  

            # map_objects.update({assembly.objects[i].name  : {}})
            # assembly.objects[i].vol_tag = vol_tag
            # map_objects[assembly.objects[i].name].update({'Vol id': i+1})
            # map_objects[assembly.objects[i].name].update({'Active': True})
            # if len(bc_ids) > 0:
            #     map_objects[assembly.objects[i].name].update({'BC id': bc_ids[i]})
                
            #map_objects.update({assembly.objects[i].id  : {}})
            #assembly.objects[i].vol_tag = vol_tag
            #map_objects[assembly.objects[i].id].update({'Vol id': i+1})
            #map_objects[assembly.objects[i].id].update({'Active': True})
            #map_objects[assembly.objects[i].id].update({'Name': assembly.objects[i].name})

            #if len(bc_ids) > 0:
            #    map_objects[assembly.objects[i].id].update({'BC id': bc_ids[i]})


            # map_objects.update({assembly.objects[i].name  : i+1})

    else:
        out = gmsh.model.geo.addSurfaceLoop(range(1,len(mesh.facets)+1))
        vol_tag = gmsh.model.geo.addVolume([out])
    
    #assembly.map_physical_volume = map_objects

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    if False:
        gmsh.fltk.initialize()
        while gmsh.fltk.isAvailable() and checkForEvent():
            gmsh.fltk.wait()
   
    entities = gmsh.model.getEntities()       

    if assembly:
        elements = np.array([]).astype(int)
        coords = np.array([])
        density_elem = np.array([])
        tag_elem = np.array([]).astype(int)

        # Get the mesh nodes for the entity (dim, tag):
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()
        coords = np.array(nodeCoords)

        for i in range(len(assembly.objects)):

            # Get the mesh elements for the entity (dim, tag):
            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(3, assembly.objects[i].vol_tag)

            elements = np.append(elements, elemNodeTags[0]-1)

            density_elem = np.append(density_elem, assembly.objects[i].material.density*np.ones((len(elemTags[0]))))
            tag_elem = np.append(tag_elem, np.array([assembly.objects[i].id]*len(elemTags[0])))
            
    else:
        # Get the mesh nodes for the entity (dim, tag):
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

        # Get the mesh elements for the entity (dim, tag):
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(3, vol_tag)

        elements = np.array(elemNodeTags[0]-1)
        coords = np.array(nodeCoords)
    
    elements.shape = (-1,4)
    coords.shape= (-1 ,3)

   # if write: 
    gmsh.write(output_folder +'/Volume/'+'%s_%s.vtk'%(output_filename, assembly.id))

   #     gmsh.model.mesh.generate(2)
   #     gmsh.write(output_folder +'/Volume/'+ '%s_%s_surf.vtk'%(output_filename, assembly.id))

    gmsh.finalize()

    return coords, elements.astype(int), density_elem, tag_elem.astype(int)

def generate_PATO_domain(obj, output_folder = ''):
    
    print('Generating PATO domain, assembly:', obj.parent_id, ' object:', obj.id)
    print('     ', obj.name)

    gmsh.initialize()
    mesh_Settings(gmsh)

    ref_objects = 0.1
    ref_joint = 1.0
    density_elem = []
    tag_elem = []

    init_ref_surf = 1
    surf_ref_init = 1
    ref_phys_surface = 1

    #map_objects = dict()
    gmsh.model.mesh.createGeometry()

    #Change refs to refine joints as well

    mesh = obj.mesh

    ref = np.ones(len(mesh.nodes))*ref_objects

    node_ref_init, edge_ref_init, surf_ref_init = object_grid(gmsh, mesh.nodes, mesh.edges, mesh.facet_edges, ref)
    init_ref_surf, ref_phys_surface = object_physical(gmsh, init_ref_surf, surf_ref_init, ref_phys_surface, 'top')

    obj_facet_index = np.arange(len(obj.facet_index))

    out = gmsh.model.geo.addSurfaceLoop(np.array(obj_facet_index)+1)
    #out = gmsh.model.geo.addSurfaceLoop(np.array(obj.facet_index)+1)

    if obj.inner_mesh:
        ref = np.ones(len(obj.inner_mesh.nodes))*ref_objects
        node_ref_end , edge_ref_end, surf_ref_end = object_grid(gmsh,obj.inner_mesh.nodes, obj.inner_mesh.edges, obj.inner_mesh.facet_edges, ref,node_ref_init, edge_ref_init, surf_ref_init)
        #assembly.objects[i].inner_node_index = np.array(range(node_ref_init-1, node_ref_end-1))
        hole = gmsh.model.geo.addSurfaceLoop(range(surf_ref_init, surf_ref_end))
        vol_tag = gmsh.model.geo.addVolume([out,hole])
        
        node_ref_init = node_ref_end
        edge_ref_init = edge_ref_end
        surf_ref_init = surf_ref_end

    else:
        vol_tag = gmsh.model.geo.addVolume([out])
    
    obj.vol_tag = vol_tag 
    ref_phys_volume = gmsh.model.geo.addPhysicalGroup(3, [vol_tag])
    gmsh.model.setPhysicalName(3, ref_phys_volume, "body")  

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write(output_folder +'/PATO_'+str(obj.global_ID)+'/mesh/'+'%s.su2'%('mesh'))
    gmsh.finalize()

def object_physical(gmsh, init_ref_surf, end_ref_surf, ref_phys_surface, name):
    #Change here for every object in assembly give a different tag

    gmsh.model.geo.addPhysicalGroup(2, range(init_ref_surf,end_ref_surf), ref_phys_surface)
    gmsh.model.setPhysicalName(2, ref_phys_surface, name)
    ref_phys_surface +=1

    return end_ref_surf, ref_phys_surface  

def object_grid(gmsh, nodes, edges, facet_edges, ref, node_ref = 1, edge_ref = 1, surf_ref = 1):

    node_dev = node_ref
    edge_dev = edge_ref
    surf_dev = surf_ref

    for i in range(len(nodes)):
        gmsh.model.geo.addPoint(nodes[i,0], nodes[i,1], nodes[i,2], ref[i], node_ref)
        node_ref +=1

    for i in range(len(edges)):
        gmsh.model.geo.addLine(edges[i,0]+node_dev, edges[i,1]+node_dev, edge_ref)
        edge_ref +=1

    for i in range(len(facet_edges)):
        gmsh.model.geo.addCurveLoop([np.sign(facet_edges[i,0])*(abs(facet_edges[i,0])+ edge_dev -1),
                                     np.sign(facet_edges[i,1])*(abs(facet_edges[i,1])+ edge_dev -1),
                                     np.sign(facet_edges[i,2])*(abs(facet_edges[i,2])+ edge_dev -1)], surf_ref)

        gmsh.model.geo.addPlaneSurface([surf_ref])
        #surface = gmsh.model.geo.addPlaneSurface([surf_ref])
        #print('surface:', surface)
        #gmsh.model.geo.addPhysicalGroup(2, [surface], tag = surf_ref, name = "top")
        #gmsh.model.setPhysicalName(2, surf_ref, "top")
        surf_ref +=1

    return node_ref, edge_ref, surf_ref

def generate_cfd_domain(assembly, dim, ref_size_surf = 1.0, ref_size_far = 1.0, output_folder = '', output_grid = 'Grid.su2', options = None):

    print("Generating CFD Mesh")

    gmsh.initialize()
    mesh_Settings(gmsh)

    ref = ref_size_surf
    ref2 = ref_size_far

    ref_phys_surface = 1
    node_ref = 1; edge_ref = 1; surf_ref = 1
    
    xmin = assembly[0].cfd_mesh.xmin
    xmax = assembly[0].cfd_mesh.xmax

    init_ref_surf = 1
    ref_phys_surface = 1
    gmsh.model.mesh.createGeometry()

    for it in range(len(assembly)):

        assembly[it].cfd_mesh.xmin = np.min(assembly[it].cfd_mesh.nodes, axis = 0)
        assembly[it].cfd_mesh.xmax = np.max(assembly[it].cfd_mesh.nodes, axis = 0)

        if xmin[0] > assembly[it].cfd_mesh.xmin[0]: xmin[0] = assembly[it].cfd_mesh.xmin[0] 
        if xmin[1] > assembly[it].cfd_mesh.xmin[1]: xmin[1] = assembly[it].cfd_mesh.xmin[1]
        if xmin[2] > assembly[it].cfd_mesh.xmin[2]: xmin[2] = assembly[it].cfd_mesh.xmin[2]

        if xmax[0] < assembly[it].cfd_mesh.xmax[0]: xmax[0] = assembly[it].cfd_mesh.xmax[0]
        if xmax[1] < assembly[it].cfd_mesh.xmax[1]: xmax[1] = assembly[it].cfd_mesh.xmax[1]
        if xmax[2] < assembly[it].cfd_mesh.xmax[2]: xmax[2] = assembly[it].cfd_mesh.xmax[2]

        node_ref, edge_ref, surf_ref = object_grid(gmsh,assembly[it].cfd_mesh.nodes,assembly[it].cfd_mesh.edges,assembly[it].cfd_mesh.facet_edges,np.ones((len(assembly[it].cfd_mesh.nodes)))*ref, node_ref, edge_ref, surf_ref)
        init_ref_surf, ref_phys_surface = object_physical(gmsh, init_ref_surf, surf_ref, ref_phys_surface, "Body_"+str(ref_phys_surface))
    
    outer_surface(gmsh,ref2, surf_ref-1, xmin, xmax, ref_phys_surface, options = None)
    
    gmsh.model.geo.synchronize()

    if False:
        gmsh.fltk.initialize()
        while gmsh.fltk.isAvailable() and checkForEvent():
            gmsh.fltk.wait()

    gmsh.model.mesh.generate(dim)

    #gmsh.model.mesh.generate(2)
    #gmsh.write(output_folder+'/CFD_Grid/'+'a.vtk')
    gmsh.write(output_folder+'/CFD_Grid/'+output_grid)
    gmsh.finalize()

def outer_surface(gmsh,ref,surf_ref, xmin,xmax, ref_phys_surface, options = None):
    
    front = 1.5
    back = -1
    side = 1.5

    gmsh.model.geo.addPoint(front*abs((xmax[0]-xmin[0]))+xmax[0], 0.5*(xmax[1]+xmin[1]), 0.5*(xmax[2]+xmin[2]), ref)
    gmsh.model.geo.addPoint(back*abs((xmax[0]-xmin[0]))+xmin[0], 0.5*(xmax[1]+xmin[1]), 0.5*(xmax[2]+xmin[2]), ref)
    if(xmax[2]-xmin[2] > xmax[1]-xmin[1]):
        iNode = gmsh.model.geo.addPoint(back*abs((xmax[0]-xmin[0]))+xmin[0], 0.5*(xmax[1]+xmin[1]), side*abs(xmax[2]-xmin[2])+xmax[2], ref)
    else: 
        iNode = gmsh.model.geo.addPoint(back*abs((xmax[0]-xmin[0]))+xmin[0], side*abs(xmax[1]-xmin[1])+xmax[1], 0.5*(xmax[2]+xmin[2]),  ref)

    ellipse = gmsh.model.geo.addEllipseArc(iNode-2 ,iNode-1, iNode-2 ,iNode)

    surf1 = gmsh.model.geo.revolve([(1,ellipse  )],0, 0.5*(xmax[1]+xmin[1]), 0.5*(xmax[2]+xmin[2]),1,0,0,np.pi/2)
    surf2 = gmsh.model.geo.revolve([(1,ellipse+1)],0, 0.5*(xmax[1]+xmin[1]), 0.5*(xmax[2]+xmin[2]),1,0,0,np.pi/2)
    surf3 = gmsh.model.geo.revolve([(1,ellipse+4)],0, 0.5*(xmax[1]+xmin[1]), 0.5*(xmax[2]+xmin[2]),1,0,0,np.pi/2)
    surf4 = gmsh.model.geo.revolve([(1,ellipse+7)],0, 0.5*(xmax[1]+xmin[1]), 0.5*(xmax[2]+xmin[2]),1,0,0,np.pi/2)
    
    base = gmsh.model.geo.addCurveLoop([surf1[2][1], surf2[2][1], surf3[2][1], surf4[2][1]])

    base = gmsh.model.geo.addPlaneSurface([base])
    hole = gmsh.model.geo.addSurfaceLoop(range(1,surf_ref+1))

    out = gmsh.model.geo.addSurfaceLoop([surf1[1][1], surf2[1][1], surf3[1][1], surf4[1][1], base])

    v = gmsh.model.geo.addVolume([out,hole])

    gmsh.model.geo.addPhysicalGroup(2, [surf1[1][1], surf2[1][1], surf3[1][1], surf4[1][1]], ref_phys_surface)
    gmsh.model.setPhysicalName(2, ref_phys_surface, "Farfield")
    ref_phys_surface+=1

    gmsh.model.geo.addPhysicalGroup(2, [base], ref_phys_surface)
    gmsh.model.setPhysicalName(2, ref_phys_surface, "Outlet")
    ref_phys_surface+=1

    gmsh.model.geo.addPhysicalGroup(3, [v])
