import sys
import re
import numpy as np

def lookup(string):

    line_num = 0
    for line in Lines:
        line_num += 1
        if line.find(string) >= 0:
            return line_num

def extract_numbers_from_line(lst):
    res = []
    x=lst.split()
    for i in x:
        s = i.replace('.','',1)
        if s.isnumeric() or s.__contains__('+') or s.__contains__('-'):
            res.append(float(i))

    return res

#name output file
mshfile = "mesh.msh"

# write msh section that is independent of su2 file
phys_vol_tag  = 34
phys_surf_tag = 33
with open(mshfile, 'w') as f:
    f.write('$MeshFormat\n')
    f.write('2.2 0 8\n')
    f.write('$EndMeshFormat\n')
    f.write('$PhysicalNames\n')
    f.write('2\n')
    f.write('2 ' + str(phys_surf_tag) + ' "top"\n')
    f.write('3 ' + str(phys_vol_tag) + ' "body"\n')
    f.write('$EndPhysicalNames\n')
    f.write('$Nodes\n')
f.close()

# read su2 file
su2file = open(str(sys.argv[1]), 'r')
# extract lines
Lines = su2file.readlines()

# extract line with point number definition and data: x, y, z, node_id
line_number = lookup("NPOIN")
n_points = int(extract_numbers_from_line(Lines[line_number-1])[0])

# write number of points
with open(mshfile, 'a') as f:
    f.write(str(n_points) + '\n')
f.close()


# extract and write point data: x, y, z, node_id
x       = np.zeros(n_points)
y       = np.zeros(n_points)
z       = np.zeros(n_points)
node_id = np.zeros(n_points)
node_id = node_id.astype(int)

for i in range(n_points):
    line = Lines[line_number+i]
    values = extract_numbers_from_line(line)
    x[i]       = values[0]
    y[i]       = values[1]
    z[i]       = values[2]
    node_id[i] = int(round(values[3]))
    with open(mshfile, 'a') as f:
        f.write(str(node_id[i]+1) + ' ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + '\n')
    f.close()

with open(mshfile, 'a') as f:
    f.write('$EndNodes\n')
    f.write('$Elements\n')
f.close()

# extract lines with element number definition
line_number_n_elem = lookup("NELEM")
n_elements = int(extract_numbers_from_line(Lines[line_number_n_elem-1])[0])

# assuming 1 marker "top"
line_number_surf = lookup("MARKER_ELEMS")
n_elements_marker = int(extract_numbers_from_line(Lines[line_number_surf-1])[0])

# write element number definition
with open(mshfile, 'a') as f:
    f.write(str(n_elements+n_elements_marker) + '\n')
f.close()

# extract and write element data: type, node connectivity, element_id


# surface (only triangles)

elem_id = 1
for i in range(n_elements_marker):
    line = Lines[line_number_surf+i]
    values = extract_numbers_from_line(line)
    phys_tag = phys_surf_tag
    connectivity = [x+1 for x in values[1:]]
    connectivity = [int(x) for x in connectivity]
    elem_type = 2
    entity = 3
    with open(mshfile, 'a') as f:
        f.write(str(elem_id) + ' ' + str(elem_type) + ' ' + str(2) + ' ' + str(phys_tag) + ' ' + str(entity))
        for x in range(len(connectivity)):
            f.write(' ' + str(connectivity[x]))
        f.write('\n')
    f.close()
    elem_id+=1


# volume tetras

for i in range(n_elements):
    line = Lines[line_number_n_elem+i]
    values = extract_numbers_from_line(line)
    if values[0] == 10: #tetra
        phys_tag = phys_vol_tag
        connectivity = [x+1 for x in values[1:-1]]
        connectivity = [int(x) for x in connectivity]
        elem_type = 4
        entity = 7
        with open(mshfile, 'a') as f:
            f.write(str(elem_id) + ' ' + str(elem_type) + ' ' + str(2) + ' ' + str(phys_tag) + ' ' + str(entity))
            for x in range(len(connectivity)):
                f.write(' ' + str(connectivity[x]))
            #f.write(' ' + str(connectivity[2]))
            #f.write(' ' + str(connectivity[1]))
            #f.write(' ' + str(connectivity[0]))
            f.write('\n')
        f.close()
        elem_id+=1

# volume pyramids

for i in range(n_elements):
    line = Lines[line_number_n_elem+i]
    values = extract_numbers_from_line(line)
    if values[0] == 14: #pyramid
        phys_tag = phys_vol_tag
        connectivity = [x+1 for x in values[1:-1]]
        connectivity = [int(x) for x in connectivity]
        elem_type = 7
        entity = 7
        with open(mshfile, 'a') as f:
            f.write(str(elem_id) + ' ' + str(elem_type) + ' ' + str(2) + ' ' + str(phys_tag) + ' ' + str(entity))
            for x in range(len(connectivity)):
                f.write(' ' + str(connectivity[x]))
            #f.write(' ' + str(connectivity[2]))
            #f.write(' ' + str(connectivity[1]))
            #f.write(' ' + str(connectivity[0]))
            f.write('\n')
        f.close()
        elem_id+=1

# volume prisms

for i in range(n_elements):
    line = Lines[line_number_n_elem+i]
    values = extract_numbers_from_line(line)
    if values[0] == 13: #prism
        phys_tag = phys_vol_tag
        connectivity = [x+1 for x in values[1:-1]]
        connectivity = [int(x) for x in connectivity]
        elem_type = 6
        entity = 1
        with open(mshfile, 'a') as f:
            f.write(str(elem_id) + ' ' + str(elem_type) + ' ' + str(2) + ' ' + str(phys_tag) + ' ' + str(entity))
            #for x in range(len(connectivity)):
            #    f.write(' ' + str(connectivity[x]))
            f.write(' ' + str(connectivity[1]))
            f.write(' ' + str(connectivity[2]))
            f.write(' ' + str(connectivity[0]))
            f.write(' ' + str(connectivity[4]))
            f.write(' ' + str(connectivity[5]))
            f.write(' ' + str(connectivity[3]))
            f.write('\n')
        f.close()
        elem_id+=1

with open(mshfile, 'a') as f:
    f.write('$EndElements\n')
f.close()
