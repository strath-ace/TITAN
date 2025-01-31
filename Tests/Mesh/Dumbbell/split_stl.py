import sys
import stl
import os

filename = sys.argv[1]
binary = True

fp = open(filename, "r")

while True:
    chunk = fp.readline()
    if chunk == '':
        fp.close()
        break

    _list = chunk.split()
    if _list:
        if _list[0] == 'solid':
            f2 = open(_list[1] + '.stl', "w")
            f2.write(chunk)

        elif _list[0] == 'endsolid':
            f2.write(chunk)
            f2.close()
            if binary: your_mesh = stl.mesh.Mesh.from_file(_list[1] + '.stl')     
            if binary: your_mesh.save(_list[1] + '.stl')   
        
        else: f2.write(chunk)

    
