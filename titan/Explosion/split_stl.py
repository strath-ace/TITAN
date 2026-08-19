"""split_stl module."""
import sys
import stl
import os


def stl_split(directory,filename):
    """Documentation for the function.
:param directory: Filesystem directory path.
:type directory: str
:param filename: Path to the relevant file.
:type filename: str"""
    fp = open(filename, "r")
    reading_file = True
    while reading_file:
        chunk = fp.readline()
        if chunk == '':
            fp.close()
            break

        _list = chunk.split()
        if _list:
            if _list[0] == 'solid':
                f2 = open(directory+'/'+_list[1] + '.stl', "w")
                f2.write(chunk)

            elif _list[0] == 'endsolid':
                f2.write(chunk)
                f2.close()
            
            else: f2.write(chunk)

if __name__=='__main__':
    filename = sys.argv[1]
    binary = False

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

    
