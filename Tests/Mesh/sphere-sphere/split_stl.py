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

    
    
