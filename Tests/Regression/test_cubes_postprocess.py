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
import subprocess

def test_postprocess():
    subprocess.run(['python','TITAN.py','--config','Tests/Configs/2cube_postprocess.txt'], text = True)
    subprocess.run(['python','TITAN.py','--config','Tests/Configs/2cube_postprocess.txt','-pp','wind'], text = True)
    subprocess.run(['python','TITAN.py','--config','Tests/Configs/2cube_postprocess.txt','-pp','ECEF'], text = True)
