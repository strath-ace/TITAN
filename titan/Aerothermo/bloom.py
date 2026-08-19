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
"""bloom module."""
import subprocess
import os

class Bloom():
    """Bloom."""
    def __init__(self, flag = False, layers = 20, spacing = 0.0006, growth_rate = 1.075):
        """Documentation for the function.
:param flag: Flag for flag.
:type flag: bool
:param layers: Value for layers.
:type layers: Any
:param spacing: Value for spacing.
:type spacing: Any
:param growth_rate: Value for growth rate.
:type growth_rate: Any"""
        self.flag = flag
        self.layers = layers
        self.spacing = spacing
        self.growth_rate = growth_rate

    def set_flag(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.flag = value

    def set_layers(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.layers = value

    def set_spacing(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.spacing = value

    def set_growth_rate(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.growth_rate = value


def create_bloom_config(num_obj, bloom, options, path):
    """Documentation for the function.
:param num_obj: Integer value for num obj.
:type num_obj: int
:param bloom: Value for bloom.
:type bloom: Any
:param options: Options or configuration object.
:type options: object
:param path: Value for path.
:type path: str"""

    try: os.remove(options.output_folder + path)
    except: pass

    with open(options.output_folder + path + 'bloom', 'w') as f:      

        f.write('BLReference \n')
        f.write(str(int(num_obj))+ ' \n')        
        for i in range(num_obj): 
            f.write(str(int(i+2)) + ' ')        
        f.write('\n \n')

       # f.write('BLSymmetryReference \n')
       # f.write('0 \n')
       # f.write('\n \n')

        f.write('NumberOfLayers \n')
        f.write(str(int(bloom.layers))+ ' \n')
        f.write('\n \n')

        f.write('InitialSpacing \n')
        f.write(str(bloom.spacing)+ ' \n')
        f.write('\n \n')

        f.write('GrowthRate \n')
        f.write(str(bloom.growth_rate)+ ' \n')
        f.write('\n \n')

        f.write('MeshDeformationReference \n')
        f.write('1 \n1 \n')

    f.close()
    pass


def generate_BL_CFD(j, options, num_obj, bloom, input_grid, output_grid):
    """Documentation for the function.
:param j: Value for j.
:type j: Any
:param options: Options or configuration object.
:type options: object
:param num_obj: Integer value for num obj.
:type num_obj: int
:param bloom: Value for bloom.
:type bloom: Any
:param input_grid: Value for input grid.
:type input_grid: Any
:param output_grid: Value for output grid.
:type output_grid: Any"""
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path_folder = '/CFD_Grid/Bloom/'

    create_bloom_config(num_obj, bloom, options, path_folder)
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m' ,options.output_folder +'/CFD_Grid/'+input_grid+'.su2','-o',options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)])
    subprocess.run([path+'/Executables/amg_bloom', '-in', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'.meshb', '-bl-data',options.output_folder+'/CFD_Grid/Bloom/bloom', '-bl-hybrid', '-out', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'_BL', '-hmsh'])
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/gmf_to_su2.py', '-m', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'_BL.meshb', '-b', options.output_folder +'/CFD_Grid/'+input_grid+'.su2', '-o', options.output_folder+'/CFD_Grid/'+output_grid])

def generate_PATO_mesh(options, object_id, bloom):
    """Documentation for the function.
:param options: Options or configuration object.
:type options: object
:param object_id: Integer value for object id.
:type object_id: int
:param bloom: Value for bloom.
:type bloom: Any"""

    print('Generating PATO mesh ...')

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path_folder = '/PATO_'+str(object_id)+'/mesh/'

    input_grid = 'mesh'
    output_grid = 'hybrid_mesh'

    create_bloom_config(1, bloom, options, path_folder)

    subprocess.run(['python', path+'/Executables/su2_to_gmf.py', '-m' ,options.output_folder + path_folder +input_grid+'.su2','-o',options.output_folder+path_folder+input_grid])
    subprocess.run([path+'/Executables/amg_bloom', '-in', options.output_folder+path_folder+input_grid+'.meshb', '-bl-data',options.output_folder+path_folder+"bloom", '-bl-hybrid', '-out', options.output_folder+path_folder+input_grid+'_PL', '-hmsh'])
    subprocess.run(['python', path+'/Executables/gmf_to_su2.py', '-m', options.output_folder+path_folder+input_grid+'_PL.meshb', '-b', options.output_folder +path_folder+input_grid+'.su2', '-o', options.output_folder+path_folder+output_grid])
    subprocess.run(['python', path+'/Executables/su2tomsh-amg.py', options.output_folder+path_folder+output_grid+".su2"])
    subprocess.run(['mv', 'mesh.msh', options.output_folder+path_folder])
    subprocess.run(['rm', '/'+options.output_folder+path_folder+'*.meshb'])
    subprocess.run(['rm', '/'+options.output_folder+path_folder+'*.su2'])
