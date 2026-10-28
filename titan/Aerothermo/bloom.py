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
    """Store boundary-layer prism-mesh settings for SU2 grid generation."""
    def __init__(self, flag = False, layers = 20, spacing = 0.0006, growth_rate = 1.075):
        """Initialize boundary-layer mesh controls.

        :param flag: Whether boundary-layer meshing is enabled.
        :type flag: bool
        :param layers: Number of prism layers to generate.
        :type layers: int
        :param spacing: Initial spacing of the prism layers.
        :type spacing: float
        :param growth_rate: Geometric growth ratio between successive layers.
        :type growth_rate: float
        """
        self.flag = flag
        self.layers = layers
        self.spacing = spacing
        self.growth_rate = growth_rate

    def set_flag(self,value):
        """Enable or disable boundary-layer meshing.

        :param value: Boundary-layer activation flag.
        :type value: bool
        """
        self.flag = value

    def set_layers(self,value):
        """Set the number of prism layers.

        :param value: Layer count.
        :type value: int
        """
        self.layers = value

    def set_spacing(self,value):
        """Set the initial prism-layer spacing.

        :param value: Initial layer spacing in the mesh's length units.
        :type value: float
        """
        self.spacing = value

    def set_growth_rate(self,value):
        """Set the prism-layer geometric growth ratio.

        :param value: Growth ratio applied between adjacent layers.
        :type value: float
        """
        self.growth_rate = value


def create_bloom_config(num_obj, bloom, options, path):
    """Write the boundary-layer configuration file consumed by SU2.

:param num_obj: Integer value for num obj.
:type num_obj: int
:param bloom: Boundary-layer settings to serialize.
:type bloom: Any
:param options: Options or configuration object.
:type options: Options
:param path: Output path relative to the simulation output folder.
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
    """Generate the next CFD grid with boundary-layer refinement.

:param j: Boundary-layer generation pass index.
:type j: Any
:param options: Options or configuration object.
:type options: Options
:param num_obj: Number of surface objects in the mesh.
:type num_obj: int
:param bloom: Boundary-layer mesh settings.
:type bloom: Any
:param input_grid: Input SU2 grid basename.
:type input_grid: Any
:param output_grid: Output SU2 grid basename.
:type output_grid: Any
"""
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path_folder = '/CFD_Grid/Bloom/'

    create_bloom_config(num_obj, bloom, options, path_folder)
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m' ,options.output_folder +'/CFD_Grid/'+input_grid+'.su2','-o',options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)])
    subprocess.run([path+'/Executables/amg_bloom', '-in', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'.meshb', '-bl-data',options.output_folder+'/CFD_Grid/Bloom/bloom', '-bl-hybrid', '-out', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'_BL', '-hmsh'])
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/gmf_to_su2.py', '-m', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'_BL.meshb', '-b', options.output_folder +'/CFD_Grid/'+input_grid+'.su2', '-o', options.output_folder+'/CFD_Grid/'+output_grid])

def generate_PATO_mesh(options, object_id, bloom):
    """Documentation for the function.
:param options: Options or configuration object.
:type options: Options
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
