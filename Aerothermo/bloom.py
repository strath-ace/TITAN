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
import subprocess
import os

class Bloom():
    def __init__(self, flag = False, layers = 20, spacing = 0.0006, growth_rate = 1.075):
        self.flag = flag
        self.layers = layers
        self.spacing = spacing
        self.growth_rate = growth_rate

    def set_flag(self,value):
        self.flag = value

    def set_layers(self,value):
        self.layers = value

    def set_spacing(self,value):
        self.spacing = value

    def set_growth_rate(self,value):
        self.growth_rate = value


def create_bloom_config(num_obj,bloom, options):

    try: os.remove(options.output_folder+'/CFD_Grid/Bloom/bloom')
    except: pass

    with open(options.output_folder+'/CFD_Grid/Bloom/bloom', 'w') as f:

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


def generate_BL(j, options, num_obj, bloom, input_grid, output_grid):
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    create_bloom_config(num_obj,bloom, options)
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m' ,options.output_folder +'/CFD_Grid/'+input_grid+'.su2','-o',options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)])
    subprocess.run([path+'/Executables/amg_bloom', '-in', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'.meshb', '-bl-data',options.output_folder+'/CFD_Grid/Bloom/bloom', '-bl-hybrid', '-out', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'_BL', '-hmsh'])
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/gmf_to_su2.py', '-m', options.output_folder+'/CFD_Grid/Bloom/'+input_grid+str(j)+'_BL.meshb', '-b', options.output_folder +'/CFD_Grid/'+input_grid+'.su2', '-o', options.output_folder+'/CFD_Grid/'+output_grid])