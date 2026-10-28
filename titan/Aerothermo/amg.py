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
"""amg module."""
import os
import subprocess

class Amg():
    """Store parameters for adaptive mesh generation with FEFLO/AMG."""
    def __init__(self, p=4,c = 100000, hgrad = 1.6, sensor = 'Mach'):
        """Initialize adaptation controls.

        :param p: Adaptation parameter passed to the mesher.
        :type p: int
        :param c: Complexity target passed to the mesher.
        :type c: int
        :param hgrad: Maximum mesh-size gradation.
        :type hgrad: float
        :param sensor: Flow variable used to construct the adaptation sensor.
        :type sensor: str
        """
        self.p = p
        self.c = c
        self.hgrad = hgrad
        self.sensor = sensor

    def set_p(self,value):
        """Set the adaptation parameter.

        :param value: New adaptation parameter.
        :type value: int
        """
        self.p = value

    def set_complex(self,value):
        """Set the target mesh complexity.

        :param value: New complexity target.
        :type value: int
        """
        self.c = value

    def set_sensor(self,value):
        """Select the flow variable used for adaptation.

        :param value: Sensor name, such as ``'Mach'``.
        :type value: str
        """
        self.sensor = value

    def set_hgrad(self,value):
        """Set the maximum mesh-size growth rate between neighbouring cells.

        :param value: Mesh gradation limit.
        :type value: float
        """
        self.hgrad = value

def adapt_mesh(amg, iteration, options,j, num_obj,input_grid, output_grid):
    """Run one adaptive-mesh pass and update the SU2 grid and restart data.

:param amg: Adaptation parameters passed to the mesh generator.
:type amg: Any
:param iteration: Current simulation iteration used to select the restart solution.
:type iteration: Any
:param options: Options or configuration object.
:type options: Options
:param j: Adaptation-pass index.
:type j: Any
:param num_obj: Number of objects whose surfaces are included in adaptation.
:type num_obj: int
:param input_grid: Input SU2 grid basename.
:type input_grid: Any
:param output_grid: Output SU2 grid basename.
:type output_grid: Any
"""

    p = amg.p
    c = amg.c
    sensor = amg.sensor
    hgrad = amg.hgrad
    adapt_surf = ' '+str(num_obj+2)+','+str(num_obj+3)
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    subprocess.run(['python',path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m',options.output_folder+'/CFD_Grid/'+ input_grid+'.su2', '-s', options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j) + '.csv','-o',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j)])
    subprocess.run(['python',path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m',options.output_folder+'/CFD_Grid/'+ input_grid+'.su2', '-s', options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j) + '.csv','-f',sensor,'-o', options.output_folder+'/CFD_Grid/Amg/sensor_'+str(j)])

    subprocess.run([path+'/Executables/feflo.a' ,'-in', options.output_folder+'/CFD_Grid/Amg/amg_'+ str(j) +'.meshb', '-sol',options.output_folder+'/CFD_Grid/Amg/sensor_'+str(j)+'.solb', '-p', str(p) ,  '-c' , str(c) ,'-hgrad',str(hgrad),'-itp',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j)+'.solb','-out',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j+1),'-adap-surf-ids',adapt_surf])
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/gmf_to_su2.py', '-m', options.output_folder+'/CFD_Grid/Amg/amg_'+str(j+1)+'.meshb', '-b', options.output_folder+'/CFD_Grid/'+ input_grid+'.su2','-s',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j)+'.itp.solb','-o', options.output_folder+'/CFD_Grid/' + output_grid])
    
    subprocess.run(['mv',options.output_folder+'/CFD_Grid/'+output_grid+'.csv',options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j+1) + '.csv'])
    subprocess.run(['cp',options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j+1) + '.csv',options.output_folder+'/CFD_sol/restart_flow.csv'])
