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
    """Amg."""
    def __init__(self, p=4,c = 100000, hgrad = 1.6, sensor = 'Mach'):
        """Documentation for the function.
:param p: Value for p.
:type p: Any
:param c: Value for c.
:type c: Any
:param hgrad: Value for hgrad.
:type hgrad: Any
:param sensor: Value for sensor.
:type sensor: Any"""
        self.p = p
        self.c = c
        self.hgrad = hgrad
        self.sensor = sensor

    def set_p(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.p = value

    def set_complex(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.c = value

    def set_sensor(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.sensor = value

    def set_hgrad(self,value):
        """Documentation for the function.
:param value: Numeric value for value.
:type value: float"""
        self.hgrad = value

def adapt_mesh(amg, iteration, options,j, num_obj,input_grid, output_grid):
    """Documentation for the function.
:param amg: Value for amg.
:type amg: Any
:param iteration: Value for iteration.
:type iteration: Any
:param options: Options or configuration object.
:type options: object
:param j: Value for j.
:type j: Any
:param num_obj: Integer value for num obj.
:type num_obj: int
:param input_grid: Value for input grid.
:type input_grid: Any
:param output_grid: Value for output grid.
:type output_grid: Any"""

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
