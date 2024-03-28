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
import os
import subprocess

class Amg():
    def __init__(self, p=4,c = 100000, hgrad = 1.6, sensor = 'Mach'):
        self.p = p
        self.c = c
        self.hgrad = hgrad
        self.sensor = sensor

    def set_p(self,value):
        self.p = value

    def set_complex(self,value):
        self.c = value

    def set_sensor(self,value):
        self.sensor = value

    def set_hgrad(self,value):
        self.hgrad = value

def adapt_mesh(amg, iteration, options,j, num_obj,input_grid, output_grid):

    p = amg.p
    c = amg.c
    sensor = amg.sensor
    hgrad = amg.hgrad
    adapt_surf = ' '+str(num_obj+2)+','+str(num_obj+3)
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    subprocess.run(['python',path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m',options.output_folder+'/CFD_Grid/'+ input_grid+'.su2', '-s', options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j) + '.csv','-o',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j)])
    subprocess.run(['python',path+'/Executables/su2io/su2gmf/su2_to_gmf.py', '-m',options.output_folder+'/CFD_Grid/'+ input_grid+'.su2', '-s', options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j) + '.csv','-f',sensor,'-o', options.output_folder+'/CFD_Grid/Amg/sensor_'+str(j)])

    subprocess.run([path+'/Executables/feflo.a' ,'-in', options.output_folder+'/CFD_Grid/Amg/amg_'+ str(j) +'.meshb', '-sol',options.output_folder+'/CFD_Grid/Amg/sensor_'+str(j)+'.solb', '-p', str(p) ,  '-c' , str(c) ,'-hgrad',str(hgrad),'-itp',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j)+'.solb','-out',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j+1),'-adap-surf-ids',adapt_surf])
    subprocess.run(['python', path+'/Executables/su2io/su2gmf/gmf_to_su2.py', '-m', options.output_folder+'/CFD_Grid/Amg/amg_'+str(j+1)+'.meshb', '-b', options.output_folder+'/CFD_Grid/'+ input_grid+'.su2','-s',options.output_folder+'/CFD_Grid/Amg/amg_'+str(j)+'.itp.solb','-o', options.output_folder+'/CFD_Grid/' + output_grid])
    
    subprocess.run(['mv',options.output_folder+'/CFD_Grid/'+output_grid+'.csv',options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j+1) + '.csv'])
    subprocess.run(['cp',options.output_folder+'/CFD_sol/restart_flow_' + str(iteration) + '_adapt_' + str(j+1) + '.csv',options.output_folder+'/CFD_sol/restart_flow.csv'])