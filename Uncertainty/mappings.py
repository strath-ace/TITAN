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

from scipy.stats import *
from scipy.spatial.transform import Rotation
import numpy as np
## Lot of messy mapping information here to enable uq parameters to be easily exposed

## Addresses are of form 'access;pointer,access;pointer,etc'
def assembly_address(index): return 'TITAN,attr;assembly,index;'+str(index)

def trajectory_address(index): return assembly_address(index) + ',attr;trajectory'

def component_address(assembly_index, component_index): 
    return assembly_address(assembly_index) + ',attr;objects,index;'+str(component_index)

def option_aero_address(assembly_index): return 'OPTIONS,attr;aerothermo'

def option_traj_address(assembly_index): return 'OPTIONS,attr;dynamics'

def get_component_index_from_name(name, titan, assembly_index=0):
    for i_component, component in enumerate(titan.assembly[assembly_index].objects):
        if name in component.name: return i_component
    raise Exception('Could not find component {} in assembly {}'.format(name, assembly_index))

# Here parameters are mapped to relevant objects
library_addresses = {assembly_address    : ['ECEF_x','ECEF_y','ECEF_z','ECEF_u','ECEF_v','ECEF_w', 'quat_w', 
                                            'quat_x', 'quat_y','quat_z','omega_roll', 'omega_pitch', 'omega_yaw', 
                                            'roll', 'aoa','slip','Ixx', 'Iyy', 'Izz', 'Ixy', 'Iyz', 'Ixz'],
                     trajectory_address  : ['altitude','gamma','chi','velocity','latitude','longitude'],
                     component_address   : ['trigger__','temperature__'],
                     option_aero_address : ['catalycity'],
                     option_traj_address : ['ECI_x','ECI_y','ECI_z','ECI_u','ECI_v','ECI_w','ECI_epoch_UNIX']}


# Here parameters are mapped to relevant attributes (of mapped objects)
library_assignments = {'ECEF_x'         : ['state_vector', 0],
                       'ECEF_y'         : ['state_vector', 1],
                       'ECEF_z'         : ['state_vector', 2],
                       'ECEF_u'         : ['state_vector', 3],
                       'ECEF_v'         : ['state_vector', 4],
                       'ECEF_w'         : ['state_vector', 5],
                       'quat_x'         : ['state_vector', 6],
                       'quat_y'         : ['state_vector', 7],
                       'quat_z'         : ['state_vector', 8],
                       'quat_w'         : ['state_vector', 9],
                       'omega_roll'     : ['state_vector', 10],
                       'omega_pitch'    : ['state_vector', 11],
                       'omega_yaw'      : ['state_vector', 12],
                       'altitude'       : ['altitude'],
                       'gamma'          : ['gamma'],
                       'chi'            : ['chi'],
                       'velocity'       : ['velocity'],
                       'latitude'       : ['latitude'],
                       'longitude'      : ['longitude'],
                       'ECI_x'          : ['trajectory_state', 0],
                       'ECI_y'          : ['trajectory_state', 1],
                       'ECI_z'          : ['trajectory_state', 2],
                       'ECI_u'          : ['trajectory_state', 3],
                       'ECI_v'          : ['trajectory_state', 4],
                       'ECI_w'          : ['trajectory_state', 5],                       
                       'roll'           : ['roll'],
                       'aoa'            : ['aoa'],
                       'slip'           : ['slip'],
                       'trigger__'      : ['trigger_value'],
                       'temperature__'  : ['temperature'],
                       'Ixx'            : ['inertia', [0,0]],
                       'Iyy'            : ['inertia', [1,1]],
                       'Izz'            : ['inertia', [2,2]],
                       'Ixy'            : ['inertia', [0,1]],
                       'Iyz'            : ['inertia', [1,2]],
                       'Ixz'            : ['inertia', [0,2]],
                       'ECI_epoch_UNIX' : ['trajectory_epoch'],
                       'catalycity'     : ['cat_rate']
                       }

def library_check(name):
    try: assert name in list(library_assignments.keys())
    except: 
        raise Exception('Could not find parameter {} in list, please check the UQ handbook'.format(name))

def state_vector_callback(titan, options, assem_ids):
    from Dynamics.propagation import construct_state_vector
    for i_assem in assem_ids: construct_state_vector(titan.assembly[i_assem])

def dynamic_attributes_callback(titan, options, assem_ids):
    from Dynamics.propagation import update_dynamic_attributes
    for i_assem in assem_ids: update_dynamic_attributes(titan.assembly[i_assem],
                                                        titan.assembly[i_assem].state_vector,
                                                        options,
                                                        force=True)

def ECI_position_callback(titan, options, assem_ids):
    import pymap3d
    import datetime as dt
    epoch = options.dynamics.trajectory_epoch
    if isinstance(epoch, str):
        epoch = dt.datetime.strptime(epoch,'%Y/%m/%d %H:%M:%S')
    elif isinstance(epoch, float) or isinstance(epoch, int): epoch = dt.datetime.fromtimestamp(epoch)
    for i_assem in assem_ids:
        state = options.dynamics.trajectory_state
        state[:3] = pymap3d.eci2ecef(state[0],state[1],state[2],time=epoch)
        rotMatrix = np.transpose([pymap3d.eci2ecef(1,0,0,epoch),
                                  pymap3d.eci2ecef(0,1,0,epoch),
                                  pymap3d.eci2ecef(0,0,1,epoch)])
        state[3:] = rotMatrix @ state[3:] -np.cross([0,0,options.planet.omega()],state[:3])
        titan.assembly[i_assem].state_vector[:6] = state
    dynamic_attributes_callback(titan, options, assem_ids)

def orientation_callback(titan, options, assem_ids):
    from Dynamics.dynamics import compute_quaternion
    for i_assem in assem_ids: 
        compute_quaternion(titan.assembly[i_assem])
        titan.assembly[i_assem].state_vector[6:10] = titan.assembly[i_assem].quaternion
    state_vector_callback(titan, options, assem_ids)


def symmetrize_inertia_callback(titan, options, assem_ids):
    for i_assem in assem_ids:
        titan.assembly[i_assem].inertia[1,0] = titan.assembly[i_assem].inertia[0,1]
        titan.assembly[i_assem].inertia[2,0] = titan.assembly[i_assem].inertia[0,2]
        titan.assembly[i_assem].inertia[2,1] = titan.assembly[i_assem].inertia[1,2]

callbacks = {state_vector_callback       : ['altitude','latitude','longitude','gamma','chi','velocity'],
             ECI_position_callback       : ['ECI_x','ECI_y','ECI_z','ECI_u','ECI_v','ECI_w','ECI_epoch_UNIX'],
             dynamic_attributes_callback : ['ECEF_x','ECEF_y','ECEF_z','ECEF_u','ECEF_v','ECEF_w', 'quat_w',
                                            'quat_x','quat_y','quat_z','omega_roll','omega_pitch','omega_yaw'],
             orientation_callback        : ['chi','gamma','roll','aoa','slip','latitude','longitude'],
             symmetrize_inertia_callback : ['Ixy','Iyz','Ixz']}

## Valid classes for the distribution have to...
#  ...accept (even a dummy) input
#  ...have the attribute random_state
#  ...have the method rvs()
class quat_rotation():
    def __init__(self, d=3):
        self.d = 3
        self.random_state = None
    def rvs(self, n=1):
        quat = Rotation.random(num=n, random_state=self.random_state).as_quat()
        return quat[0]

# Sometimes extensibility ain't pretty
available_distris = {'alpha':alpha,'anglit':anglit,'arcsine':arcsine,'argus':argus,'beta':beta,'betaprime':betaprime,
               'bradford':bradford,'burr':burr,'burr12':burr12,'cauchy':cauchy,'chi':chi,'chi2':chi2,'cosine':cosine,
               'crystalball':crystalball,'dgamma':dgamma,'dweibull':dweibull,'erlang':erlang,'expon':expon,
               'exponnorm':exponnorm,'exponweib':exponweib,'exponpow':exponpow,'f':f,'fatiguelife':fatiguelife,
               'fisk':fisk,'foldcauchy':foldcauchy,'foldnorm':foldnorm,'genlogistic':genlogistic,'gennorm':gennorm,
               'genpareto':genpareto,'genexpon':genexpon,'genextreme':genextreme,'gausshyper':gausshyper,'gamma':gamma,
               'gengamma':gengamma,'genhalflogistic':genhalflogistic,'genhyperbolic':genhyperbolic,
               'geninvgauss':geninvgauss,'gibrat':gibrat,'gompertz':gompertz,'gumbel_r':gumbel_r,'gumbel_l':gumbel_l,
               'halfcauchy':halfcauchy,'halflogistic':halflogistic,'halfnorm':halfnorm,'halfgennorm':halfgennorm,
               'hypsecant':hypsecant,'invgamma':invgamma,'invgauss':invgauss,'invweibull':invweibull,
               'johnsonsb':johnsonsb,'johnsonsu':johnsonsu,'kappa4':kappa4,'kappa3':kappa3,'ksone':ksone,'kstwo':kstwo,
               'kstwobign':kstwobign,'laplace':laplace,'laplace_asymmetric':laplace_asymmetric,'levy':levy,
               'levy_l':levy_l,'levy_stable':levy_stable,'logistic':logistic,'loggamma':loggamma,'loglaplace':loglaplace,
               'lognorm':lognorm,'loguniform':loguniform,'lomax':lomax,'maxwell':maxwell,'mielke':mielke,'moyal':moyal,
               'nakagami':nakagami,'ncx2':ncx2,'ncf':ncf,'nct':nct,'norm':norm,'normal':norm,'norminvgauss':norminvgauss,
               'pareto':pareto,'pearson3':pearson3,'powerlaw':powerlaw,'powerlognorm':powerlognorm,'powernorm':powernorm,
               'rdist':rdist,'rayleigh':rayleigh,'rice':rice,'recipinvgauss':recipinvgauss,'semicircular':semicircular,
               'skewcauchy':skewcauchy,'skewnorm':skewnorm,'studentized_range':studentized_range,'t':t,
               'trapezoid':trapezoid,'triang':triang,'truncexpon':truncexpon,'truncnorm':truncnorm,
               'truncpareto':truncpareto,'truncweibull_min':truncweibull_min,'tukeylambda':tukeylambda,
               'uniform':uniform,'vonmises':vonmises,'vonmises_line':vonmises_line,'wald':wald,'weibull_min':weibull_min,
               'weibull_max':weibull_max,'wrapcauchy':wrapcauchy, 'multivariate_normal':multivariate_normal,
               'matrix_normal':matrix_normal,'dirichlet':dirichlet,'wishart':wishart,'invwishart':invwishart,
               'multinomial':multinomial,'special_ortho_group':special_ortho_group,'ortho_group':ortho_group,
               'unitary_group':unitary_group,'random_correlation':random_correlation,'multivariate_t':multivariate_t,
               'multivariate_hypergeom':multivariate_hypergeom,'random_table':random_table,
               'uniform_direction':uniform_direction,'bernoulli':bernoulli,'betabinom':betabinom,'binom':binom,
               'boltzmann':boltzmann,'dlaplace':dlaplace,'geom':geom,'hypergeom':hypergeom,'logser':logser,
               'nbinom':nbinom,'nchypergeom_fisher':nchypergeom_fisher,'nchypergeom_wallenius':nchypergeom_wallenius,
               'nhypergeom':nhypergeom,'planck':planck,'poisson':poisson,'randint':randint,'skellam':skellam,
               'yulesimon':yulesimon,'zipf':zipf,'zipfian':zipfian, 'quat_rotation':quat_rotation}
