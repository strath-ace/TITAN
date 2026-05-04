#
# Copyright (c) 2026 TITAN Contributors (cf. AUTHORS.md).
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
def assembly_address(assembly_index : int) -> str: 
    """Maps from an assembly index to an ""address" string targeting it as an object

    :param assembly_index: Index of target assembly
    :type index: int
    :return: Resultant address
    :rtype: str
    """
    return 'TITAN,attr;assembly,index;'+str(assembly_index)

def trajectory_address(assembly_index : int) -> str: 
    """Maps from an assembly index to an ""address" string targeting its
    trajectory attribute

    :param index: Index of target assembly
    :type index: int
    :return: Resultant address
    :rtype: str
    """
    return assembly_address(assembly_index) + ',attr;trajectory'

def component_address(assembly_index : int, component_index : int) -> str: 
    """Maps from assembly and component indices to an ""address" string 
    targeting the component as an object

    :param assembly_index: Index of target assembly
    :type assembly_index: int
    :param component_index: Index of target component
    :type component_index: int
    :return: Resultant address
    :rtype: str
    """
    return assembly_address(assembly_index) + ',attr;objects,index;'+str(component_index)

def option_aero_address(assembly_index :int) -> str: 
    """Maps from an assembly index to an ""address" string targeting its
    aerothermo attribute

    :param index: Index of target assembly
    :type index: int
    :return: Resultant address
    :rtype: str
    """
    return 'OPTIONS,attr;aerothermo'

def option_traj_address(index : int) -> str: 
    """Maps to an ""address" string targeting the trajectory attribute of the options 
    object

    :param index: Index, dummy variable to preserve signature
    :type index: int
    :return: Resultant address
    :rtype: str
    """
    return 'OPTIONS,attr;dynamics'

def option_free_address(index : int) -> str: 
    """Maps to an ""address" string targeting the freestream attribute of the options 
    object

    :param index: Index, dummy variable to preserve signature
    :type index: int
    :return: Resultant address
    :rtype: str
    """
    
    return 'OPTIONS,attr;freestream'

def get_component_index_from_name(name : str, titan, assembly_index=0) -> int:
    """Retrieves the index of a target component using its name

    :param name: Component name
    :type name: str
    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param assembly_index: Index of assembly to retreive component from, defaults to 0
    :type assembly_index: int, optional
    :return: Component index 
    :rtype: int
    """
    print('Retrieving {} from assembly {}'.format(name, assembly_index))
    for i_component, component in enumerate(titan.assembly[assembly_index].objects):
        if name in component.name: return i_component
    raise Exception('Could not find component {} in assembly {}'.format(name, assembly_index))

#: Parameter mappings to relevant objects
library_addresses = {assembly_address    : ['ECEF_x','ECEF_y','ECEF_z','ECEF_u','ECEF_v','ECEF_w', 'quat_w', 
                                            'quat_x', 'quat_y','quat_z','omega_roll', 'omega_pitch', 'omega_yaw', 
                                            'roll', 'aoa','slip','Ixx', 'Iyy', 'Izz', 'Ixy', 'Iyz', 'Ixz', 
                                            'RIC_x','RIC_y','RIC_z','RIC_u','RIC_v','RIC_w',
                                            'TVN_x','TVN_y','TVN_z','TVN_u','TVN_v','TVN_w'],
                     trajectory_address  : ['altitude','gamma','chi','velocity','latitude','longitude'],
                     component_address   : ['trigger__','temperature__'],
                     option_aero_address : ['catalycity','CP_mult','CTau_mult','CH_mult'],
                     option_traj_address : ['ECI_x','ECI_y','ECI_z','ECI_u','ECI_v','ECI_w','ECI_epoch_UNIX'],
                     option_free_address : ['density_mult']}


#: Parameter mappings to relevant attributes (of mapped objects)
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
                       'catalycity'     : ['cat_rate'],
                       'density_mult'   : ['density_mult'],
                       'CP_mult'        : ['CP_mult'],
                       'CTau_mult'      : ['CTau_mult'],
                       'CH_mult'        : ['CH_mult'],
                       'RIC_x'          : ['RIC', 0],
                       'RIC_y'          : ['RIC', 1],
                       'RIC_z'          : ['RIC', 2],
                       'RIC_u'          : ['RIC', 3],
                       'RIC_v'          : ['RIC', 4],
                       'RIC_w'          : ['RIC', 5],
                       'TVN_x'          : ['RIC', 0],
                       'TVN_y'          : ['RIC', 1],
                       'TVN_z'          : ['RIC', 2],
                       'TVN_u'          : ['RIC', 3],
                       'TVN_v'          : ['RIC', 4],
                       'TVN_w'          : ['RIC', 5],
                       }

def library_check(name : str):
    """Searches the UQ library for the specified parameter, throws an exception if it doesn't exist

    :param name: Name of target parameter
    :type name: str
    :raises Exception: Parameter not found in UQ library
    """
    try: assert name in list(library_assignments.keys())
    except: 
        raise Exception('Could not find parameter {} in list, please check the UQ handbook'.format(name))

def state_vector_callback(titan, options, assem_ids:list):
    """
    Constructs the ECEF states based upon dynamic attributes for a set of assemblies defined by assem_ids

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """
    from Dynamics.propagation import construct_state_vector
    for i_assem in assem_ids: construct_state_vector(titan.assembly[i_assem], options.dynamics.augmented_state)

def dynamic_attributes_callback(titan, options, assem_ids : list):
    """
    Recalculates the dynamic attributes based upon ECEF states
    of a set of assemblies defined by assem_ids

    This function is a "callback" in that it is called after UQ mapping

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """
    from Dynamics.propagation import update_dynamic_attributes
    for i_assem in assem_ids: 
        update_dynamic_attributes(titan.assembly[i_assem],
                                  titan.assembly[i_assem].state_vector,
                                  options,
                                  force=True)

def ECI_position_callback(titan, options, assem_ids : list):
    """
    Recalculates the (ECEF) state vectors based upon ECI states
    and epochs of a set of assemblies defined by assem_ids

    This function is a "callback" in that it is called after UQ mapping

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """
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

def orientation_callback(titan, options, assem_ids : list):
    """
    Recomputes the quaternion of a of a set of assemblies defined by assem_ids

    This function is a "callback" in that it is called after UQ mapping

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """
    from Dynamics.dynamics import compute_quaternion
    for i_assem in assem_ids: 
        compute_quaternion(titan.assembly[i_assem])
        titan.assembly[i_assem].state_vector[6:10] = titan.assembly[i_assem].quaternion
    state_vector_callback(titan, options, assem_ids)

def RIC_callback(titan, options, assem_ids : list):
    """
    Adds pre-specified deltas in Radial, In-Track, Cross-Track (RIC/RTN) frame to
    the states of a set of assemblies defined by assem_ids

    This function is a "callback" in that it is called after UQ mapping

    The frame is defined as detailed in https://sanaregistry.org/r/orbit_relative_reference_frames/records/9

    Note that for circular orbits RIC and TVN are identical


    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """

    for i_assem in assem_ids:
        state = np.array(titan.assembly[i_assem].state_vector)
        delta_RIC = titan.assembly[i_assem].RIC
        r_hat = state[0:3] / np.linalg.norm(state[0:3])
        v_hat = state[3:6] / np.linalg.norm(state[3:6])
        
        # For RTN/RIC our T/I vector is not velocity aligned 
        x_track = np.cross(r_hat, v_hat)
        in_track = np.cross(x_track, r_hat) # i.e. this is not (necessarily) v_hat
        radial = r_hat

        delta_ECEF = np.zeros(6)
        # Position changes
        delta_ECEF[0:3] += delta_RIC[0] * radial + delta_RIC[1] * in_track + delta_RIC[2] * x_track
        # Velocity Changes
        delta_ECEF[3:6] += delta_RIC[3] * radial + delta_RIC[4] * in_track + delta_RIC[5] * x_track
        state[0:6] += delta_ECEF
        titan.assembly[i_assem].state = state

    dynamic_attributes_callback(titan, options, assem_ids)

def TVN_callback(titan, options, assem_ids : list):
    """
    Adds pre-specified deltas in Transverse, Velocity, Normal (TVN/NTW/PTW) frame to
    the states of a set of assemblies defined by assem_ids

    This function is a "callback" in that it is called after UQ mapping

    The frame is defined as detailed in https://sanaregistry.org/r/orbit_relative_reference_frames/records/6

    Note that for circular orbits RIC and TVN are identical

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """

    for i_assem in assem_ids:
        state = np.array(titan.assembly[i_assem].state_vector)
        delta_RIC = titan.assembly[i_assem].RIC
        r_hat = state[0:3] / np.linalg.norm(state[0:3])
        v_hat = state[3:6] / np.linalg.norm(state[3:6])
        
        # For TVN/NTW our T/N vector is not position aligned 
        x_track = np.cross(r_hat, v_hat)
        in_track = v_hat
        radial = np.cross(in_track, x_track) # i.e. this is not (necessarily) r_hat

        delta_ECEF = np.zeros(6)
        # Position changes
        delta_ECEF[0:3] += delta_RIC[0] * radial + delta_RIC[1] * in_track + delta_RIC[2] * x_track
        # Velocity Changes
        delta_ECEF[3:6] += delta_RIC[3] * radial + delta_RIC[4] * in_track + delta_RIC[5] * x_track
        state[0:6] += delta_ECEF
        titan.assembly[i_assem].state = state

def symmetrize_inertia_callback(titan, options, assem_ids : list):
    """
    Copies the lower in inertia tensor triangle assigned by UQ mapping to
    the upper inertia tensor to "symmetrize" the tensor of a set of assemblies 
    defined by assem_ids

    This function is a "callback" in that it is called after UQ mapping

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base TITAN options
    :type options: Options
    :param assem_ids: List of target assemblies
    :type assem_ids: list
    """
    for i_assem in assem_ids:
        titan.assembly[i_assem].inertia[1,0] = titan.assembly[i_assem].inertia[0,1]
        titan.assembly[i_assem].inertia[2,0] = titan.assembly[i_assem].inertia[0,2]
        titan.assembly[i_assem].inertia[2,1] = titan.assembly[i_assem].inertia[1,2]

#: Parameter mappings to functions that need to be called after UQ mapping has been completed
callbacks = {state_vector_callback       : ['altitude','latitude','longitude','gamma','chi','velocity'],
             ECI_position_callback       : ['ECI_x','ECI_y','ECI_z','ECI_u','ECI_v','ECI_w','ECI_epoch_UNIX'],
             RIC_callback                : ['RIC_x','RIC_y','RIC_z','RIC_u','RIC_v','RIC_w'],
             TVN_callback                : ['TVN_x','TVN_y','TVN_z','TVN_u','TVN_v','TVN_w'],
             dynamic_attributes_callback : ['ECEF_x','ECEF_y','ECEF_z','ECEF_u','ECEF_v','ECEF_w', 'quat_w',
                                            'quat_x','quat_y','quat_z','omega_roll','omega_pitch','omega_yaw'],
             orientation_callback        : ['chi','gamma','roll','aoa','slip','latitude','longitude'],
             symmetrize_inertia_callback : ['Ixy','Iyz','Ixz']}

#: Valid classes for the distribution have to...
#:  ...accept (even a dummy) input
#:  ...have the attribute random_state
#:  ...have the method rvs()
class quat_rotation():
    def __init__(self, d=3):
        self.d = 3
        self.random_state = None
    def rvs(self, n=1):
        quat = Rotation.random(num=n, random_state=self.random_state).as_quat()
        return quat[0]

#Sometimes extensibility ain't pretty
#: Available distributions for representing uncertain parameters
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
