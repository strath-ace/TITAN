import os
import numpy as np
from pyapprox import variables
from scipy.stats import *
from scipy.stats._multivariate import multi_rv_frozen
from statsmodels.stats.correlation_tools import cov_nearest
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
import pandas as pd
from Uncertainty.dynamics_tools import apply_velocity_wind
from Dynamics.frames import *
from Dynamics.dynamics import compute_cartesian,compute_cartesian_derivatives, compute_quaternion, compute_angular_derivatives
from pymap3d import ecef2geodetic, geodetic2ecef, Ellipsoid
from Configuration.configuration import Trajectory
from matplotlib import cm, pyplot as plt
import seaborn as sns
import pickle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PolyCollection
import elevation
import rasterio
import datetime as dt
import warnings
from collections.abc import MutableMapping
from copy import copy
try:
    from messaging import messenger
except:
    pass


# Sometimes extensibility ain't pretty
distri_list = {'alpha':alpha,'anglit':anglit,'arcsine':arcsine,'argus':argus,'beta':beta,'betaprime':betaprime,
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
               'yulesimon':yulesimon,'zipf':zipf,'zipfian':zipfian}
class uncertaintySupervisor():

    def __init__(self, isActive = False, rngSeed = 'Auto'):

        # [Bool] Allow the uncertainty supervisor to step in?
        self.isActive = isActive
        # [Float] Seed for variable sampling, should be unique to each instance to ensure proper randomness
        if not rngSeed == 'Auto': self.RNG = np.random.RandomState(seed=rngSeed)
        else: self.RNG = np.random.RandomState(seed = os.getpid() + dt.datetime.now().microsecond)
        self.special_cases = {'deorbit': '','rotation': ''}
        self.special_funcs = {'deorbit': self.deorbitBurn,'rotation':self.applyRotation}

    def constructInputs(self,filepath):

        # If you can describe a function using a scipy.stats distribution you can use it in TITAN...
        # just specify distribution function as a dict with name-value inputs as a dict

        inputs = []
        self.inputLabels =[]
        included_cases=[]
        with open(filepath,'r') as file: uncertainties = yaml.load(file,Loader)

        for key, value in uncertainties.items():

            if key.lower() in self.special_cases:
                self.special_cases[key] = value
                included_cases.append(key)
            else:
                if not len(value)==1:
                    for field in value.keys(): self.inputLabels.append(key + '<- Object : Field ->' + field)
                elif not value.copy().popitem()[0] in distri_list:
                    for field in value.keys(): self.inputLabels.append(key + '<- Object : Field ->' + field)
                else: self.inputLabels.append(key)

                for dist, moments in value.items():
                    if dist not in distri_list:
                        distribution, arguments = moments.copy().popitem()
                    else: distribution, arguments = dist, moments

                    inputs.append(distri_list[distribution](**arguments))

        self.inputs = variables.IndependentMarginalsVariable(inputs)

        for case in included_cases:
            self.inputLabels.append(case)


    def sample(self,n_samples):
        return self.inputs.rvs(n_samples,[self.RNG for i in range(self.inputs.nvars)])
    
    def sampleConfig(self,cfg):
        self.inputNames=self.inputLabels.copy()
        toAssign = self.inputNames.copy()
        vals = np.reshape(self.sample(1),-1)
        self.inputVector = []

        for name in self.inputLabels:
            if name in toAssign:
                if name in self.special_cases:
                    if len(toAssign)<=len(self.special_cases): # leave special cases to last
                        self.special_funcs[name](cfg,self.special_cases[name])
                        toAssign.remove(name)
                else:
                    self.inputVector.append(vals[self.inputNames.index(name)])
                    toAssign.remove(name)

            if len(toAssign)<1:
                cfg=self.configFromVector(cfg=cfg)
                return cfg

        raise Exception('Error assigning all inputs to cfg, you maybe made a typo? Inputs to assign: ',', '.join(toAssign))
    
    def assignFloat(self,value):
        # This function assigns a float that may or may not be uncertain
        # value must either be a float or of a specific dictionary structure as detailed in uncertainty.yaml
        if isinstance(value,dict):
                dist, args = value.copy().popitem()
                if dist in distri_list:
                    assigned = distri_list[dist](**args).rvs(random_state=self.RNG)
                else: raise Exception('Error assigning probability to value: ',value)
        else: assigned = value
        return assigned
    
    def configFromVector(self,cfg,vector=None):
        if isinstance(vector,(list,np.ndarray)): self.inputVector=vector
        toAssign = self.inputNames.copy()
        rotation = ['rot_x','rot_y','rot_z']
        omega = [ float(var) for var in cfg['Initial Conditions']['Angular Velocity'].split('[')[-1].split(']')[0].split(',')]

        for section in cfg.sections():
            for i_name, name in enumerate(self.inputNames):
                if name in toAssign:

                    if '<- Object : Field ->' in name:
                        target_object,target_field = name.split('<- Object : Field ->')

                        for object in cfg['Objects']:
                            if target_object==object:

                                new_val = target_field+'='+str(self.inputVector[i_name])
                                object_data = cfg.get('Objects',object).strip('[]').split(',')
                                fields = [f.split('=')[0] for f in object_data]

                                replacer = cfg['Objects'][object].rstrip(']') + ',' + new_val + ']'
                                for i_field, replace in enumerate([target_field in f for f in fields]):
                                    if replace:
                                        object_data[i_field] = new_val
                                        replacer = '['+','.join(object_data)+']'

                                cfg.set('Objects',object,replacer)
                                toAssign.remove(name)

                    if name.lower() in cfg.options(section):
                        cfg.set(section,name,str(self.inputVector[i_name]))
                        toAssign.remove(name)

                    if name in rotation:
                        omega[rotation.index(name)]=self.inputVector[i_name]
                        toAssign.remove(name)
                    

            if len(toAssign)<1: 
                self.writeRotation(cfg,omega)
                return cfg
        raise Exception('Error assigning all inputs to cfg, you maybe made a typo? Inputs to assign: ',', '.join(toAssign))

        
    def applyRotation(self,cfg,rotationData):

        if rotationData['tumbling']: 
            magnitude = self.assignFloat(rotationData['magnitude'])
            omega = magnitude*uniform_direction(dim=3).rvs(random_state=self.RNG)

        else:
            omega = []
            for axis in ['body_x','body_y','body_z']:
                rotation = self.assignFloat(rotationData[axis])
                omega.append(rotation)

        self.inputNames.remove('rotation')
        for i_ax, ax_name in enumerate(['rot_x','rot_y','rot_z']):
            self.inputNames.append(ax_name)
            self.inputVector.append(omega[i_ax])
        self.writeRotation(cfg,omega)
       

    def writeRotation(self, cfg, vector):
        cfg.set('Initial Conditions','Angular Velocity','(1:['+','.join(str(val) for val in vector)+'])')

    def deorbitBurn(self,cfg,burndata):
        # First calculate magnitude and direction (retrograde) of deorbit burn
        # Tsiolkovsky rocket eq...
        v_e = burndata['specific_impulse']*9.80665
        mass_ratio= (burndata['propellant_mass']+burndata['vehicle_mass'])/burndata['vehicle_mass']
        delta_v = v_e * np.log(mass_ratio)

        velocity = cfg.getfloat('Trajectory','Velocity')
        gamma = cfg.getfloat('Trajectory','Flight_path_angle')
        chi = cfg.getfloat('Trajectory','Heading_angle')

        trajectory = Trajectory(velocity=velocity,gamma=np.radians(gamma),chi=np.radians(chi))

        manoeuvre = [-delta_v, 0, 0] # just retrograde for now

        # Perturbation of manoeuvre...
        pf = burndata['sigma_pointing_fixed']
        pp = burndata['sigma_pointing_proportional']
        mf = burndata['sigma_magnitude_fixed']
        mp = burndata['sigma_magnitude_proportional']

        dx = norm(loc=0, scale=np.sqrt(mf ** 2 + mp ** 2 * delta_v)).rvs(random_state=self.RNG)
        dy = norm(loc=0, scale=np.sqrt(pf ** 2 + pp ** 2 * delta_v)).rvs(random_state=self.RNG)
        dz = norm(loc=0, scale=np.sqrt(pf ** 2 + pp ** 2 * delta_v)).rvs(random_state=self.RNG)

        perturbed_manoeuvre = manoeuvre + dx + dy + dz

        trajectory = apply_velocity_wind(trajectory, perturbed_manoeuvre)

        cfg.set('Trajectory', 'Velocity', str(trajectory.velocity))
        cfg.set('Trajectory', 'Flight_path_angle', str(np.degrees(trajectory.gamma)))
        cfg.set('Trajectory', 'Heading_angle', str(np.degrees(trajectory.chi)))
        
        if not 'Velocity' in self.inputNames:
            self.inputNames.append('Velocity')
            self.inputVector.append(trajectory.velocity)
        else:
            self.inputVector[self.inputNames.index('Velocity')] = trajectory.velocity

        if not 'Flight_path_angle' in self.inputNames:
            self.inputNames.append('Flight_path_angle')
            self.inputVector.append(np.degrees(trajectory.gamma))
        else:
            self.inputVector[self.inputNames.index('Flight_path_angle')] = np.degrees(trajectory.gamma)

        if not 'Heading_angle' in self.inputNames:
            self.inputNames.append('Heading_angle')
            self.inputVector.append(np.degrees(trajectory.chi))
        else:
            self.inputVector[self.inputNames.index('Heading_angle')] = np.degrees(trajectory.chi)
        
        self.inputNames.remove('deorbit')

        print('Performed manoeuvre with a delta v of ', np.round(abs(velocity - trajectory.velocity), 4),
            'm/s actual (',np.round(delta_v, 4), ' m/s ideal)')

        print('State before || Velocity:', np.round(velocity,4), 'm/s | Flight Path Angle:',
            np.round(gamma,4), 'deg | Heading Angle:', np.round(chi,4),'deg')

        print('State after || Velocity:', np.round(trajectory.velocity,4), 'm/s | Flight Path Angle:',
            np.round(np.degrees(trajectory.gamma),4), 'deg | Heading Angle:',
            np.round(np.degrees(trajectory.chi),4),'deg')

class uncertaintyHandler(MutableMapping):
    # This class looks after all uncertain variables, it can more or less be treated as a dict with some nice extra funcs
    def __init__(self,titan,options,rngSeed = 'Auto',filepath='uq.yaml'):
        # rngSeed can be either a numpy RandomState, 'Auto' to select one automatically (thread-safe) or a float
        if rngSeed == 'Auto':  self.RNG = np.random.RandomState(seed = os.getpid() + dt.datetime.now().microsecond)
        elif isinstance(rngSeed,np.random.RandomState): self.RNG=rngSeed
        else: self.RNG = np.random.RandomState(seed=rngSeed)
        
        assembly_list = titan.assembly

        self.frames = {}
        self.uq_dict = {}
        self.vehicle_mass = 0
        self.vehicle_mass = assembly_list[0].mass
        self.covariances = {}
        self.input_pointer = 0

        self.position_frame = None
        self.rotation_frame = None
        self.velocity_frame = None
        self.spin_frame = None

        # filepath should point to a .yaml file of a format as described in 'uncertainty_example.yaml'
        with open(filepath,'r') as file: self.uncertainty_data = yaml.load(file,Loader)
        object_list = []
        for assem in assembly_list: object_list.append([component.name.split('/')[-1] for component in assem.objects])
        self.instantiate_variables(object_list[0])

    def __delitem__(self, key):
        del self.uq_dict[key]

    def __getitem__(self, key):
        return self.uq_dict[key]
    
    def __iter__(self):
        return iter(self.uq_dict)
    
    def __len__(self):
        return len(self.uq_dict)
    
    def __setitem__(self,key):
        return self.uq_dict[key]
    
    def instantiate_variables(self, obj_names):
        special_cases = {'Position_vector' : self.make_x, 
                         'Rotation_vector' : self.make_theta, 
                         'Velocity_vector' : self.make_x_dot, 
                         'Spin_vector'     : self.make_theta_dot,
                         'Deorbit'         : self.deorbit,
                         'Covariances'      : self.make_cov}
         
        for key, value in self.uncertainty_data.items():
            if key in special_cases.keys(): special_cases[key](value) 
            elif key in obj_names: self.instantiate_object(key,value)
            else: 
                nominal = value['nominal'] if 'nominal' in value.keys() else None
                distribution = value['distribution'] if 'distribution' in value.keys() else None
                self.uq_dict[key.lower()] = self.uncertainVariable(position_pointer=self.input_pointer,nominal=nominal, distribution=distribution, rngState=self.RNG)
                self.input_pointer+=self.uq_dict[key.lower()].n

        self.input_dimensionality = np.sum([var.n for _, var in self.uq_dict.items()])
        
    def instantiate_object(self,name,obj_dict):
        object_variables = ['trigger_value','temperature']
        for key, value in obj_dict.items():
            if key.lower() in object_variables:
                nominal = value['nominal'] if 'nominal' in value.keys() else None
                distribution = value['distribution'] if 'distribution' in value.keys() else None
                self.uq_dict[name.lower()+'__'+key.lower()] = self.uncertainVariable(position_pointer=self.input_pointer,nominal=nominal, distribution=distribution, 
                                                                                     rngState=self.RNG, obj=name)
                self.input_pointer+=self.uq_dict[name.lower()+'__'+key.lower()].n

    def make_x(self, position_dict):
        self.position_frame = position_dict['frame']
        nominal = position_dict['nominal'] if 'nominal' in position_dict.keys() else None
        distribution = position_dict['distribution'] if 'distribution' in position_dict.keys() else None
        self.uq_dict['position_vector'] = self.uncertainVariable(position_pointer=self.input_pointer,nominal=nominal,distribution=distribution,rngState=self.RNG)
        self.input_pointer+=self.uq_dict['position_vector'].n
    
    def make_x_dot(self,velocity_dict):
        self.velocity_frame = velocity_dict['frame']
        nominal = velocity_dict['nominal'] if 'nominal' in velocity_dict.keys() else None
        distribution = velocity_dict['distribution'] if 'distribution' in velocity_dict.keys() else None
        self.uq_dict['velocity_vector'] = self.uncertainVariable(position_pointer=self.input_pointer,nominal=nominal,distribution=distribution,rngState=self.RNG)
        self.input_pointer+=self.uq_dict['velocity_vector'].n
    
    def make_theta(self,rotation_dict):
        self.rotation_frame = rotation_dict['frame']
        nominal = rotation_dict['nominal'] if 'nominal' in rotation_dict.keys() else None
        distribution = rotation_dict['distribution'] if 'distribution' in rotation_dict.keys() else None
        self.uq_dict['rotation_vector'] = self.uncertainVariable(position_pointer=self.input_pointer,nominal=nominal,distribution=distribution,rngState=self.RNG)
        self.input_pointer+=self.uq_dict['rotation_vector'].n
    
    def make_theta_dot(self,spin_dict):
        self.spin_frame = spin_dict['frame']
        nominal = spin_dict['nominal'] if 'nominal' in spin_dict.keys() else None
        distribution = spin_dict['distribution'] if 'distribution' in spin_dict.keys() else None
        self.uq_dict['spin_vector'] = self.uncertainVariable(position_pointer=self.input_pointer,nominal=nominal,distribution=distribution,rngState=self.RNG)
        self.input_pointer+=self.uq_dict['spin_vector'].n
    
    def sample_inputs(self, n_samples=1):
        input_vector = np.zeros((n_samples,self.input_dimensionality))

        for _, var in self.uq_dict.items():
            if var.cov_id == None: input_vector[:,var.position.astype(int)] = var.rvs(n_samples)
        for _, cov in self.covariances.items():
            input_vector[:,cov['positions'].astype(int)] = cov['distribution'].rvs(n_samples)
        return input_vector

    def communicate_sample(self,assembly,use_nominals=False):
        trajectory_mapping = {'altitude':'altitude','velocity':'velocity','flight_path_angle':'gamma','heading_angle':'chi',
                              'latitude':'latitude','longitude':'longitude'}
        assembly_mapping = {'mass':'mass','nose_radius':'','area_reference':'Aref','sideslip' : 'slip','angle_of_attack':'aoa',
                            'roll':'roll'}
        special_cases = {'position_vector' : self.update_position, 
                         'rotation_vector' : self.update_rotation, 
                         'velocity_vector' : self.update_velocity, 
                         'spin_vector'     : self.update_spin,
                         'deorbit'         : self.do_maneouvre,
                         'covariances'     : ''}
        
        if not use_nominals: input_vector = self.sample_inputs().flatten()
        else: 
            input_vector = np.zeros(self.input_dimensionality)
            for name, input_data in self.uq_dict.items(): input_vector[input_data.position] = input_data.nominal

        for name, input_data in self.uq_dict.items():
            obj_to_operate = assembly
            attr_to_operate = name
            if input_data.object_name is not None:
                obj_to_operate = assembly.objects[[component.name.split('/')[-1].lower() for component in assembly.objects].index(input_data.object_name.lower())]
                attr_to_operate = name.split('__')[-1].strip()
            elif name in trajectory_mapping.keys(): 
                obj_to_operate = assembly.trajectory
                attr_to_operate = trajectory_mapping[name]
            elif name in assembly_mapping.keys(): 
                attr_to_operate = assembly_mapping[name]

            data = input_vector[input_data.position]
            if len(data) == 1: data = np.array(data).flatten()[0]

            if name in special_cases.keys(): special_cases[name](obj_to_operate,data)
            else: setattr(obj_to_operate,attr_to_operate,data)

            return assembly

    def make_cov(self,cov_data):
        for cov_id,cov_dict in cov_data.items():
            cov = {}
            cov['positions'] = []
            cov['members'] =[]
            for member in cov_dict['members'].split(','):
                name = member.lower().strip()
                cov['members'].append(name)
                self.uq_dict[name].cov_id = cov_id
                cov['positions'] = np.hstack((cov['positions'],self.uq_dict[name].position))
            cov['dimensionality'] = len(cov['positions'])
            i_pointer = 0
            if not isinstance(cov_dict['distribution'],dict):
                cov['means'] = []
                cov['covariance'] = np.zeros([cov['dimensionality'],cov['dimensionality']])
                for member in cov['members']:
                    for i_dist, dist in enumerate(self.uq_dict[member].distribution):
                            mean_addition = dist.mean()
                            cov['means'] = np.hstack((cov['means'],mean_addition))
                            if isinstance(mean_addition,(float,np.float64)):
                                cov['covariance'][i_pointer,i_pointer] = dist.var()
                            else:
                                cov['covariance'][i_pointer:i_pointer+len(mean_addition),i_pointer:i_pointer+len(mean_addition)] = dist.cov
                            i_pointer+=1

                cov['covariance'] = cov_nearest(cov['covariance'])
                cov['distribution'] = multivariate_normal(cov['means'],cov=cov['covariance'],seed=self.RNG,allow_singular=True)
            else:
                for dist, args in cov_dict['distribution'].items():
                    cov['distribution'] = distri_list[dist](**args)
            if not 'is_dynamic' in cov_dict.keys(): cov['is_dynamic'] = False
            else: cov['is_dynamic'] = cov_dict['is_dynamic']
            cov['frame'] = cov_dict['frame'] if cov['is_dynamic'] else None
            self.covariances[cov_id] = cov

    def deorbit(self,deorbit_data):
        self.uq_dict['deorbit']=self.uncertainThruster(position_pointer=self.input_pointer,
                                                       Isp=deorbit_data['specific_impulse'],
                                                       m_prop=deorbit_data['propellant_mass'],
                                                       m_vehicle=self.vehicle_mass,
                                                       mp=deorbit_data['magnitude_proportional'],
                                                       mf=deorbit_data['magnitude_fixed'],
                                                       pp=deorbit_data['pointing_proportional'],
                                                       pf=deorbit_data['pointing_fixed'],
                                                       rngState=self.RNG)
        self.input_pointer+=self.uq_dict['deorbit'].n

    def update_position(self,assembly,options,data):
        if self.position_frame=='GEO':
            assembly.trajectory.latitude = data[0]
            assembly.trajectory.longitude = data[1]
            assembly.trajectory.altitude = data[2]
            compute_cartesian(assembly, options)
        elif self.position_frame=='ECEF': 
            ECEF = data
            assembly.position = ECEF
            # Get the new latitude, longitude and altitude
            [latitude, longitude, altitude] = ecef2geodetic(assembly.position[0], assembly.position[1], assembly.position[2],
                                        ell=Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                        deg = False);
            assembly.trajectory.latitude = latitude
            assembly.trajectory.longitude = longitude
            assembly.trajectory.altitude = altitude
        else: raise Exception('Error! Could not find position frame: {}'.format(self.position_frame))
        return assembly

    def update_rotation(self,assembly,options,data):
        if self.rotation_frame=='QUAT':
            assembly.quaternion = data
            R_NED_ECEF = R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
            #Should it be like this??
            R_B_NED_quat = (R_NED_ECEF).inv()*Rot.from_quat(assembly.quaternion)
            [yaw,pitch,roll] = R_B_NED_quat.as_euler('ZYX')
            assembly.yaw = yaw
            assembly.pitch = pitch
            assembly.roll = roll
        elif self.rotation_frame=='BODY':
            assembly.roll = data[0]
            assembly.pitch = data[1]
            assembly.yaw = data[2]
            compute_quaternion(assembly)
        else: raise Exception('Error! Could not find rotation frame: {}'.format(self.rotation_frame))
        return assembly

    def update_velocity(self,assembly,options,data):
        if self.velocity_frame=='WIND':
            assembly.trajectory.velocity = data[0]
            assembly.trajectory.chi = data[1]
            assembly.trajectory.gamma = data[2]
        elif self.velocity_frame=='NED':
            assembly.trajectory.velocity = np.linalg.norm(data)
            assembly.trajectory.chi = np.arctan2(data[1], data[0])
            assembly.trajectory.gamma = -np.arcsin(data[2] / assembly.trajectory.velocity)
        elif self.velocity_frame=='ECEF':
            R_NED=R_NED_ECEF(assembly.trajectory.latitude,assembly.trajectory.longitude)
            v = R_NED.apply(data,inverse=True)
            assembly.trajectory.velocity = np.linalg.norm(v)
            assembly.trajectory.chi = np.arctan2(v[1], v[0])
            assembly.trajectory.gamma = -np.arcsin(v[2] / assembly.trajectory.velocity)
        else: raise Exception('Error! Could not find velocity frame: {}'.format(self.velocity_frame))
        return assembly


    def update_spin(self,assembly,options,data):
        if not self.spin_frame=='BODY': raise Exception('Error! Could not find spin frame: {}'.format(self.spin_frame))
        assembly.roll_vel = data[0]
        assembly.pitch_vel = data[1]
        assembly.yaw_vel = data[2]
        return assembly


    def do_maneouvre(self,assembly,options,data):
        self.uq_dict['deorbit'].value = data
        self.uq_dict['deorbit'].burn(assembly.trajectory)
        return assembly.trajectory
        
    class uncertainVariable():
        # Each uncertain variable in the uncertainty handler is stored as an instance of this class
        # All variables represent n-dimenionsal vectors of floats
        def __init__(self,position_pointer,nominal=None,distribution=None,rngState=None, obj=None):
            # Nominal represents the "default" value that deterministic TITAN runs will see, will be selected as the mean if no value is provided
            # Distri is either a dict that constructs a scipy distribution or a scipy distribution itself
            self.rng = rngState
            self.is_multivariate=False
            self.has_distribution = True
            self.cov_id = None
            self.cov = []
            self.object_name = obj

            if isinstance(distribution,(rv_continuous,multi_rv_frozen)):
                self.distribution = copy(distribution)
                self.n = len(self.distribution.mean)
                if self.n>1: self.is_multivariate=True

            elif isinstance(distribution,dict):
                self.n = len(distribution)
                self.distribution = []
                    
                for dist, args in distribution.items():
                    if dist in distri_list: self.distribution.append(distri_list[dist](**args))
                    else: raise Exception('Error creating distribution {} with arguments {}'.format(dist, args))
                self.rvs()
                
            else: 
                self.distribution=nominal
                self.has_distribution=False
                try:
                    self.n = len(nominal)
                except: raise Exception('Error: Must provide a suitable nominal value if no distribution is provided!')
            
            self.position = np.arange(position_pointer,position_pointer+self.n)

            if nominal is not None: self.nominal=nominal
            else:
                self.nominal = []
                for distri in self.distribution: self.nominal.append(distri.mean)
                self.nominal = np.array(self.nominal).flatten()

        
        def rvs(self,n_samples=1):
            if not self.has_distribution: raise Exception('Error: No distribution for this variable!')
            if self.cov_id is None:
                self.value = []
                for distri in self.distribution: 
                    self.value.append(distri.rvs(n_samples,random_state=self.rng))
                self.value = np.reshape(self.value,[n_samples, -1])
                self.n = np.shape(self.value)[1]
                return self.value
        
    class uncertainThruster():
        # An uncertain thruster acts identically to an uncertain variable from a black box perspective but it can be called as it's own object to 
        # affect the vehicle's trajectory
        # Represented by the gates model of uncertain manoeuvres
        def __init__(self,position_pointer,Isp,m_prop,m_vehicle,mp,mf,pp,pf,rngState=None):
        # First calculate magnitude and direction (retrograde) of deorbit burn
            self.rng = rngState
            self.is_multivariate=True
            self.has_distribution = True
            self.cov_id = None
            self.object_name = None

            self.pf = pf
            self.pp = pp
            self.mf = mf
            self.mp = mp

            self.n = 3
            self.position = np.arange(position_pointer,position_pointer+self.n)
            
            self.exhaust_velocity = Isp*9.80665
            self.m_vehicle = m_vehicle
            self.m_prop_original = m_prop
            self.m_prop = m_prop
            mass_ratio = (m_prop + m_vehicle)/m_vehicle
            
            self.compute_thruster(mass_ratio)

            self.nominal = self.manoeuvre

        def compute_thruster(self,MR,manouevre_type='retrograde'):
            # Tsiolkovsky rocket eq...
            delta_v = self.exhaust_velocity * np.log(MR)
            if manouevre_type=='retrograde':
                self.manoeuvre = [-delta_v, 0, 0]
            elif manouevre_type=='prograde':
                self.manoeuvre = [delta_v, 0, 0]
            elif manouevre_type=='radial_in':
                self.manoeuvre = [0, 0, delta_v]
            elif manouevre_type=='radial_out':
                self.manoeuvre = [0, 0, -delta_v]
            elif manouevre_type=='normal':
                self.manoeuvre = [0, delta_v, 0]
            elif manouevre_type=='anti_normal':
                self.manoeuvre = [0, -delta_v, 0]
            else:
                self.manoeuvre = delta_v*manouevre_type

            dx = norm(loc=0, scale=np.sqrt(self.mf ** 2 + self.mp ** 2 * delta_v))
            dy = norm(loc=0, scale=np.sqrt(self.pf ** 2 + self.pp ** 2 * delta_v))
            dz = norm(loc=0, scale=np.sqrt(self.pf ** 2 + self.pp ** 2 * delta_v))

            self.distribution = [dx,dy,dz]

        def rvs(self,n_samples=1):
            perturbations = np.reshape([dim.rvs(n_samples) for dim in self.distribution],[-1,3])
            self.value = np.array([self.manoeuvre for _ in range(n_samples)]) + perturbations
            return self.value
        
        def burn(self,trajectory,propellant_fraction=1.0,manouevre_type='retrograde',use_value=True):
            # propellant_fraction = 1.0 => burn all fuel, = 0.0 => burn nothing, note very low values will have unphysical perturbation effects
            # i.e. this model can only be used for impulsive burns
            if self.m_prop<=0: print('Attempted to perform a burn with zero available propellant!')
            else:
                prop_to_expend = propellant_fraction*self.m_prop_original
                if prop_to_expend>self.m_prop: 
                    print('Not enough propellant to satisfy specification, using what is available...')
                    prop_to_expend = self.m_prop
                mass_ratio = (self.m_prop + self.m_vehicle)/(self.m_vehicle+self.m_prop-prop_to_expend)
                self.compute_thruster(mass_ratio,manouevre_type=manouevre_type)
                burn = self.value if use_value else self.rvs()
                trajectory = apply_velocity_wind(trajectory, burn)
                self.m_prop-=prop_to_expend

def extractQoI(cfg,csvfile):

    output_array = []
    # Load data and assembly files...
    assem_csv = csvfile.rsplit('.', maxsplit=1)[0] + '_assembly.csv'
    with open(csvfile, 'r') as file:
        data = pd.read_csv(file)
    with open(assem_csv, 'r') as file:
        assembly_data = pd.read_csv(file)

    output_names = [name.strip() for name in cfg['QoI']['Outputs'].split(',')]

    object_names = [cfg['Assembly']['Path'] + name.strip() for name in cfg['QoI']['Objects'].split(',')]

    # Clip and index dataseries to enhance performance of searching...
    assembly_data = assembly_data[['Assembly_ID', 'Obj_name']].set_index(['Obj_name'])
    clipped_data = data[['Iter', 'Assembly_ID'] + output_names].set_index(['Assembly_ID'])

    # Iterate over each object
    for name in object_names:

        # Find object's latest assembly id
        assem_id = assembly_data.loc[name].loc[:, 'Assembly_ID'].iloc[-1]
        # Find latest instance of that assembly id
        final_data = clipped_data.loc[assem_id].iloc[-1]

        # Iterate over each output
        for output in output_names:

            if output.strip() == 'Time':  # Simple linear extrapolation to find time at h=0

                index = int(final_data.loc['Iter'])
                # TODO lerp other values of interest (a huge pain for lat and lon)
                vel_z = (data.loc[:, 'Velocity'][index] *
                         np.sin(np.radians(data.loc[:, 'FlighPathAngle'][index])))
                h = data.loc[:, 'Altitude'][index]
                delta_t = -h / vel_z
                time = data.loc[:, 'Time'][index] + delta_t
                output_array.append(time)

            else:  # Otherwise just report value
                output_array.append(final_data.loc[output])

    return output_array


def write_demise(assembly,demise_object,options):

    lookup={'Assembly_ID':assembly.id,'Mass':assembly.mass,'Altitude':assembly.trajectory.altitude,'Velocity':assembly.trajectory.velocity,
        'FlighPathAngle':assembly.trajectory.gamma*180/np.pi,'HeadingAngle':assembly.trajectory.chi*180/np.pi,
        'Latitude':assembly.trajectory.latitude*180/np.pi,'Longitude': assembly.trajectory.longitude*180/np.pi,'AngleAttack': assembly.aoa*180/np.pi,
        'AngleSideslip': assembly.slip*180/np.pi,'ECEF_X': assembly.position[0],'ECEF_Y': assembly.position[1],'ECEF_Z': assembly.position[2],
        'ECEF_vU':assembly.velocity[0],'ECEF_vV':assembly.velocity[1],'ECEF_vW':assembly.velocity[2],'BODY_COM_X':assembly.COG[0],
        'BODY_COM_Y':assembly.COG[1],'BODY_COM_Z':assembly.COG[2],'Aero_Fx_B':assembly.body_force.force[0],'Aero_Fy_B':assembly.body_force.force[1],
        'Aero_Fz_B':assembly.body_force.force[2],'Aero_Mx_B':assembly.body_force.moment[0],'Aero_My_B':assembly.body_force.moment[1],
        'Aero_Mz_B':assembly.body_force.moment[2],'Lift': assembly.wind_force.lift,'Drag': assembly.wind_force.drag,
        'Crosswind': assembly.wind_force.crosswind,'Mass':assembly.mass,'Inertia_xx':assembly.inertia[0,0],'Inertia_xy':assembly.inertia[0,1],
        'Inertia_xz':assembly.inertia[0,2],'Inertia_yy':assembly.inertia[1,1],'Inertia_yz':assembly.inertia[1,2],'Inertia_zz':assembly.inertia[2,2],
        'Roll': assembly.roll*180/np.pi,'Pitch': assembly.pitch*180/np.pi,'Yaw':  assembly.yaw*180/np.pi,'VelRoll': assembly.roll_vel,
        'VelPitch':assembly.pitch_vel,'VelYaw': assembly.yaw_vel,'Quat_w': assembly.quaternion[3],'Quat_x': assembly.quaternion[0],
        'Quat_y': assembly.quaternion[1],'Quat_z': assembly.quaternion[2],'Mach':assembly.freestream.mach,
        'Density':assembly.freestream.density,'Temperature':assembly.freestream.temperature,'Pressure':assembly.freestream.pressure,
        'SpecificHeatRatio':assembly.freestream.gamma,"Aref":assembly.Aref,"Lref":assembly.Lref}
    
    try:
        lookup['Qstag'] = [assembly.aerothermo.qconvstag]
        lookup['Qradstag'] = [assembly.aerothermo.qradstag]
    except:
        pass
    try:
        lookup['Speedsound']=[assembly.freestream.sound]
        lookup['Pstag'] = [assembly.freestream.P1_s]
        lookup['Tstag'] = [assembly.freestream.T1_s]
        lookup['Rhostag'] = [assembly.freestream.rho_s]
    except:
        pass
    # pct_mass =assembly.freestream.percent_mass if options.freestream.method == "Mutationpp" else assembly.freestream.percent_mass[0]
    # for specie, pct in zip(assembly.freestream.species_index, pct_mass) :
    #     lookup[specie+"_mass_pct"] = [pct]
    proc = ''.join(character for character in options.output_folder if character.isdigit())
    proc=int(proc) if len(proc)>0 else 0

    with open(options.uncertainty.qoi_filepath,'rb') as file: options.uncertainty.quantities=pickle.load(file)

    stl = demise_object.name
    try:
        msg=messenger(rank=proc)
        msg.read_data()
        msg.print_n_send('Object of interest \'{}\' has demised at altitude {}'.format(stl,assembly.trajectory.altitude))
    except: pass
    for output_name, _ in options.uncertainty.quantities[stl].items():
        options.uncertainty.quantities[stl][output_name].append(lookup[output_name])

    with open(options.uncertainty.qoi_filepath,'wb') as file: pickle.dump(options.uncertainty.quantities,file)

def plot_demise(input_quantities,outputs, names):
    figs = []
    axes = []
    i_plot = 0
    quantities = {}
    for obj, val in input_quantities.items(): 
        for name in names:
            if name in obj: quantities[obj] = val
    plt.style.use('dark_background')

    toPlot = outputs
    
    if 'Latitude' in outputs and 'Longitude' in outputs and 'Altitude' in toPlot:
        [toPlot.remove(val) for val in ['Latitude','Longitude','Altitude']]
        figs.append(plt.figure())
        axes.append(figs[i_plot].add_subplot(projection = '3d',computed_zorder=False))
        axes[i_plot].set_zlim(bottom=0)
        axes[i_plot].set_xlabel('East/West Distance (m)')
        axes[i_plot].set_ylabel('North/South Distance (m)')
        axes[i_plot].set_zlabel('Altitude (m)')
        i_obj = 0
        edgespace = 0.05
        minlat=np.min([np.min(data['Latitude']) for obj, data in quantities.items()])
        minlat-=edgespace*minlat
        minlon=np.min([np.min(data['Longitude']) for obj, data in quantities.items()])
        minlon-=edgespace*minlon
        maxlat=np.max([np.max(data['Latitude']) for obj, data in quantities.items()])
        maxlat+=edgespace*maxlat
        maxlon=np.max([np.max(data['Longitude']) for obj, data in quantities.items()])
        maxlon+=edgespace*maxlon
        maxalt = np.max([np.max(data['Altitude']) for obj, data in quantities.items()])
        delta = np.max([maxlat-minlat, maxlon - minlon])
        if maxalt<=0: maxalt=0.01
        maxalt+=edgespace*maxalt

        directory=os.getcwd()

        m = Basemap(projection='merc',
        llcrnrlat=minlat,urcrnrlat=maxlat,
        llcrnrlon=minlon,urcrnrlon=maxlon,
        resolution='h',area_thresh=10, ellps = 'WGS84',suppress_ticks=False,
        lat_0=0.5*(maxlat-minlat),lon_0=0.5*(maxlon-minlon),fix_aspect=False)
        try:
            elevation.clip(bounds=(minlon, minlat, maxlon, maxlat), output=directory+'/plotSurface.tif',product = 'SRTM3')
            with rasterio.open('plotSurface.tif') as dem:
                z = dem.read(1)
            
            nrows, ncols = z.shape
            x = np.linspace(minlon, maxlon, ncols)
            y = np.linspace(maxlat, minlat, nrows)
            x, y = np.meshgrid(x, y)
            x,y = m(x,y)
            surf=axes[i_plot].plot_surface(x,y,np.clip(z,a_min=-1,a_max=None),facecolors=None,shade=False,cmap=map,alpha=1,zorder=1,vmin=1e-6)
            surf.set_edgecolors(surf.to_rgba(surf._A))
            surf.set_facecolor([0,0,0,0])
        except Exception as e:
            print('Could not get terrain data: %s'%e)
            polys = PolyCollection([polygon.get_coords() for polygon in m.landpolygons],facecolor = [0,0.35,0],closed=False)
            axes[i_plot].add_collection3d(polys)

        maxx,maxy = m(maxlon,maxlat)
        minx,miny = m(minlon,minlat)

        map = cm.get_cmap('pink').copy()
        map.set_under(color=[0.05,0,0.25])

        
        axes[i_plot].add_collection3d(m.drawcoastlines(linewidth=0.5,color='w',zorder=3))
        axes[i_plot].add_collection3d(m.drawcountries(linewidth=0.25,color='0.8',zorder=2))
        
        for obj, data in quantities.items():
            obj_x,obj_y = m(data['Longitude'],data['Latitude'])
            axes[i_plot].scatter(obj_x,obj_y,np.clip(data['Altitude'],a_min=0,a_max=None),label = names[i_obj],zorder=10)
            i_obj+=1
        axes[i_plot].legend()
        axes[i_plot].set_xlim3d([minx,maxx])
        axes[i_plot].set_ylim3d([miny, maxy])
        axes[i_plot].set_zlim3d([0, maxalt])
        axes[i_plot].xaxis.set_pane_color([0,0,0])
        axes[i_plot].yaxis.set_pane_color([0,0,0])
        axes[i_plot].zaxis.set_pane_color([0,0,0])
        axes[i_plot].set_box_aspect((maxlon-minlon,maxlat-minlat,1e-3*maxalt))
        figs[0].suptitle('Demise points centred at {}, {} degrees Lat/Lon'.format(0.5*(maxlat+minlat),0.5*(maxlon+minlon)))
        i_plot+=1
    while len(toPlot)>0:
        if len(toPlot)>4 or len(toPlot)==3:
            figs.append(plt.figure())
            ax3d = figs[i_plot].add_subplot(projection='3d')
            axes.append(ax3d)
            axes[i_plot].set_xlabel(toPlot[0])
            axes[i_plot].set_ylabel(toPlot[1])
            axes[i_plot].set_zlabel(toPlot[2])
            i_obj = 0
            for obj, data in quantities.items(): 
                axes[i_plot].scatter(data[toPlot[0]],data[toPlot[1]],data[toPlot[2]],label= names[i_obj],marker='x')
                i_obj+=1
            [toPlot.pop(0) for i in range(3)]
            axes[i_plot].legend()
        elif len(toPlot)>1:
            figs.append(plt.figure())
            axes.append(figs[i_plot].add_subplot())
            axes[i_plot].set_xlabel(outputs[0])
            axes[i_plot].set_ylabel(outputs[1])
            i_obj = 0
            for obj, data in quantities.items(): 
                axes[i_plot].scatter(data[outputs[0]],data[outputs[1]],label = names[i_obj],marker='x')
                i_obj+=1
            [toPlot.pop(0) for i in range(2)]
            axes[i_plot].legend()
        i_plot+=1

    plt.show()
