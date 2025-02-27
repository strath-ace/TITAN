from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import concurrent.futures, psutil
from Dynamics.euler import compute_Euler
from Dynamics.dynamics import compute_quaternion
from Dynamics import frames
from Dynamics.propagation import quaternion_mult, quaternion_normalize, compute_jacobian_diagonal
import pandas as pd
import os
import numpy as np
from collections.abc import MutableSequence
from copy import deepcopy
import pymap3d
from scipy.linalg import sqrtm, cholesky
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from Dynamics.frames import R_NED_ECEF
from warnings import warn
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
from statsmodels.stats.correlation_tools import cov_nearest
from copy import copy

params_names =['quaternion','trajectory','yaw','pitch','roll','aoa','slip','roll_vel','pitch_vel','yaw_vel','velocity']
n_assem = 0

class dynamicDistribution():
    
    def __init__(self, assembly, mean_states, cov, RNG = np.random.RandomState(42069), is_quaternionic=False, conversion_samples = 10000, DOF = 6, distribution = multivariate_normal):
        self.mean = mean_states
        self.cov = cov
        self.R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
        self.DOF = DOF
        self.n = 13 if self.DOF==6 else 6
        if self.DOF==6 and not is_quaternionic: self.mean, self.cov = self.propagate_to_quaternion(mean_states, cov, conversion_samples)
        self.RNG = RNG
        self.distri = distribution(self.mean,self.cov,seed=self.RNG, allow_singular = True)


    def propagate_to_quaternion(self,mean_states,cov, n_samples): # Sometimes Monte Carlo really is the best way
        distri_euler_angles = multivariate_normal(mean_states,cov, allow_singular = True)
        states_euler_angles = distri_euler_angles.rvs(n_samples)
        set_of_states = np.zeros((n_samples,13))

        for i_state, euler_state in enumerate(states_euler_angles):
            set_of_states[i_state,:6] = euler_state[:6]
            set_of_states[i_state,-3:] = euler_state[-3:]
            R_B_NED =   frames.R_B_NED(roll = euler_state[6], pitch = euler_state[7], yaw = euler_state[8]) 
            q = quaternion_normalize((self.R_NED_ECEF*R_B_NED).as_quat())
            set_of_states[i_state,6:10] = q
        mean = np.mean(set_of_states, axis=0)
        cov = np.cov(set_of_states, rowvar=False)

        return mean, np.diag(np.diag(cov))
    
    def rvs(self, N = 1, RNG = None):
        if RNG is None: RNG = self.RNG
        return self.distri.rvs(N, RNG)

class recursive_gaussan_mixture():
    def __init__(self, mean, cov, weight = 1.0, is_leaf = True, is_root = False, library_size = 3, tree_size = 1, rng = np.random.RandomState(seed=42069)):
        self.mean = np.array(mean)
        self.mean_prev1 = None
        self.mean_prev2 = None
        self.d_dt_mean_prev1 = None
        self.d_dt_mean_prev2 = None
        self.cov = cov
        self.n_leaf_nodes = tree_size
        self.rng = rng
        self.leaf_list = [] if not is_leaf else [self]
        self.is_leaf = is_leaf
        self.library_size = library_size
        self.weight = weight
        self.is_root = is_root
        self.shannon_entropy = []
        self.dynamical_entropy = None
        self.distribution = None
        self.children = []
        self.GSR = []

        # Standard splitting libraries with 3 and 5 from DeMars et al, doi.org/10.2514/1.58987
        # 4 element library from Huber et al, doi.org/10.1109/MFI.2008.4648062
        self.libraries = {3:{'weights' : np.array([ 0.2252246249,  0.5495507502,  0.2252246249]),
                             'means'   : np.array([-1.0575154615,  0.0,           1.0575154615]),
                             'std'     :   0.6715662887},
                          4:{'weights' : np.array([ 0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098]),
                             'means'   : np.array([-1.4131205233, -0.44973059608, 0.44973059608, 1.4131205233]),
                             'std'     :   0.51751260421},
                          5:{'weights' : np.array([ 0.0763216491,  0.2474417860,  0.3524731300,  0.2474417860, 0.0763216491]),
                             'means'  :  np.array([-1.6899729111, -0.8009283834,  0.0,           0.8009283834, 1.6899729111]),
                             'std'    :   0.4422555386}}
        if not is_leaf:
            self.split()
        else: self.update_distribution()

    def get_mixture_parameters(self):
        # Split component along principal axis of covariance hyper-ellipsoid
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)
        split_axis = np.argmax(self.eigenvalues)
        weights = self.weight * self.libraries[self.library_size]['weights']
        means = []
        covs = []
        for i_component in range(self.library_size):
            component_eigenvalues= self.eigenvalues
            component_eigenvalues[split_axis] *= (self.libraries[self.library_size]['std'])**2
            covs.append(self.eigenvectors @ np.diag(self.eigenvalues) @ self.eigenvectors.transpose())
            means.append(self.mean + np.sqrt(self.eigenvalues[split_axis]) * self.libraries[self.library_size]['means'][i_component] * self.eigenvectors[:,split_axis])
        return weights, means, covs
    
    def get_shannon_entropy_change(self):
        #self.shannon_entropy.append(0.5 * np.log(np.linalg.det(2 * np.pi * np.e * self.cov)))
        n = len(self.mean)
        #self.shannon_entropy.append(0.5 * n * np.log(2 * np.pi * np.e * np.linalg.det(self.cov)**(1/n)))
        self.shannon_entropy.append(0.5 * n * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.cov)) + n/2)
        dH = self.shannon_entropy[-1] - self.shannon_entropy[-2] if len(self.shannon_entropy)>1 else None
        return dH
    
    def recalculate_dynamical_entropy(self,sigma_points, sigma_derivatives):
        jac = compute_jacobian_diagonal(sigma_points,sigma_derivatives)
        self.dynamical_entropy = -1*np.trace(jac)
    
    def update_distribution(self):
        self.distribution = multivariate_normal(self.mean,self.cov, allow_singular = True)

    def rvs(self,n=1):
        if self.is_leaf: 
            result = self.distribution.rvs(n)
        else:
            result = np.zeros([n,len(self.mean)])
            for child in self.children:
                result += child.rvs(n)
            # if self.is_root: # Rescale to true distri
            #     result *= self.cov
            #     result += self.mean
        print(result)
        return result

    def get_leaf_by_index(self,index):
        if index < len(self.leaf_list): return self.leaf_list[index]
        i_leaf = 0
        while i_leaf <= index:
            if len(self.children)<1: return self
            for child in self.children:
                if child.is_leaf: leaf = child
                else: leaf = child.get_leaf_by_index(i_leaf) 
                if i_leaf==index: return leaf
                i_leaf += 1

    def build_leaf_list(self, recursive = False):
        self.n_leaf_nodes = 0
        if self.is_leaf: 
            self.leaf_list = [self]
            self.n_leaf_nodes = 1
        else:
            self.leaf_list = []
            for child in self.children: 
                if recursive: child.build_leaf_list(recursive=True)
                self.n_leaf_nodes += child.n_leaf_nodes

            for i_leaf in range(self.n_leaf_nodes):
                self.leaf_list.append(self.get_leaf_by_index(i_leaf))



    def split_self(self):
        # This node is no longer a leaf node
        self.is_leaf = False
        self.n_leaf_nodes -= 1

        self.n_leaf_nodes += self.library_size
        self.children = []
        weights, means, covs = self.get_mixture_parameters()
        for i_child in range(self.library_size):
            self.children.append(recursive_gaussan_mixture(mean         =  means[i_child],
                                                           cov          =  covs[i_child],
                                                           weight       =  weights[i_child],
                                                           is_leaf      =  True,
                                                           is_root      =  False,
                                                           library_size =  self.library_size,
                                                           tree_size    =  1,
                                                           rng          =  self.rng))
    
    def split_leaf(self):
        max_uncertainty = 0.0
        split_candidate = None
        for leaf in self.leaf_list:
            if np.max(np.linalg.eigh(leaf.cov)[0]) > max_uncertainty:
                max_uncertainty = np.max(np.linalg.eigh(leaf.cov)[0])
                split_candidate = leaf
        split_candidate.split_self()
        self.build_leaf_list(recursive=True)
    
    def recalculate_mean(self):
        if self.is_leaf: return self.weight*self.mean
        else:
            mean = np.zeros_like(self.mean)
            for child in self.children: mean += child.recalculate_mean()
            return mean


def create_distribution(assembly,options, mean= None, cov = None, is_Library=True):
    if mean is None and cov is None:
        with open(options.uncertainty.yaml_path,'r') as file: dynamic_distri_data = yaml.load(file,Loader)['Covariances']['dynamic']
        #Get our means and cov...
        try: 
            mean = dynamic_distri_data['distribution']['multivariate_normal']['mean']
            cov  = dynamic_distri_data['distribution']['multivariate_normal']['cov']
        except:
            try: 
                mean = dynamic_distri_data['distribution']['mean']
                cov = dynamic_distri_data['distribution']['cov']
            except Exception as e: 
                raise Exception('Error generating dynamical distribution! Perhaps your yaml file is not set up correctly. Error: {}'.format(e))
    DOF = 3 if len(mean)<12 else 6
    is_quaternionic = False if len(mean)<13 else True
    distri =  dynamicDistribution(assembly,mean,cov,is_quaternionic=is_quaternionic,DOF=DOF)
    if not is_Library: return distri
    assembly.gaussian_library = recursive_gaussan_mixture(mean = distri.mean,
                                                          cov = distri.cov,
                                                          is_leaf=True,
                                                          is_root=True,
                                                          library_size=options.uncertainty.GMM_library)
    if options.uncertainty.use_GMM:
        for _ in range(options.uncertainty.GMM_a_priori_splits): assembly.gaussian_library.split_leaf()
    return assembly.gaussian_library
class sigmaPosition(MutableSequence):

    def __init__(self, dims=6,alpha=0.5,beta=2,kappa=0,position=[0,0,0],cov=[[1,0,0],[0,1,0],[0,0,1]]):
        self.n=dims
        self.points=MerweScaledSigmaPoints(n=self.n, alpha=alpha, beta=beta, kappa=kappa)
        self.position=position
        self.mu=position
        self.cov=cov
        self.sqrtcov = sqrtm(cov)
        self.regenerate_sigmas()
        self.params = [[] for i in range(2*self.n+1)]
        self.params_names = params_names
        self.extras=[]
        
        if dims>=6:
            self.extras.append('velocity')
        if dims>=12:
            self.extras.append('roll')
            self.extras.append('pitch')
            self.extras.append('yaw')
            self.extras.append('roll_vel')
            self.extras.append('pitch_vel')
            self.extras.append('yaw_vel')


    def __getitem__(self,i):
        return self.position[i]
    
    def __setitem__(self,i,value):
        self.position[i]=value

    def __len__(self):
        return len(self.position)
    
    def __delitem__(self,i):
        del self.list[i] 

    def insert(self,i,item):
        self.position.insert(i,item)

    def regenerate_sigmas(self,mu='',cov=''):
        try:
            mu=self.mu if mu=='' else mu
            cov=self.cov if cov=='' else cov
            self.sigmas = self.points.sigma_points(mu,cov)
        except:
            pass

    def UT(self):
        self.mu, self.cov = unscented_transform(sigmas=self.sigmas,Wm=self.points.Wm,Wc=self.points.Wc)
        # try: cholesky(self.cov)
        # except:
        #     print('Covariance is not PSD! Punching it until it is!')
        self.cov = cov_nearest(self.cov)
        self.regenerate_sigmas()

    def select_point(self,i):
        self.position = self.sigmas[i].copy()[:3]

        start_pointer = 3
        for also in self.extras:
            param = self.params[i][params_names.index(also)]
            try: param_len = len(param)
            except: param_len=1
            if param_len==1: 
                self.params[i][params_names.index(also)] = self.sigmas[i].copy()[start_pointer]
            else: self.params[i][params_names.index(also)]=self.sigmas[i].copy()[start_pointer:start_pointer+param_len]
            start_pointer+=param_len
    
    def get_trajectory(self,i):
        return deepcopy(self.trajectories[i])

    def adjust_dof(self,i_point=0):
        for also in self.extras:
            param = self.params[i_point][params_names.index(also)]
            self.mu = np.hstack((self.mu,param))
        self.regenerate_sigmas()

    def retrieve_point(self,i):
        sigma_point = self.position

        for also in self.extras:
            param = self.params[i][params_names.index(also)]
            sigma_point = np.hstack((sigma_point,param)).flatten()
        return sigma_point
    
    def write_data(self,options,assem_ID,time):
        data_sp = []
        data_cov = sqrtm(self.cov).flatten()
        columns=['Time']
        ax_labels = ['_dim_'+str(i_ax) for i_ax in range(self.n)]
        ax_labels[0] = '_lat'
        ax_labels[1] = '_lon'
        ax_labels[2] = '_alt'
            
        
        for i_point in range(2*options.uncertainty.ut_DoF+1):
            [columns.append('s'+str(i_point)+ax_label) for ax_label in ax_labels]

            geo = pymap3d.ecef2geodetic(self.sigmas[i_point][0], self.sigmas[i_point][1], self.sigmas[i_point][2],
                                    ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                    deg = False)
            # R_ned = R_NED_ECEF(lat =geo[0],lon=geo[1])
            # #if i_point==0: data_cov = R_ned.as_matrix()*self.cov*(R_ned.as_matrix().transpose())
            # ned = R_ned.apply(self.sigmas[i_point],inverse=True)
            # for iax, ax in enumerate(['north','east','down']):
            # #for iax, ax in enumerate(['lat','lon','alt']):
            # #for iax, ax in enumerate(['x','y','z']):
            #     columns.append('sigma_'+ax+'_'+str(i_point))
            #     data_sp.append(ned[iax])#self.sigmas[i_point][iax])
            if len(data_sp)==0: data_sp = np.hstack((geo,self.sigmas[i_point][3:]))
            else: data_sp = np.hstack((data_sp,geo,self.sigmas[i_point][3:]))
        
        [columns.append('Cov_'+str(i)) for i in range(len(data_cov))]
        data_array = np.real(np.hstack((time,data_sp,data_cov)).reshape((1,-1)))
        data=pd.DataFrame(data_array,columns=columns)
        doHeader = False if os.path.exists(options.output_folder+'/utstats_{}.csv'.format(assem_ID)) else True
        data.to_csv(options.output_folder+'/utstats_{}.csv'.format(assem_ID),mode='a',index=False,header=doHeader)

    def communicate_params(self,i_point,params_objects):
         for i_param, param in enumerate(self.params[i_point]):
            params_objects[i_param] = deepcopy(param)

    def build_params(self,i_point,params_objects):
        for i_param, param in enumerate(params_objects):
            self.params[i_point].append(deepcopy(param))
    
    def update_params(self,i_point,params_objects):
        for i_param, param in enumerate(params_objects):
            self.params[i_point][i_param]=deepcopy(param)
        return self.params
    
    def retrieve_param(self, i_point, i_param):
        return deepcopy(self.params[i_point][i_param])

def setupUT(options):
    from Uncertainty.uncertainty import uncertaintySupervisor
    cov = None
    if options.uncertainty.cov_UT == True:
        with open(options.uncertainty.yaml_path,'r') as file: uncertainty_db = yaml.load(file,Loader)
        for key, value in uncertainty_db.items():
            if key=='UT_covariance':
                cov = value
            if key=='UT_vector':
                position = value
        if cov is None: raise Exception('Error: could not find \'UT_covariance\' entry in .yaml file!')
        options.cov=np.array(cov)
        return options

    UQ = uncertaintySupervisor(isActive=1)
    UQ.constructInputs(options.uncertainty.yaml_path)
    position = UQ.inputs.get_statistics('mean')
    sigma = UQ.inputs.get_statistics('std')

    if not len(position)==options.uncertainty.ut_DoF: raise Exception('Error: UT uncertainty could not be well constructed!')
    if options.uncertainty.ut_DoF ==3:
        [alt,lat,lon] = position
        [sigma_alt,sigma_lat,sigma_lon] = sigma
    elif options.uncertainty.ut_DoF ==6:
        [alt,lat,lon,velocity,fpa,ha] = position
        [sigma_alt,sigma_lat,sigma_lon,sigma_velocity,sigma_fpa,sigma_ha] = position
        velocity=velocity[0]
        fpa = fpa[0]
        ha = ha[0]
    elif options.uncertainty.ut_Dof == 12:
        raise Exception('12 DoF not yet implemented, sorry!')
    alt=alt[0]
    lat=lat[0]
    lon=lon[0]
    [x,y,z]= pymap3d.geodetic2ecef(lat=lat,
                        lon=lon,
                        alt=alt,
                        ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                        deg=True)
    [x_sig,y_sig,z_sig]= pymap3d.geodetic2ecef(lat=lat+sigma_lat[0],
                        lon=lon+sigma_lon[0],
                        alt=alt+sigma_alt[0],
                        ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                        deg=True)
    var_x = (x_sig-x)**2
    var_y = (y_sig-y)**2
    var_z = (z_sig-z)**2
    options.cov = np.diag([var_x,var_y,var_z])
    return options
    
def setupAssembly(assembly,options):
    assembly.position = sigmaPosition(dims=options.uncertainty.ut_DoF,position=assembly.position,cov=options.cov)
    params_objects = [getattr(assembly, name) for name in params_names]
    for i_point in range(2*assembly.position.n+1):
        assembly.position.build_params(i_point,params_objects)
    assembly.position.adjust_dof()
    return assembly



def wrapper(i_point,titan,options):
    for assem in titan.assembly:
        assem.position.select_point(i_point)
        for i_param, name in enumerate(params_names):
            setattr(assem,name, assem.position.retrieve_param(i_point,i_param))
        #assem.position.communicate_params(i_point,params_objects)
    if i_point==0:
        options.output_dynamics=True     
    else: options.output_dynamics=False

    compute_Euler(titan,options)
    sigmas =[]
    params = []
    
    for i_assem, assem in enumerate(titan.assembly):
        params_objects = [getattr(assem, name) for name in params_names]
        new_params = assem.position.update_params(i_point,params_objects)[i_point]
        sigmas.append(deepcopy(assem.position.retrieve_point(i_point)))
        params.append(deepcopy(new_params))

    return sigmas, params
    

def unscentedPropagation(titan, options):
    n_sigma = 2*options.uncertainty.ut_DoF+1
    n_procs=n_sigma if n_sigma<psutil.cpu_count(logical=False) else psutil.cpu_count(logical=False)

    with concurrent.futures.ProcessPoolExecutor(n_procs) as executor:
        output_futures = [executor.submit(wrapper,i_point,titan,options) for i_point in range(n_sigma)]

        # for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
        # #    print('Finished point: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/n_sigma,4))+'%)')
        #     pass
        concurrent.futures.wait(output_futures)
    for i_future, future in enumerate(output_futures):
        if future._exception:
            print('Error on result number {}: {}'.format(i_future,future.exception()))
        else:
            for i_assem, assem in enumerate(titan.assembly):
                assem.position.update_params(i_future,future.result()[1][i_assem])
                if i_future==0: 
                    assem.position.position = future.result()[0][i_assem]
                    #assem.position.communicate_params(i_future,params_objects)
                    for i_param, name in enumerate(params_names):
                        setattr(assem, name, assem.position.retrieve_param(i_future,i_param))
                assem.position.sigmas[i_future] = future.result()[0][i_assem]
    
    global n_assem
    if n_assem==0: n_assem=len(titan.assembly)
    if n_assem == len(titan.assembly):
        # yikes what an ugly hack, the UT double counts timesteps at fragmentation, counteract by only incrementing time in non-fragmentation cases
        titan.time+=options.dynamics.time_step

    for i_assem, assem in enumerate(titan.assembly):
        assem.position.UT()
        assem.position.write_data(options,assem.id,titan.time)
    n_assem = len(titan.assembly)

def UT_propagator(subpropagator, state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    ## Determnine the number of propagations to do (this can get very high)
    n_calcs = np.max([_assembly.gaussian_library.n_leaf_nodes for _assembly in titan.assembly])
    dim = len(state_vectors[0])
    point_generator = MerweScaledSigmaPoints(n=dim, alpha=options.uncertainty.UT_alpha, 
                                             beta=options.uncertainty.UT_beta, kappa=options.uncertainty.UT_kappa)
    # Everything in this script is wrapped in this i_calc iterator which is called for each Gaussian in the biggest gaussian library
    for i_calc in range(n_calcs):
        
        ##  First collect our big old list of points
        sigmas_per_assembly = []
        distri_per_assembly = []
        for i_assem, _assembly in enumerate(titan.assembly): 
            if _assembly.gaussian_library.n_leaf_nodes<=i_calc:
                _assembly.compute = False
                sigmas_per_assembly.append(np.zeros(2*dim+1,len(state_vectors_prior[i_assem])))
                distri_per_assembly.append(None)
            else:
                _assembly.compute = True
                distri_per_assembly.append(_assembly.gaussian_library.get_leaf_by_index(i_calc))        
                sigmas_per_assembly.append(point_generator.sigma_points(distri_per_assembly[i_assem].mean,
                                                                        distri_per_assembly[i_assem].cov))
                
                distri_per_assembly[i_assem].mean_prev2 = copy(distri_per_assembly[i_assem].mean_prev1)
                distri_per_assembly[i_assem].mean_prev1 = copy(distri_per_assembly[i_assem].mean)
                distri_per_assembly[i_assem].d_dt_mean_prev2 = copy(distri_per_assembly[i_assem].d_dt_mean_prev1)
        
        ### Then propagate our means
        print('Propagating means of distribution {}'.format(i_calc+1))
        mean_vectors_per_assembly = [sigmas_per_assembly[i_assem][0,:] for i_assem, assem in enumerate(titan.assembly)]
        new_state, new_dt = subpropagator(mean_vectors_per_assembly,None,None,dt,titan,options)
        if isinstance(new_state,np.ndarray): new_state = new_state.tolist()
        if isinstance(new_dt,np.ndarray): new_dt = new_dt.tolist()
        ## And assign to distris
        sigma_point_iterable = []
        for i_assem, _assembly in enumerate(titan.assembly):
            if _assembly.compute:
                distri_per_assembly[i_assem].d_dt_mean_prev1 = new_dt[i_assem]
                distri_per_assembly[i_assem].mean = new_state[i_assem]
                new_state[i_assem] = [new_state[i_assem]]
                new_dt[i_assem] = [new_dt[i_assem]]
        

        ## Then we need to reformat our sigma points so they are parallelisable
        for i_sigma in range(2*dim+1):
            sigma_point_iterable.append([])
            for i_assem, _assembly in enumerate(titan.assembly): sigma_point_iterable[i_sigma].append(sigmas_per_assembly[i_assem][i_sigma,:])
    
        ## Before propagating all other sigma points for this distri
        print('Propagating remaining sigma points ({}) of distribution {}'.format(len(sigma_point_iterable)-1,i_calc+1))
        if options.uncertainty.n_procs>1:
            with concurrent.futures.ProcessPoolExecutor(16) as executor:
                output_futures = [executor.submit(subpropagator,sigma_point,None,None,dt,titan,options) for sigma_point in sigma_point_iterable[1:]]
                concurrent.futures.wait(output_futures)
        
            for i_future, future in enumerate(output_futures):
                if future._exception:
                    raise Exception('Error on result number {}: {}'.format(i_future,future.exception()))
                else:
                    for i_assem, _assembly in enumerate(titan.assembly):
                        new_state[i_assem] = np.vstack((new_state[i_assem],future.result()[0][i_assem]))
                        new_dt[i_assem] = np.vstack((new_dt[i_assem],future.result()[1][i_assem]))
        else:
            for sigma_point in sigma_point_iterable[1:]:

                sigma_state, sigma_dt = subpropagator(sigma_point, None, None, dt, titan, options)
                for i_assem, _assembly in enumerate(titan.assembly):
                    new_state[i_assem] = np.vstack((new_state[i_assem],sigma_state[i_assem]))
                    new_dt[i_assem] = np.vstack((new_dt[i_assem],sigma_dt[i_assem]))


        ## Finally we can recombine our sigma points to get our new distribution
        for i_assem, _assembly in enumerate(titan.assembly):
            distri_per_assembly[i_assem].recalculate_dynamical_entropy(sigmas_per_assembly[i_assem],new_dt[i_assem])
            mu, cov = unscented_transform(sigmas=new_state[i_assem],Wm=point_generator.Wm, Wc=point_generator.Wc)
            distri_per_assembly[i_assem].mean = mu
            #distri_per_assembly[i_assem].cov = update_cov(titan, options, distri_per_assembly[i_assem], cov)
            distri_per_assembly[i_assem].cov = cov_nearest((reject_small_eigenvalues(cov,2)))
            
    output_vector = []
    for i_assem, _assembly in enumerate(titan.assembly):
        _assembly.gaussian_library.mean = _assembly.gaussian_library.recalculate_mean()
        output_vector.append(_assembly.gaussian_library.mean)
        #rvs = _assembly.gaussian_library.rvs(1500)
        # t = np.transpose([titan.time*np.ones(1500)])
        # columns = [['Time','X','Y','Z','U','V','W','Q_w','Q_i','Q_j','Q_k','Omega_roll','Omega_pitch','Omega_yaw']]
        # data_array = np.hstack((t,rvs))
        # write_to_series(data_array,columns,options.output_folder+'/UT_Assem_{}_rvs.csv'.format(_assembly.id,))
        for i_leaf in range(_assembly.gaussian_library.n_leaf_nodes):
            leaf = _assembly.gaussian_library.get_leaf_by_index(i_leaf)
            
            [latitude, longitude, altitude] = pymap3d.ecef2geodetic(leaf.mean[0], 
                                                                    leaf.mean[1], 
                                                                    leaf.mean[2],
                                                                    ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                                                    deg = True);
            data_array = np.hstack(([titan.time, altitude, latitude, longitude],leaf.mean,leaf.cov.flatten())).reshape(1,-1)
            columns = [['Time','Alt','Lat','Lon','X','Y','Z','U','V','W','Q_w','Q_i','Q_j','Q_k','Roll_omega','Pitch_omega','Yaw_omega']]

            for ax1 in     ['x','y','z','u','v','w','qw','qi','qj','qk','ox','oy','oz']:
                for ax2 in ['x','y','z','u','v','w','qw','qi','qj','qk','ox','oy','oz']:
                    columns[0].append('cov_{}/{}'.format(ax1,ax2))
            write_to_series(data_array,columns,options.output_folder+'/UT_Assem_{}_Gaussian_{}.csv'.format(_assembly.id,i_leaf))
    return output_vector, None

def update_cov(titan, options, distri, input_cov):
    grandsum_ratio = np.sum(input_cov,(0,1))/np.sum(distri.cov,(0,1))
    new_cov  = cov_nearest(abs(input_cov))# + np.eye(np.shape(cov)[0])*1e-9
    if len(distri.GSR)>2:
        delta_sum = [distri.GSR[-i-1]-distri.GSR[-i-2] for i in range(len(distri.GSR) - 1)]
        new_dS = delta_sum[0]
        print(delta_sum)
        comparison_dS = np.array(delta_sum[1:])[np.nonzero(delta_sum[1:])][0]
        print('Delta sum factor is {} ({} / {})'.format(new_dS/comparison_dS,new_dS,comparison_dS))
        if abs(new_dS)>25*abs(comparison_dS):# or (distri_per_assembly[i_assem].TR[-2]-distri_per_assembly[i_assem].TR[-1])>0.01:
            print('Skipping cov!!!')
            new_cov = None
            distri.GSR.append(distri.GSR[-1])
    if new_cov is not None:
        distri.GSR.append(grandsum_ratio)
        distri.cov = new_cov
    write_to_series([[titan.time,grandsum_ratio,distri.GSR[-1]]],[['Time','Sum_Ratio','Smoothed Ratio']],options.output_folder+'/GSR.csv')
        

def adaptive_entropy_splitting(assembly,options,dt=1.0):
    do_split = False

    for i_leaf in range(assembly.gaussian_library.n_leaf_nodes):
        leaf = assembly.gaussian_library.get_leaf_by_index(i_leaf)
        dH_distri = leaf.get_shannon_entropy_change()
        if leaf.dynamical_entropy is None or dH_distri is None: continue
        dH_prop = leaf.dynamical_entropy * dt
        print(dH_distri,dH_prop)
        delta_H = abs(dH_distri - dH_prop) - abs(options.uncertainty.GMM_eps * dH_prop)
        print('dH is {} (|{} - {}| - |{} * {}|)'.format(delta_H,dH_distri,dH_prop,options.uncertainty.GMM_eps,dH_prop))
        if delta_H >0: 
            do_split = True
            break

    if do_split and options.uncertainty.split_GMM:
        for _ in range(options.uncertainty.GMM_n_splits): assembly.gaussian_library.split_leaf()

def write_to_series(data_array,columns,filename):
    import pandas as pd
    import os
    data=pd.DataFrame(data_array,columns=columns)
    doHeader = False if os.path.exists(filename) else True
    data.to_csv(filename,mode='a',index=False,header=doHeader)

def untwist_sigmas(sigma_points):
    # If our sigma points cross over we can do terrible topological crimes to our coviariance matrix
    # Need to make sure that doesn't happen
    n = np.shape(sigma_points)[1]
    mean = sigma_points[0,:]
    # Sigmas points are an N x 2N +1 array of where the first element is the mean
    simplex = sigma_points[1:,:]
    # This remaining N x 2N array has as its ith element a positive displacement in the ith dimension
    # ...and its i+Nth element as a negative displacement in the ith dimension
    
    new_sigmas = np.zeros_like(sigma_points)
    new_sigmas[0,:] = mean
    available_slots = [i for i in range(2*n)]

    for i_axis in range(n):
        plus_point = simplex[i_axis,:]
        minus_point = simplex[i_axis+n,:]
        delta_plus = plus_point[i_axis] - mean[i_axis]
        delta_minus = minus_point[i_axis] - mean[i_axis]
        
        ## Nominal Case, no untwisting needed
        if delta_plus>0 and delta_minus<0:
            new_sigmas[i_axis+1,:]   = plus_point
            new_sigmas[i_axis+n+1,:] = minus_point
        ## Points have crossed the axis, flip to avoid an inverted covariance
        elif delta_plus<0 and delta_minus>0:
            print('Untwisting Covariance in axis {}'.format(i_axis))
            new_sigmas[i_axis+1,:]   = minus_point
            new_sigmas[i_axis+n+1,:] = plus_point
        ## Worst case scenario, if both points are on one side of the mean
        elif delta_plus<0 and delta_minus<0 or delta_plus>0 and delta_minus>0:
            warn('Sigma points are lopsided! Unphysical behaviour may occur')
            new_sigmas[i_axis+1,:]   = [plus_point,minus_point][np.argmax([delta_plus,delta_minus])]
            new_sigmas[i_axis+n+1,:] = [plus_point,minus_point][np.argmin([delta_plus,delta_minus])]
    return new_sigmas

def reject_small_eigenvalues(Cov,n_to_clip):
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)
    for clip in range(n_to_clip): eigenvalues[clip] = 0
    return eigenvectors @ np.diag(eigenvalues) @ np.transpose(eigenvectors)      
