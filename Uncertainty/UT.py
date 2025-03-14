from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import concurrent.futures, psutil
from Dynamics.euler import compute_Euler
from Dynamics import frames
from Dynamics.propagation import quaternion_normalize, compute_jacobian_diagonal
from Output.output import write_to_series
import pandas as pd
import os
import numpy as np
from collections.abc import MutableSequence
from copy import deepcopy
import pymap3d
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from Dynamics.frames import R_NED_ECEF
from warnings import warn
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
from statsmodels.stats.correlation_tools import cov_nearest
from copy import copy
from functools import partial
import gc
params_names =['quaternion','trajectory','yaw','pitch','roll','aoa','slip','roll_vel','pitch_vel','yaw_vel','velocity']
n_assem = 0

class dynamicDistribution():
    
    def __init__(self, assembly, mean_states, cov, RNG = np.random.RandomState(42069), convert_quaternionic=False, convert_geodetic=False, conversion_samples = 10000, DOF = 6, distribution = multivariate_normal, altitudinal=False):
        self.mean = mean_states
        self.cov = cov
        self.R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
        self.DOF = DOF
        self.n = 13 if self.DOF==6 else 6
        if altitudinal: convert_geodetic = True

        set_of_states = None
        if convert_geodetic or convert_quaternionic: mc_states = distribution(mean_states,cov,allow_singular = True).rvs(conversion_samples)
        if convert_geodetic: set_of_states = self.propagate_to_geodetic(mc_states)
        if convert_quaternionic and self.DOF==6: set_of_states = self.propagate_to_quaternion(mc_states,set_of_states)
        if altitudinal: set_of_states = set_of_states[:,1:]
        self.mean = np.mean(set_of_states, axis=0)
        self.cov = np.cov(set_of_states, rowvar=False)
    
        self.RNG = RNG
        self.distri = distribution(self.mean,self.cov,seed=self.RNG, allow_singular = True)


    def propagate_to_quaternion(self,mc_states,set_of_states=None): # Sometimes Monte Carlo really is the best way
        no_existing_states = False
        if set_of_states is None: 
            no_existing_states = True
            set_of_states = np.zeros((np.shape(mc_states)[0],13))

        for i_state, euler_state in enumerate(mc_states):
            if no_existing_states: set_of_states[i_state,:6] = euler_state[:6]
            set_of_states[i_state,-3:] = euler_state[-3:]
            R_B_NED =   frames.R_B_NED(roll = euler_state[6], pitch = euler_state[7], yaw = euler_state[8]) 
            q = quaternion_normalize((self.R_NED_ECEF*R_B_NED).as_quat())
            set_of_states[i_state,6:10] = q
        return set_of_states
    
    def propagate_to_geodetic(self,mc_states):
        set_of_states = np.zeros((np.shape(mc_states)[0],self.n))

        for i_state, ecef_state in enumerate(mc_states):
            lat, lon, alt = pymap3d.ecef2geodetic(ecef_state[0],ecef_state[1],ecef_state[2],deg=False)
            E, N, U = pymap3d.uvw2enu(ecef_state[3],ecef_state[4],ecef_state[5],lat,lon,deg=False)

            vel = np.linalg.norm(ecef_state[3:6])
            fpa = np.arcsin(np.dot(ecef_state[:3], ecef_state[3:6])/(np.linalg.norm(ecef_state[:3])*np.linalg.norm(ecef_state[3:6])))
            ha = np.arctan2(E,N)
            set_of_states[i_state,:6] = [alt, lat, lon, vel, fpa, ha]
        
        return set_of_states

    def rvs(self, N = 1, RNG = None):
        if RNG is None: RNG = self.RNG
        return self.distri.rvs(N, RNG)

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
    options.uncertainty.DOF = 3 if len(mean)<12 else 6
    convert_quaternionic = True if len(mean)<13 else True
    convert_geodetic = True if options.uncertainty.propagate_geodetic else False
    altitudinal = True if options.uncertainty.altitudinal else False
    distri =  dynamicDistribution(assembly,mean,cov,convert_quaternionic=convert_quaternionic,
                                  convert_geodetic=convert_geodetic,DOF=options.uncertainty.DOF, altitudinal=altitudinal)
    if not is_Library: return distri
    assembly.gaussian_library = recursive_gaussan_mixture(mean = distri.mean,
                                                          cov = distri.cov,
                                                          is_leaf=True,
                                                          library_size=options.uncertainty.GMM_library)
    if options.uncertainty.use_GMM:
        for _ in range(options.uncertainty.GMM_a_priori_splits): assembly.gaussian_library.split_leaf()
    return assembly.gaussian_library

class recursive_gaussan_mixture():
    def __init__(self, mean, cov, weight = 1.0, is_leaf = True, library_size = 3, tree_size = 1, rng = np.random.RandomState(seed=42069)):

        ## Statistical Parameters of Gaussian
        self.mean = np.array(mean)
        self.cov = cov
        self.empirical_cov = None
        self.dim = len(self.mean.flatten())
        self.distribution = None

        ## Node Parameters of Tree
        self.n_leaf_nodes = tree_size
        self.rng = rng
        self.leaf_list = [] if not is_leaf else [self]
        self.is_leaf = is_leaf
        self.children = []

        ## Parameters of Mixture
        self.library_size = library_size
        self.weight = weight
        self.shannon_entropy = []
        self.dynamical_entropy = None

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
            self.split_self()
        else: self.update_distribution()

    def get_mixture_parameters(self,rootcov):
        # Split component along principal axis of covariance hyper-ellipsoid
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(rootcov)
        split_axis = len(self.eigenvalues)-1
        weights = self.weight * self.libraries[self.library_size]['weights']
        component_eigenvalues= copy(self.eigenvalues)
        component_eigenvalues[-1] = component_eigenvalues[-1]*(self.libraries[self.library_size]['std'])**2
        cov = self.eigenvectors @ np.diag(component_eigenvalues) @ self.eigenvectors.transpose()
        means = []
        for i_component in range(self.library_size):
            means.append(self.mean + np.sqrt(self.eigenvalues[-1]) * self.libraries[self.library_size]['means'][i_component] * self.eigenvectors[:,split_axis])
        return weights, means, cov
    
    def get_shannon_entropy_change(self):
        self.shannon_entropy.append(0.5 * self.dim * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.cov)) + self.dim/2)
        dH = self.shannon_entropy[-1] - self.shannon_entropy[-2] if len(self.shannon_entropy)>1 else None
        return dH
    
    def recalculate_dynamical_entropy(self,sigma_points, sigma_derivatives):
        jac = compute_jacobian_diagonal(sigma_points,sigma_derivatives,self.dim)
        self.dynamical_entropy = -1*np.trace(jac)
    
    def update_distribution(self):
        self.distribution = multivariate_normal(self.mean,self.cov, allow_singular = True)
        self.empirical_cov = self.run_empirical_cov()

    def rvs(self,n=1):
        if self.is_leaf: 
            result = self.distribution.rvs(n)
        else:
            result = np.zeros_like(self.mean)
            probabilities = [leaf.weight for leaf in self.leaf_list]
            probabilities = np.divide(probabilities,np.sum(probabilities))
            leaf_selection = np.random.choice(a=len(self.leaf_list),size=n,p=probabilities)
            selected_leaf, n_per_leaf = np.unique(leaf_selection, return_counts=True)

            for i_leaf, leaf_index in enumerate(selected_leaf):
                leaf_result = self.get_leaf_by_index(leaf_index).rvs(n=n_per_leaf[i_leaf])
                result = np.vstack((result,leaf_result))
            result = result[1:,:]
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
                [self.leaf_list.append(leaf) for leaf in child.leaf_list]



    def split_self(self):
        rootcov = self.run_empirical_cov()
        # This node is no longer a leaf node
        self.is_leaf = False
        self.n_leaf_nodes -= 1

        self.n_leaf_nodes += self.library_size
        self.children = []
        
        weights, means, cov = self.get_mixture_parameters(rootcov)
        for i_child in range(self.library_size):
            self.children.append(recursive_gaussan_mixture(mean         =  means[i_child],
                                                           cov          =  cov_nearest(cov),
                                                           weight       =  weights[i_child],
                                                           is_leaf      =  True,
                                                           library_size =  self.library_size,
                                                           tree_size    =  1,
                                                           rng          =  self.rng))
    
    def split_leaf(self):
        max_uncertainty = 0.0
        split_candidate = None
        i_candidate = None
        for i_leaf, leaf in enumerate(self.leaf_list):
            leaf_cov = leaf.empirical_cov
            uncertainty, direction = np.linalg.eigh(leaf_cov)
            if uncertainty[-1] > max_uncertainty:
                max_uncertainty = uncertainty[-1]
                eigvec = direction[-1]
                split_candidate = leaf
                i_candidate = i_leaf
        print('Splitting Leaf {} (with max variance {} at {})'.format(i_candidate,max_uncertainty,eigvec))
        split_candidate.split_self()
        self.build_leaf_list(recursive=True)
    
    def recalculate_mean(self):
        if self.is_leaf: return self.weight*self.mean
        else:
            mean = np.zeros_like(self.mean)
            for child in self.children: mean += child.recalculate_mean()
            return mean
    
    def run_empirical_cov(self,fidelity=10000):
        return np.cov(self.rvs(fidelity),rowvar=False)
        
    def write_to_series(self, options, time, assem_id, leaf_id):
        [latitude, longitude, altitude] = pymap3d.ecef2geodetic(self.mean[0], 
                                                                self.mean[1], 
                                                                self.mean[2],
                                                                ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                                                deg = True);
        data_array = np.hstack(([time, altitude, latitude, longitude],self.mean,self.cov.flatten())).reshape(1,-1)
        
        if self.dim==13:
            columns = [['Time','Alt','Lat','Lon','X','Y','Z','U','V','W','Q_w','Q_i','Q_j','Q_k','P','Q','R']]
            axes =  ['x','y','z','u','v','w','qw','qi','qj','qk','p','q','r']
        else:
            columns = [['Time','Alt','Lat','Lon','X','Y','Z','U','V','W']]
            axes =  ['x','y','z','u','v','w']
        for ax1 in axes:
            for ax2 in axes:
                columns[0].append('cov_{}/{}'.format(ax1,ax2))
        
        write_to_series(data_array,columns,options.output_folder+'/UT_Assem_{}_Gaussian_{}.csv'.format(assem_id,leaf_id))
    
    def do_unscented_transform(self,sigmas,Wm,Wc,altitudinal):
        if self.dim<13: 
            if altitudinal: sigmas=sigmas[:,1:self.dim+1]
            else: sigmas = sigmas[:,:self.dim]
        mu, cov = unscented_transform(sigmas=sigmas, Wm=Wm, Wc=Wc)
        self.mean = mu
        self.cov = cov_nearest((reject_small_eigenvalues(cov,2)))
        self.update_distribution()       

def UT_propagator(subpropagator, state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    ## Determnine the number of propagations to do (this can get very high)
    if not hasattr(options, "n_points"): options.n_points = 0
    for _assembly in titan.assembly: adaptive_entropy_splitting(_assembly,options,dt)
    n_calcs = np.max([_assembly.gaussian_library.n_leaf_nodes for _assembly in titan.assembly])
    dim = 13 if options.uncertainty.DOF==6 else 6
    point_generator = MerweScaledSigmaPoints(n=dim, alpha=options.uncertainty.UT_alpha, 
                                             beta=options.uncertainty.UT_beta, kappa=options.uncertainty.UT_kappa)
    
    # Need to do a not insignificant amount of preprocessing to get a nicely parallelisable iterable of all our points
    propagation_iterable = []
    distribution_id_iterable = []
    compute_flag_iterable = []
    for i_calc in range(n_calcs):
        ## Generate our points
        sigmas_per_assembly, compute_flags = extract_propagation_points(titan,dim,i_calc,point_generator)
        
        sigma_point_iterable = []
        for i_sigma in range(2*dim+1):
            sigma_point_iterable.append([])
            for i_assem, _assembly in enumerate(titan.assembly): 
                sigma_point_iterable[i_sigma].append(sigmas_per_assembly[i_assem][i_sigma,:])
            propagation_iterable.append(sigma_point_iterable[i_sigma])
            distribution_id_iterable.append(i_calc)
            compute_flag_iterable.append(compute_flags)
    if options.uncertainty.n_procs>1:
        wrapper_func = partial(propagation_wrapper,subpropagator, dt, options, titan)   
    else: wrapper_func = partial(serial_propagator,subpropagator,dt,titan, options)
    nominal_state = update_mixtures(titan, options, wrapper_func, dim, point_generator,
                                            propagation_iterable,distribution_id_iterable,compute_flag_iterable)
    
    output_vector = []
    for i_assem, _assembly in enumerate(titan.assembly):    
        _assembly.gaussian_library.mean = _assembly.gaussian_library.recalculate_mean()
        output_vector.append(_assembly.gaussian_library.mean)
        output_vector[i_assem] = np.hstack((output_vector[i_assem],nominal_state[i_assem][dim:])) 
        fixed_alt = _assembly.gaussian_library.mean[0]

        write_mini_monte_carlo(_assembly,options,titan.time,fixed_alt)
        for i_leaf in range(_assembly.gaussian_library.n_leaf_nodes):
            leaf = _assembly.gaussian_library.get_leaf_by_index(i_leaf)
            leaf.write_to_series(options, titan.time, _assembly.id, i_leaf)
    if options.uncertainty.propagate_geodetic: output_vector = convert_to_ecef(output_vector)
    return output_vector, None

def update_mixtures(titan, options, wrapper_func, dim, point_generator,propagation_iterable,distribution_id_iterable,compute_flag_iterable, return_states=False):
    n_points = len(distribution_id_iterable)
    options.n_points = n_points
    n_distris = len(np.unique(distribution_id_iterable))
    if n_points>100: 
        #from messaging import messenger
        message = 'Propagating {} points(!) for {} Gaussians (iter {})'.format(n_points, n_distris, titan.iter)
        #msg = messenger('',threshold=900)
        #msg.print_n_send(message)
        print(message)
    else: print('Propagating {} points for {} Gaussians'.format(n_points, n_distris))
    
    if options.uncertainty.propagate_geodetic: propagation_iterable = convert_to_ecef(propagation_iterable)
    if options.uncertainty.n_procs>1: 
        new_states, new_d_dt, = parallel_propagator(wrapper_func, options.uncertainty.n_procs, 
                                                   titan, propagation_iterable, compute_flag_iterable)
    else: new_states, new_d_dt = wrapper_func(propagation_iterable, compute_flag_iterable)
    if return_states: return new_states
    n_sigmas_per_distri = 2*dim+1
    nominal_state = []
    for i_distri, point_pointer in zip(range(n_distris),np.arange(0,n_points,n_sigmas_per_distri)):
        old_points = np.array(propagation_iterable[point_pointer:point_pointer+n_sigmas_per_distri])
        new_points = new_states[:,point_pointer:point_pointer+n_sigmas_per_distri,:]
        derivatives = new_d_dt[:,point_pointer:point_pointer+n_sigmas_per_distri,:]
        for i_assem, _assembly in enumerate(titan.assembly):
            if compute_flag_iterable[point_pointer][i_assem]:
                distri = _assembly.gaussian_library.get_leaf_by_index(i_distri)
                distri.recalculate_dynamical_entropy(old_points[:,i_assem,:],derivatives[i_assem,:,:])
                ## Finally we can recombine our sigma points to get our new distribution
                if options.uncertainty.propagate_geodetic: new_points[i_assem,:,:] = convert_to_geodetic(new_points[i_assem,:,:])
                distri.do_unscented_transform(sigmas=new_points[i_assem,:,:],Wm=point_generator.Wm,Wc=point_generator.Wc,altitudinal=options.uncertainty.altitudinal)
                if i_distri==0: nominal_state.append(new_points[i_assem,0,:])
    return nominal_state

def extract_propagation_points(titan,dim,i_calc,point_generator,alt = None):
##  Collect all the sigma points to propagate
    sigmas_per_assembly = []
    distri_per_assembly = []
    compute_flags = []
    for i_assem, _assembly in enumerate(titan.assembly): 
        if _assembly.gaussian_library.n_leaf_nodes<=i_calc:
            compute_flags.append(False)
            sigmas_per_assembly.append(np.zeros([2*dim+1,13]))
            distri_per_assembly.append(None)
        else:
            compute_flags.append(True)
            distri_per_assembly.append(_assembly.gaussian_library.get_leaf_by_index(i_calc))        
            points = point_generator.sigma_points(distri_per_assembly[i_assem].mean,distri_per_assembly[i_assem].cov)
            if alt is not None: points = np.hstack((alt*np.ones([2*dim+1,1]),points))
            sigmas_per_assembly.append(points)
            if dim<12:
                assembly_rotation = np.array([_assembly.state_vector[6:] for _ in range(2*dim+1)])
                sigmas_per_assembly[-1] = np.hstack((sigmas_per_assembly[-1],assembly_rotation))

    return sigmas_per_assembly, compute_flags

def propagation_wrapper(subpropagator, dt, options, titan, sigma_points, compute_flags):
    states = np.zeros_like(sigma_points)
    d_dts  = np.zeros_like(sigma_points)
    for i_point, sigma_point in enumerate(np.rollaxis(sigma_points,1,0)):
        for _assembly,compute in zip(titan.assembly,compute_flags[i_point]):
            _assembly.compute = True if compute==True else False
        state, d_dt = subpropagator(sigma_point,None,None,dt,titan,options)
        for i_assem, _assembly in enumerate(titan.assembly):
            if titan.post_event_iter==0:
                states[i_assem,i_point,:] = state[i_assem]
                d_dts[i_assem,i_point,:] = d_dt[i_assem]
        del state, d_dt
    return states, d_dts, titan

def parallel_propagator(wrapper_func, n_procs, titan, sigma_points, compute_flags):
    def parallel_collation(i_future,point_index,future):
        nonlocal new_d_dt, new_state, titan
        if future._exception:
            raise Exception('Error on result number {}: {}'.format(i_future,future.exception()))
        else:
            result = future.result()
            for i_assem, _assembly in enumerate(titan.assembly):
                if i_future==0: 
                    titan.assembly[i_assem] = copy(result[2].assembly[i_assem])
                for i_point, index in enumerate(point_index):
                    new_state[i_assem,index,:] = result[0][i_assem,i_point,:]
                    new_d_dt[i_assem,index,:] = result[1][i_assem,i_point,:]
                finished_points = np.count_nonzero(new_state,axis=1)[0][0]
                total_points = np.shape(new_state)[1]
                print('Propagated {}/{} ({}%)'.format(finished_points,total_points,round(100*finished_points/total_points,2)))
        del future
        del result
        gc.collect()
        
    n_points = len(compute_flags)
    batches  = int(np.floor(n_points / n_procs))
    remainder = n_points % n_procs
    sigmas_per_proc = [np.zeros([len(titan.assembly),batches,13]) for _ in range(n_procs)]
    point_index = [[] for _ in range(n_procs)]
    flags_per_proc = [[] for _ in range(n_procs)]
    for i_proc in range(remainder):
        sigmas_per_proc[i_proc] = np.hstack((sigmas_per_proc[i_proc],np.zeros([len(titan.assembly),1,13])))

    for i_batch in range(batches):
        for i_proc in range(n_procs):
            index = i_batch*n_procs+i_proc
            sigmas_per_proc[i_proc][:,i_batch,:] = sigma_points[:,index,:]
            point_index[i_proc].append(index)
            flags_per_proc[i_proc].append(compute_flags[index])
    for i_proc in range(remainder):
        index = batches*n_procs+i_proc
        sigmas_per_proc[i_proc][:,batches,:] = sigma_points[:,index,:]
        point_index[i_proc].append(index)
        flags_per_proc[i_proc].append(compute_flags[index])

    new_state = np.zeros([len(titan.assembly),len(compute_flags),13])
    new_d_dt  = np.zeros_like(new_state)
    with concurrent.futures.ProcessPoolExecutor(n_procs) as executor:
        output_futures = [executor.submit(wrapper_func, sigmas, compute_flags) for sigmas, compute_flags in zip(sigmas_per_proc, flags_per_proc)]
        for i_future, future in enumerate(output_futures):
            future.add_done_callback(partial(parallel_collation,i_future,point_index[i_future]))

        concurrent.futures.wait(output_futures)   
    return new_state, new_d_dt

def serial_propagator(subpropagator,dt,titan, options, sigma_points, compute_flags):
    new_state = np.zeros([len(titan.assembly),len(compute_flags),13])
    new_d_dt  = np.zeros_like(new_state)
    for sigma_point, compute_flag, i_point in zip(sigma_points, compute_flags, range(len(compute_flags))):
        for _assembly,compute in zip(titan.assembly,compute_flag):
            _assembly.compute = True if compute==True else False
        state, d_dt = subpropagator(sigma_point,None,None,dt,titan,options)
        
        for i_assem, _assembly in enumerate(titan.assembly):
            new_state[i_assem,i_point,:] = state[i_assem]
            new_d_dt[i_assem,i_point,:] = d_dt[i_assem]
        if i_point % 10 == 0: 
                    print('Propagated Point {}'.format(i_point))
                    finished_points = np.count_nonzero(new_state,axis=1)[0][0]
                    total_points = np.shape(new_state)[1]
                    print('Progress {}/{} ({}%)'.format(finished_points,total_points,round(100*finished_points/total_points,2)))
    return new_state, new_d_dt

def convert_to_ecef(states, alt = None):
    state_shape = np.shape(states)
    states = np.reshape(states,[-1,13])
    alt = states[:,0] if alt is None else alt*np.ones(states.shape[0])
    E = states[:,3]*np.cos(states[:,4])*np.sin(states[:,5])
    N = states[:,3]*np.cos(states[:,4])*np.cos(states[:,5])
    U = states[:,3]*np.sin(states[:,4])
    x, y, z = pymap3d.geodetic2ecef(alt=alt,lat=states[:,1],lon=states[:,2],deg=False)
    u, v, w = pymap3d.enu2uvw(E,N,U,states[:,1],states[:,2],deg=False)
    states[:,:6] = np.transpose([x,y,z,u,v,w])
    return np.reshape(states, state_shape)

def convert_to_geodetic(states):
    state_shape = np.shape(states)
    states = np.reshape(states,[-1,13])
    lat, lon, alt = pymap3d.ecef2geodetic(states[:,0],states[:,1],states[:,2],deg=False)
    e, n, u = pymap3d.uvw2enu(states[:,3],states[:,4],states[:,5],lat,lon,deg=False)
    vel = np.linalg.norm(states[:,3:6],axis=1)
    fpa = [np.arcsin(np.dot(state[:3], state[3:6])/(np.linalg.norm(state[:3])*np.linalg.norm(state[3:6]))) for state in states]
    ha = np.arctan2(e,n)
    states[:,:6] = np.transpose([alt, lat, lon, vel, fpa, ha])
    return np.reshape(states,state_shape)

def adaptive_entropy_splitting(assembly,options,dt=1.0):
    do_split = False

    for i_leaf in range(assembly.gaussian_library.n_leaf_nodes):
        leaf = assembly.gaussian_library.get_leaf_by_index(i_leaf)
        dH_distri = leaf.get_shannon_entropy_change()
        if leaf.dynamical_entropy is None or dH_distri is None: continue
        dH_prop = leaf.dynamical_entropy * dt
        delta_H = abs(dH_distri - dH_prop) - abs(options.uncertainty.GMM_eps * dH_prop)
        print('dH is {} (|{} - {}| - |{} * {}|)'.format(delta_H,dH_distri,dH_prop,options.uncertainty.GMM_eps,dH_prop))
        if delta_H >0: 
            do_split = True
            break

    if do_split and options.uncertainty.split_GMM and options.n_points<options.uncertainty.max_points:
        for _ in range(options.uncertainty.GMM_n_splits): assembly.gaussian_library.split_leaf()

def reject_small_eigenvalues(Cov,n_to_clip):
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)
    for clip in range(n_to_clip): eigenvalues[clip] = 0
    return eigenvectors @ np.diag(eigenvalues) @ np.transpose(eigenvectors)

def write_mini_monte_carlo(_assembly, options, time, mean_alt= None):
    # We sample our Gaussian mixture to describe the true distribution
    rvs = _assembly.gaussian_library.rvs(1500)
    t = np.transpose([time*np.ones(1500)])
    if _assembly.gaussian_library.dim==13: 
        if options.uncertainty.propagate_geodetic:
            if options.uncertainty.altitudinal: rvs = np.hstack((mean_alt*np.ones((1500,1)),rvs))
            print(np.shape(rvs))
            rvs[:,[1,2,4,5]] *= 180/np.pi
            columns = [['Time','Alt','Lat','Lon','Vel','FPA','HA','Q_w','Q_i','Q_j','Q_k','P','Q','R']]
        else: columns = [['Time','X','Y','Z','U','V','W','Q_w','Q_i','Q_j','Q_k','P','Q','R']]
    else: 
        if options.uncertainty.propagate_geodetic:
            if options.uncertainty.altitudinal: rvs = np.hstack((mean_alt*np.ones((1500,1)),rvs))
            columns = [['Time','Alt','Lat','Lon','Vel','FPA','HA']]
        else: columns = [['Time','X','Y','Z','U','V','W']]
    data_array = np.hstack((t,rvs))
    print(data_array)
    write_to_series(data_array,columns,options.output_folder+'/UT_Assem_{}_rvs.csv'.format(_assembly.id))

def altitude_sliced_UT_propagator(subpropagator, state_vectors,state_vectors_prior,derivatives_prior,dt,titan,options):
    ## This code transforms based on altitude, not time, works only with ECEF presently but a GEO version would be nice

    if not hasattr(options, "n_points"): options.n_points = 0

    ## Determine the number of propagations to do (this can get very high)
    for _assembly in titan.assembly: adaptive_entropy_splitting(_assembly,options,dt)
    n_calcs = np.max([_assembly.gaussian_library.n_leaf_nodes for _assembly in titan.assembly])
    dim = 13 if options.uncertainty.DOF==6 else 6
    alt = None
    if options.uncertainty.altitudinal: 
        dim-=1
        alt = titan.assembly[0].trajectory.altitude
    point_generator = MerweScaledSigmaPoints(n=dim, alpha=options.uncertainty.UT_alpha, 
                                             beta=options.uncertainty.UT_beta, kappa=options.uncertainty.UT_kappa)
    

    propagation_iterable = []
    distribution_id_iterable = []
    compute_flag_iterable = []
    for i_calc in range(n_calcs):
        ## Generate our points
        sigmas_per_assembly, compute_flags = extract_propagation_points(titan,dim,i_calc,point_generator, alt=alt)
        
        sigma_point_iterable = []
        mean_iterable = []
        for i_sigma in range(2*dim+1):
            sigma_point_iterable.append([])
            for i_assem, _assembly in enumerate(titan.assembly): 
                sigma_point_iterable[i_sigma].append(sigmas_per_assembly[i_assem][i_sigma,:])
            propagation_iterable.append(sigma_point_iterable[i_sigma])
            distribution_id_iterable.append(i_calc)
            compute_flag_iterable.append(compute_flags)
    if options.uncertainty.n_procs>1:
        wrapper_func = partial(propagation_wrapper,subpropagator, dt, options, titan)   
    else: wrapper_func = partial(serial_propagator,subpropagator,dt,titan, options)

    nominal_state, alt = update_mixtures_alt(titan, options, wrapper_func, dim, point_generator,
                                        propagation_iterable,distribution_id_iterable,compute_flag_iterable)

    output_vector = []
    for i_assem, _assembly in enumerate(titan.assembly):    
        _assembly.gaussian_library.mean = _assembly.gaussian_library.recalculate_mean()
        if options.uncertainty.propagate_geodetic:
            if options.uncertainty.altitudinal: mean = convert_to_ecef(np.hstack((alt,_assembly.gaussian_library.mean)))
            else: mean = convert_to_ecef(_assembly.gaussian_library.mean,alt=alt)
        else: mean = _assembly.gaussian_library.mean
        output_vector.append(mean)
        output_vector[i_assem] = np.hstack((output_vector[i_assem],nominal_state[i_assem][dim:])) 
        write_mini_monte_carlo(_assembly,options,titan.time, alt)
        for i_leaf in range(_assembly.gaussian_library.n_leaf_nodes):
            leaf = _assembly.gaussian_library.get_leaf_by_index(i_leaf)
            leaf.write_to_series(options, titan.time, _assembly.id, i_leaf)
            
    return output_vector, None

def update_mixtures_alt(titan, options, wrapper_func, dim, point_generator,propagation_iterable,distribution_id_iterable,compute_flag_iterable, return_states=False):
    n_points = len(distribution_id_iterable)
    options.n_points = n_points
    n_distris = len(np.unique(distribution_id_iterable))
    if n_points>100: 
        from messaging import messenger
        message = 'Propagating {} points(!) for {} Gaussians (iter {})'.format(n_points, n_distris, titan.iter)
        msg = messenger('',threshold=900)
        msg.print_n_send(message)
    else: print('Propagating {} points for {} Gaussians'.format(n_points, n_distris))
    

    lowest_alt = None
    if options.uncertainty.propagate_geodetic: 
        alt = titan.assembly[0].trajectory.altitude if options.uncertainty.altitudinal else None
        propagation_iterable = convert_to_ecef(propagation_iterable, alt)
    else:
        points_as_list = np.reshape(propagation_iterable,[-1,13])
        initial_alts = pymap3d.ecef2geodetic(points_as_list[:,0],points_as_list[:,1],points_as_list[:,2])[2]
        lowest_alt = np.min(initial_alts)
        if lowest_alt<0: lowest_alt = 0

    halt_propagation = False
    n_props = 0
    propagation_iterable = np.rollaxis(np.array(propagation_iterable),1,0)
    in_propagation = propagation_iterable
    in_compute = np.array(compute_flag_iterable)
    propagated_states = np.zeros_like(propagation_iterable)
    iterable_indices = np.arange(0,np.shape(in_propagation)[1])
 
    while not halt_propagation:
        if options.uncertainty.n_procs>1: 
            in_propagation, _, = parallel_propagator(wrapper_func, options.uncertainty.n_procs, 
                                                            titan, in_propagation, in_compute)
        else: in_propagation, _ = wrapper_func(in_propagation, in_compute)

        alts = [pymap3d.ecef2geodetic(in_propagation[i_assem,:,0],
                                       in_propagation[i_assem,:,1],
                                       in_propagation[i_assem,:,2])[2] for i_assem in range(len(titan.assembly))]
        if lowest_alt is None: lowest_alt = np.min(alts)
        below_alt = np.argwhere(alts<lowest_alt)
        for completed in below_alt:
            propagated_states[completed[0],iterable_indices[completed[1]],:] = in_propagation[completed[0],completed[1],:]
            in_compute[completed[1],completed[0]] = False
        to_delete = np.argwhere(np.all(~in_compute,axis=1))
        if len(to_delete)>0:
            to_delete = to_delete.flatten()
            in_propagation = np.delete(in_propagation,to_delete,axis=1)
            iterable_indices = np.delete(iterable_indices,to_delete)
            in_compute = np.delete(in_compute,to_delete,axis=0)
            if np.shape(in_propagation)[1]==0: halt_propagation = True

    n_sigmas_per_distri = 2*dim+1
    nominal_state = []
    for i_distri, point_pointer in zip(range(n_distris),np.arange(0,n_points,n_sigmas_per_distri)):
        new_points = propagated_states[:,point_pointer:point_pointer+n_sigmas_per_distri,:]
        for i_assem, _assembly in enumerate(titan.assembly):
            if compute_flag_iterable[point_pointer][i_assem]:
                distri = _assembly.gaussian_library.get_leaf_by_index(i_distri)
                distri.dynamical_entropy = 0.0
                ## Finally we can recombine our sigma points to get our new distribution
                if options.uncertainty.propagate_geodetic: new_points[i_assem,:,:] = convert_to_geodetic(new_points[i_assem,:,:])
                distri.do_unscented_transform(sigmas=new_points[i_assem,:,:],Wm=point_generator.Wm,Wc=point_generator.Wc, altitudinal = options.uncertainty.altitudinal)
                if i_distri==0: nominal_state.append(new_points[i_assem,0,:])
    return nominal_state, np.mean(alts)