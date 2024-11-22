from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import concurrent.futures, psutil
from Dynamics.euler import compute_Euler
from Dynamics.dynamics import compute_quaternion
import pandas as pd
import os
import numpy as np
from collections.abc import MutableSequence
from copy import deepcopy
import pymap3d
from scipy.linalg import sqrtm, cholesky
from Dynamics.frames import R_NED_ECEF
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
from statsmodels.stats.correlation_tools import cov_nearest

params_names =['quaternion','trajectory','yaw','pitch','roll','aoa','slip','roll_vel','pitch_vel','yaw_vel','velocity']
n_assem = 0
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