from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import concurrent.futures
from Dynamics.euler import compute_Euler
import pandas as pd
import os
import numpy as np
from collections.abc import MutableSequence
from copy import deepcopy
import pymap3d
from scipy.linalg import sqrtm
from Dynamics.frames import R_NED_ECEF

params_names =['quaternion','trajectory','yaw','pitch','roll','aoa','slip','roll_vel','pitch_vel','yaw_vel','velocity']
class sigmaPosition(MutableSequence):

    def __init__(self, dims=3,alpha=1,beta=2,kappa=0,position=[0,0,0],cov=[[1,0,0],[0,1,0],[0,0,1]],trajectory=''):
        self.n=dims
        self.points=MerweScaledSigmaPoints(n=3, alpha=alpha, beta=2., kappa=0)
        self.position=position
        self.mu=position
        self.cov=cov
        self.sqrtcov = sqrtm(cov)
        self.regenerate_sigmas()
        self.params = [[] for i in range(len(self.sigmas))]
        self.params_names = params_names


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
        mu=self.mu if mu=='' else mu
        cov=self.cov if cov=='' else cov
        self.sigmas = self.points.sigma_points(mu,cov)

    def UT(self):
        self.mu, self.cov = unscented_transform(sigmas=self.sigmas,Wm=self.points.Wm,Wc=self.points.Wc)
        self.regenerate_sigmas()

    def select_point(self,i):
        self.position = self.sigmas[i].copy()
    
    def get_trajectory(self,i):
        return deepcopy(self.trajectories[i])
    
    def update_sigma(self,i_point):
        self.sigmas[i_point]=self.position.copy()
    
    def write_data(self,options,assem_ID,time):
        #data_sp = np.reshape(self.sigmas, (1,21))
        data_cov = np.reshape(self.cov, (1,9))
        data_sp = []
        columns=['Time']
        for i_point in range(7):
            geo = pymap3d.ecef2geodetic(self.sigmas[i_point][0], self.sigmas[i_point][1], self.sigmas[i_point][2],
                                    ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                    deg = False)
            R_ned = R_NED_ECEF(lat =geo[0],lon=geo[1])
            if i_point==0: data_cov = R_ned.as_matrix()*self.cov*(R_ned.as_matrix().transpose())
            ned = R_ned.apply(self.sigmas[i_point],inverse=True)
            for iax, ax in enumerate(['north','east','down']):
            #for iax, ax in enumerate(['lat','lon','alt']):
            #for iax, ax in enumerate(['x','y','z']):
                columns.append('sigma_'+ax+'_'+str(i_point))
                data_sp.append(ned[iax])#self.sigmas[i_point][iax])
        
        data_cov = np.reshape(data_cov,(1,9))
        [columns.append('Cov_'+str(i)) for i in range(9)]
        data=pd.DataFrame(np.hstack(([[time]],[data_sp],data_cov)),columns=columns)
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
    UQ = uncertaintySupervisor(isActive=1)
    UQ.constructInputs(options.uncertainty.yaml_path)
    [alt,lat,lon] = UQ.inputs.get_statistics('mean')
    [sigma_alt,sigma_lat,sigma_lon] = UQ.inputs.get_statistics('std')
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
    assembly.position = sigmaPosition(position=assembly.position,cov=options.cov,trajectory=deepcopy(assembly.trajectory))
    params_objects = [assembly.quaternion,assembly.trajectory,assembly.yaw,assembly.pitch,assembly.roll,assembly.aoa,assembly.slip,assembly.roll_vel,assembly.pitch_vel,assembly.yaw_vel,assembly.velocity]
    for i_point in range(7):
        assembly.position.build_params(i_point,params_objects)
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
        sigmas.append(deepcopy(assem.position.position))
        params_objects = [getattr(assem, name) for name in params_names]
        params.append(deepcopy(assem.position.update_params(i_point,params_objects)[i_point]))

    
    return sigmas, params
    

def unscentedPropagation(titan, options):

    n_sigma = 7

    with concurrent.futures.ProcessPoolExecutor(n_sigma) as executor:
        output_futures = [executor.submit(wrapper,i_point,titan,options) for i_point in range(n_sigma)]

        # for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
        #    print('Finished point: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/n_sigma,4))+'%)')
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
                
    titan.time+=options.dynamics.time_step
    for i_assem, assem in enumerate(titan.assembly):
        assem.position.UT()
        assem.position.write_data(options,assem.id,titan.time)