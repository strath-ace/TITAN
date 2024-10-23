import os
import numpy as np
from pyapprox import variables
from scipy.stats import *
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
import pandas as pd
from Uncertainty.dynamics_tools import apply_velocity_wind
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
from messaging import messenger


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



# missing_distris = {'irwinhall':irwinhall,'jf_skew_t':jf_skew_t,'rel_breitwigner':rel_breitwigner,
#                    'dirichlet_multinomial':dirichlet_multinomial,'vonmises_fisher':vonmises_fisher,
#                    betanbinom':betanbinom}
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
    msg=messenger(rank=proc)
    msg.read_data()
    with open(options.uncertainty.qoi_filepath,'rb') as file: options.uncertainty.quantities=pickle.load(file)

    stl = demise_object.name
    msg.print_n_send('Object of interest \'{}\' has demised at altitude {}'.format(stl,assembly.trajectory.altitude))
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
