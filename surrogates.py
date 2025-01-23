from Uncertainty.HDMR import HDMR_sampling, HDMR_ROM, HDMR_postprocessing, recompose_multifi
from Uncertainty.atmosphere import pull_freestream_stats, mpp_solve_freestream
from Uncertainty.double_loop import double_loop_UQ, plot_pbox
from Uncertainty.function_helpers import partial_helper
from Aerothermo import su2,aerothermo
from Configuration import configuration
from Freestream import mix_properties, atmosphere
from Dynamics.dynamics import compute_quaternion
import concurrent.futures
import configparser
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform_direction, uniform
from filelock import FileLock
from matplotlib import pyplot as plt
import pickle, os
import seaborn as sns
import pandas as pd
import dask.dataframe as dd
import gc
from messaging import messenger
from math import floor
import psutil
is_freestream = ['temperature','density','velocity']
is_aleatory = ['density','temperature','velocity','aoa','slip','roll']
is_material = ['catalycity', 'specificHeatCapacity', 'emissivity']
plt.style.use('dark_background')
helper = partial_helper()
n_procs = psutil.cpu_count(logical=False)


## TODO Surrogates: Fix double loop to be single output, restructure HDMR and surrogates scripts for co

def compute_fluxes(assembly,dt):
    Tref = 273

    for obj in assembly.objects:
        facet_area = np.linalg.norm(obj.mesh.facet_normal, ord = 2, axis = 1)
        heatflux = assembly.aerothermo.heatflux[obj.facet_index]
        Qin = np.sum(heatflux*facet_area)
        
        cp  = obj.material.specificHeatCapacity(obj.temperature)
        emissivity = obj.material.emissivity

        Atot = np.sum(facet_area)

        # Estimating the radiation heat-flux
        Qrad = 5.670373e-8*emissivity*(obj.temperature**4 - Tref**4)*Atot
        T_ss = (Qin/(5.670373e-8*emissivity*Atot)-Tref**4)**(0.25)
        #print('T_ss is {}'.format(T_ss))

        # Computing temperature change
        dT = (Qin-Qrad)*dt/(obj.mass*cp)

        new_T = obj.temperature + dT
    return Qin-Qrad, T_ss

def check_input(parameter_vector,db_name):

    def condition(row, parameter_vector):
        return (row.tolist() == parameter_vector).all()
    output = None
    try:
        with FileLock(db_name+'_HDMR.lock'):
            if not os.path.exists(db_name) or os.stat(db_name).st_size == 0: return None
            database = pd.read_csv(db_name, header=None)

        n_params = len(parameter_vector)
        parameter_rows = database.iloc[:,:n_params]
        matches = (parameter_rows.values == parameter_vector).all(axis=1)

        found_rows = matches.nonzero()[0]
        if len(found_rows)<1: 
            output = None
        else:
            output=database.iloc[found_rows[0],n_params:].values.flatten()
    except Exception as e:
        print('Error in input check! {}'.format(e))
        output = None
    return output

def add_output(parameter_vector,output_vector,db_name):
    try:
        new_row = pd.DataFrame([np.hstack((parameter_vector,output_vector))])
        with FileLock(db_name+'_HDMR.lock'): new_row.to_csv(db_name,header=False,mode='a',index=False)
    except Exception as e:
        print('Error writing file: {}'.format(e))


def rebase_parameters(parameter_vector,loc_scale_vector,assembly):
    freestream_vector = [0,0,0]
    for value, param in zip(parameter_vector, loc_scale_vector.items()):
        name, loc_scale = param
        if name in is_freestream:
            freestream_vector[is_freestream.index(name)] = loc_scale[0]+value*loc_scale[1]
        elif name in is_material:
            setattr(assembly.objects[0].material,name,loc_scale[0]+value*loc_scale[1])
        else: setattr(assembly,name,loc_scale[0]+value*loc_scale[1])

    assembly.freestream = mpp_solve_freestream(*freestream_vector,assembly.freestream)
    mix_properties.compute_stagnation(assembly.freestream,options)
    compute_quaternion(assembly)
    return assembly

def flow_lofi(extra_args,parameter_vector):
    
    [assem,titan,options,loc_scale_vector,csv] = extra_args
    
    output = check_input(parameter_vector,csv)
    if output is not None:
        #print('Found existing solve at {}'.format(parameter_vector))
        return output
    assem = rebase_parameters(parameter_vector,loc_scale_vector,assem)
    aerothermo.compute_low_fidelity_aerothermo([assem],options)
    q_integrated, body_temp = compute_fluxes(assembly=assem,dt=options.dynamics.time_step)
    #body_temp+=norm(loc=0,scale=0.03*body_temp).rvs()
    try: add_output(parameter_vector,[body_temp],csv)
    except Exception as e: print('Error adding data! {}'.format(e))
    return np.array([body_temp])


def borehole_func(is_hifi,loc_scale_vector,parameter_vector):
    var ={}
    for i_param, (name,[loc, scale]) in enumerate(loc_scale_vector.items()):
        var[name]=loc+parameter_vector[i_param]*scale

    fidelity_vars = [2 * np.pi, 1] if is_hifi else [5,1.5]
    flow_rate_numerator = fidelity_vars[0] * var['trans_aquifer_u'] * (var['head_aquifer_u'] - var['head_aquifer_l'])

    radius_ratio_ln = np.log(var['r'] / var['RoI'])

    flow_rate_denominator = radius_ratio_ln*(fidelity_vars[1]+(2*var['L']*var['trans_aquifer_u'])/(radius_ratio_ln*var['cond_hydro']*var['r']**2)+(var['trans_aquifer_u']/var['trans_aquifer_l']))
    flow_rate = flow_rate_numerator / flow_rate_denominator
    
    #csv = 'hifi_ORACLE.csv' if is_hifi else 'lofi_ORACLE.csv'
    #add_output(parameter_vector,[flow_rate],csv)
    return flow_rate

def flow_hifi(extra_args,parameter_vector):

    [assem,titan,options,loc_scale_vector,csv] = extra_args
    output = check_input(parameter_vector,csv)
    if output is not None: return output
    assem = rebase_parameters(parameter_vector,loc_scale_vector,assem)

    su2.compute_cfd_aerothermo([assem],titan,options)

    assem.aerothermo.heatflux = assem.aerothermo_cfd.heatflux
    q_integrated, body_temp = compute_fluxes(assembly=assem,dt=options.dynamics.time_step)
    add_output(parameter_vector,[body_temp],csv)
    return np.array([body_temp])

def build_surrogates(n_params, outputs, lofi_callable, additive_multifi_callable, hf_order, lf_order, use_direct_error=False, output_epsilon='mean', error_criterion=1.0,plot_conv=False):
    # Note you should always pass an additive multifi callable, not a hifi or scale one
    # Note that this ^ is terrible and that Tommy should really fix it
    print('\nBeginning sampling and surrogate construction, reminder if you have changed... \n...meshes, \n...material parameters, \n...trajectory information, \n...model parameters \nYou need to delete your ORACLE files or you\'ll be using old results!\n')
    is_homo=True if lf_order%1==0 else False
    lf_order=int(floor(lf_order))
    sampler_add = HDMR_sampling(the_function=additive_multifi_callable,max_domain_order=hf_order,number_of_parameters=n_params,number_of_outputs=len(outputs),database_name='discrepancy.csv',parallel=True)
    sampler_lf = HDMR_sampling(the_function=lofi_callable,max_domain_order=lf_order,number_of_parameters=n_params,number_of_outputs=len(outputs),database_name='lofi.csv',parallel=True)
    print('\nFinished sampling, constructing multifi distance surrogate...')
    hdmr_add=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
    print('Finished constructing scale surrogate, constructing hifi and lofi surrogate...')
    hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
    hdmr_add_callable = helper.construct_callable(type='corrected_surrogate',comparison_method='distance',
                                                  arguments=[helper.construct_callable(arguments=[hdmr_lf,outputs]),
                                                             helper.construct_callable(arguments=[hdmr_add,outputs])])

    n_lofi = len(pd.read_csv('lofi.csv', header=None).index)
    n_hifi = len(pd.read_csv('discrepancy.csv', header=None).index)
    mean_error = np.inf
    list_of_errors = []
    max_increment = 250
    n_increment = 0
    rolling_avg = 5*n_procs
    n_hifi_enrichmented = 100*(hf_order**2)
    previous_error = np.Inf
    print('Finished construction, enriching high fidelity from {} to {} samples...'.format(n_hifi,n_hifi_enrichmented))

    while n_hifi<n_hifi_enrichmented:
        DoE, subdomains, _ = sampler_add.enrich(rom=hdmr_add, samples_per_enrichment=n_procs,output_name=outputs[-1])
        #hdmr_add=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
        n_hifi = len(pd.read_csv('discrepancy.csv', header=None).index)
        print('{} / {}...'.format(n_hifi,n_hifi_enrichmented))
    try:    
        hdmr_scale = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
        hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
        hdmr_scale_callable = helper.construct_callable(type='corrected_surrogate',comparison_method='scale',
                                                        arguments=[helper.construct_callable(arguments=[hdmr_lf,outputs]),
                                                                helper.construct_callable(arguments=[hdmr_scale,outputs])])
        hifi_hdmr_callable = helper.construct_callable(arguments=[hdmr_hf,outputs])
    except:
        hdmr_scale = None
        hdmr_hf = None
        hdmr_scale_callable = None
        hifi_hdmr_callable = None

    n_lofi = len(pd.read_csv('lofi.csv', header=None).index)
    n_lofi_minimal = n_lofi#floor(1.1*n_lofi)
    
    print('Finished hifi enrichment, doing lofi adaptive enrichment until ε≤{}%...'.format(error_criterion))
    should_continue=True
    if plot_conv: 
        plt.ion()
        fig, ax  = plt.subplots()
        error_line, = ax.plot([n_lofi], [100], label="Error",marker='x',linestyle='None')
        rolling_avg_line, = ax.plot([n_lofi], [100], label="Rolling Average")
        ax.set_title("Converging...")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Error (%)")
        ax.set_yscale('log') 
        ax.legend()
        ax.grid(True)
    while should_continue:
        if n_increment>=max_increment: break
        if is_homo:
            _, _, list_of_subdomains = sampler_add.variance_GP_selection(rom=hdmr_add, samples_per_enrichment=n_procs, output_name=outputs[-1])
            list_of_subdimensions = [subd.subdimensions for subd in list_of_subdomains]
            _, _, epsilons = sampler_lf.enrich(rom=hdmr_lf, samples_per_enrichment=n_procs,output_name=outputs[-1],list_of_subdimensions=list_of_subdimensions)#, ground_truth=lofi_callable)
        else: _, _, epsilons = sampler_lf.enrich(rom=hdmr_lf, samples_per_enrichment=n_procs,output_name=outputs[-1])#, ground_truth=lofi_callable)
        

        n_lofi = len(pd.read_csv('lofi.csv', header=None).index)
        if n_lofi>n_lofi_minimal:
            
            if len(list_of_errors)>=2:
                 if list_of_errors[-1]>=list_of_errors[-2]: pass
            #if n_increment % 5 ==0: hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
            lofi_hdmr_callable = helper.construct_callable(arguments=[hdmr_lf,outputs])
            if use_direct_error: 
                errors = direct_error_check(n_params, lofi_callable, lofi_hdmr_callable, region=1.0, n_samples=1000)
                iter_error =np.mean(errors)
                
            else:
                if output_epsilon=='summed':
                    iter_error = np.sum([(epsilons[output]) for output in epsilons.keys()],axis=0)
                elif output_epsilon=='mean':
                    iter_error = np.mean([(epsilons[output]) for output in epsilons.keys()],axis=0)
                else:
                    iter_error = epsilons[output_epsilon]
                iter_error = np.array(iter_error).flatten()
            list_of_errors = np.hstack((list_of_errors,iter_error))

            if len(list_of_errors) < rolling_avg:
                error_series = list_of_errors
                rolling_var = np.inf
            else:
                error_series = list_of_errors[-rolling_avg:]
                rolling_var = np.std(error_series)
            mean_error = np.mean(error_series)
            if mean_error>previous_error: hdmr_lf=HDMR_ROM(order=hdmr_lf.order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv',subdomain_blacklist=hdmr_lf.subdomain_blacklist)
            previous_error = mean_error
            print('Mean error is... {}%, std of {} | Hifi {} Lofi {}'.format(mean_error,rolling_var,n_hifi,n_lofi))
            if plot_conv:
                try:
                    error_line.set_data(np.arange(n_lofi_minimal,n_lofi),list_of_errors)
                    averages = np.hstack((rolling_avg_line.get_ydata(),[mean_error])).flatten()
                    rolling_avg_line.set_data(np.arange(n_lofi_minimal-n_procs,n_lofi,n_procs),averages)
                    ax.set_title("Converging Order {}... ({} Batches)".format(hdmr_lf.order,n_increment))
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                except: 
                    pass
            if mean_error<error_criterion or rolling_var<0.3*error_criterion:
                print('Surrogate is converging! ({}%)'.format(mean_error))
                # hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
                # lofi_hdmr_callable = helper.construct_callable(arguments=[hdmr_lf,outputs])
                # errors = direct_error_check(n_params, lofi_callable, lofi_hdmr_callable, region=1.0, n_samples=1000)
                # mean_error = np.mean(errors)
                if rolling_var<np.inf: 
                    if hdmr_lf.order<=3:
                        print('################################### \n           ORDER\n       INCREASING ({})\n###################################'.format(hdmr_lf.order+1))
                        hdmr_lf, sampler_lf = sampler_lf.increase_order(rom=hdmr_lf,percentage_tolerance=100*1e-7)
                        print('Updated Blacklist =',hdmr_lf.subdomain_blacklist)
                        list_of_errors = []
                        if plot_conv:
                            n_lofi = len(pd.read_csv('lofi.csv', header=None).index)
                            n_lofi_minimal = n_lofi
                            plt.ion()
                            fig = plt.figure()
                            ax  = fig.add_subplot()
                            error_line, = ax.plot([n_lofi], [100], label="Error",marker='x',linestyle='None')
                            rolling_avg_line, = ax.plot([n_lofi], [100], label="Rolling Average")
                            ax.set_title("Converging...")
                            ax.set_xlabel("Samples")
                            ax.set_ylabel("Error (%)")
                            ax.set_yscale('log') 
                            ax.legend()
                            ax.grid(True)
                    else:
                        if mean_error>error_criterion and rolling_var>0.3*error_criterion: print('Nevermind! Error is {}%, continuing sampling...'.format(mean_error))
                        else: should_continue=False
            # elif rolling_var<error_criterion or n_increment % 10 ==0: 
            #     hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
            #     plotter = HDMR_postprocessing(hdmr_lf,output_name='Flow Rate (m^3/s)',
            #                                   input_names=['r','RoI','trans_aquifer_u','head_aquifer_u','trans_aquifer_l','head_aquifer_l','L','cond_hydro']).plot_subdomains()

            n_increment+=1
    if plot_conv: plt.ioff()
    print('Finished enrichment! Writing models...')
    n_lofi = len(pd.read_csv('lofi.csv', header=None).index)
    with open('error_history.pkl','wb') as file: pickle.dump(list_of_errors,file)

    hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
    lofi_hdmr_callable = helper.construct_callable(arguments=[hdmr_lf,outputs])
    hdmr_add=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
    hdmr_add_callable = helper.construct_callable(type='corrected_surrogate',comparison_method='distance',
                                                  arguments=[helper.construct_callable(arguments=[hdmr_lf,outputs]),
                                                             helper.construct_callable(arguments=[hdmr_add,outputs])])
    try:
        hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
        hdmr_scale_callable = helper.construct_callable(type='corrected_surrogate',comparison_method='scale',
                                                        arguments=[helper.construct_callable(arguments=[hdmr_lf,outputs]),
                                                                helper.construct_callable(arguments=[hdmr_scale,outputs]),
                                                                outputs])
        hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
        hifi_hdmr_callable = helper.construct_callable(arguments=[hdmr_hf,outputs])
    except:
        hdmr_scale = None
        hdmr_hf = None
        hdmr_scale_callable = None
        hifi_hdmr_callable = None


    return {'Additive HDMR':hdmr_add_callable,'Multiplicative HDMR':hdmr_scale_callable,'Lofi HDMR':lofi_hdmr_callable,'Hifi HDMR':hifi_hdmr_callable,'n_samples':[n_hifi,n_lofi]}

def direct_error_check(n_params, ground_truth_callable, surrogate_callable, region=1.0, n_samples=10):
    errors = []
    interval = uniform(loc=-1,scale=2)
    samples = np.array([interval.rvs(n_samples) for _ in range(n_params)]).reshape((-1,n_params))
    for i,x in enumerate(samples):
        lofi_val = surrogate_callable(x)
        ground_truth = np.array(ground_truth_callable(x)).flatten()[0]
        errors.append(abs(100*(lofi_val-ground_truth)/ground_truth))
        if i%50==0: print('Model predicted {} (actual is {}), {}% error'.format(lofi_val,ground_truth,errors[-1]))
    return errors

def model_comparison(n_params, surrogates, ground_truth_callable,n_samples=2000):

    region = 1.0
    errors_vector = {}
    for name in surrogates.keys(): errors_vector[name]=[]
    gt = []
    values = {}
    interval = uniform(loc=-1,scale=2)
    samples = np.array([interval.rvs(n_samples) for _ in range(n_params)]).reshape((-1,n_params))
    for i,x in enumerate(samples):
        ground_truth = np.array(ground_truth_callable(x)).flatten()[0]
        for name, func in surrogates.items():
            values[name] = np.array(func(x)).flatten()[0]
            errors_vector[name].append(100*abs(values[name]-ground_truth)/ground_truth)
            #pct = 100 * errors_vector[name][-1]/ground_truth
            if i%100==0: print('{} error of {}%'.format(name,errors_vector[name][-1]))
        gt.append(ground_truth)
    printable ='MEAN VALS'
    for name, err in errors_vector.items(): printable+=' | {} {} |'.format(name,np.mean(err))
    print(printable)
    data = pd.DataFrame.from_dict(errors_vector)
    with open('error_compare.pkl','wb') as file: pickle.dump(data,file)
    return data

if __name__=='__main__':

    hfpath = '/home/ckb18135/uq/aviation_HDMR.cfg'
    lfpath = '/home/ckb18135/uq/aviation_HDMR_lofi.cfg'

    configParser = configparser.RawConfigParser()   

    configParser.read(hfpath)
    options_hf,titan_hf=configuration.read_config_file(configParser,'')
    options_hf.method=options_hf.freestream.method
    assem_hf = titan_hf.assembly[0]

    configParser.read(lfpath)
    options_lf,titan_lf=configuration.read_config_file(configParser,'')
    options_lf.method=options_lf.freestream.method
    assem_lf = titan_lf.assembly[0]
    msg = messenger()
    # Note one cfg must be taken as reference, here is lf purely for convenience
    for options, assem in zip([options_hf,options_lf],[assem_hf,assem_lf]):
    # The object also needs to have a freestream object thus...
        mix_properties.compute_freestream(model=options.freestream.model,
                                            altitude=assem.trajectory.altitude,
                                            velocity=assem.trajectory.velocity,
                                            lref=1.26,
                                            assembly=assem,
                                            freestream=assem.freestream,
                                            options=options)
        
    fs_stats = pull_freestream_stats(options,assem.trajectory.velocity)
    # Surface data for ZrB2-SiC EDM from https://doi.org/10.1111/j.1551-2916.2008.02325.x
    # e = 0.81+-0.04, catalycity = 0.00191+-0.3*0.00191
    # Thermal Data from https://doi.org/10.1111/j.1551-2916.2008.02268.x
    # cp = 0.70+-0.03*0.70, k = 
    # Yes I know, it's a vector that's actually a dict, sue me
    # location_scale_vector = {'temperature' : [0,0],'density' : [0,0],'velocity' :[0,0],'aoa' : [0,np.deg2rad(15)],'slip' : [0,np.deg2rad(3)],
    #                          'catalycity' : [0.00417,0.00327], 'specificHeatCapacity' : [628,1], 'emissivity' : [0.81,0.04]} # Need to find some values for material props, currently using ZrB2
    location_scale_vector = {'temperature' : [0,0],'density' : [0,0],'velocity' :[0,0],'aoa' : [0,np.deg2rad(15)],'slip' : [0,np.deg2rad(3)],
                             'catalycity' : [0.00417,0.00327], 'emissivity' : [0.81,0.04]} # Need to find some values for material props, currently using ZrB2
    # it's also not a vector its a matrix so double sue me
    
    # Borehole function data from https://www.sfu.ca/~ssurjano/borehole.html
    borehole_vector = {'r' : [0.1,0.05],'RoI':[25050,24950],'trans_aquifer_u':[89335,26265],
                       'head_aquifer_u':[1050,60],'trans_aquifer_l':[89.55,26.45],'head_aquifer_l':[760,60],
                       'L':[1400,280],'cond_hydro':[10950,1095]}
    
    use_borehole = True

    if not use_borehole:
        fs_stats = pull_freestream_stats(options,assem.trajectory.velocity)
        n_params = len(location_scale_vector)
        outputs = ['Surface Temperature']
        for freestream_param, stats in fs_stats.items():
            if freestream_param in location_scale_vector: location_scale_vector[freestream_param] = stats
        # Bound problem by +/- 5 sigma for aleatory parameters
        aleatory_flags =[]
        for param, location_scale in location_scale_vector.items():
            if param in is_aleatory:
                location_scale[1] *= 10
                aleatory_flags.append(True)
            else: aleatory_flags.append(False)
    

        # Use partial to construct functions that only accept our parameter vector
        lofi = partial(flow_lofi,[assem_lf,titan_lf,options_lf,location_scale_vector,'lofi_ORACLE.csv'])
        hifi_mesh = partial(flow_lofi,[assem_hf,titan_hf,options_hf,location_scale_vector,'hifi_ORACLE.csv'])

    else:
        n_params = len(borehole_vector)
        outputs = ['Flow Rate (m^3/s)']

        lofi = partial(borehole_func, False, borehole_vector)
        hifi_mesh = partial(borehole_func, True, borehole_vector)

    multifi_dist = helper.construct_callable(type='model_discrepancies',comparison_method='distance',arguments=[hifi_mesh,lofi])
    multifi_scal = helper.construct_callable(type='model_discrepancies',comparison_method='scale',arguments=[hifi_mesh,lofi])
    
    hf_order = 2 # be wary of changing this number lol
    lf_order = 3
    skip = False
    hetero_models = {}
    sample_count = {}
    hetero_orders =[1.5,2.5,3.5,4.5]
    #colors = ['']
    if not skip:
        for hf_order in [1,2,3,4]:
            # try: os.remove('discrepancy.csv')
            # except: pass
            surrogates = build_surrogates(n_params=n_params, outputs=outputs, lofi_callable=lofi, additive_multifi_callable=multifi_dist, 
                                          hf_order=hf_order, lf_order=hf_order+0.5,plot_conv=True,error_criterion=0.1,use_direct_error=False)
            msg.print_n_send('Finished low fi homo of order {}'.format(hf_order))
            model_name = 'Homogeneous at Order {}'.format(hf_order)
            hetero_models[model_name] = surrogates['Additive HDMR']
            sample_count[model_name] = surrogates['n_samples'][-1]
            os.remove('lofi.csv')
            for lf_order in hetero_orders:
                surrogates = build_surrogates(n_params=n_params, outputs=outputs, lofi_callable=lofi, additive_multifi_callable=multifi_dist, 
                                              hf_order=hf_order, lf_order=lf_order,plot_conv=True,error_criterion=0.1,use_direct_error=False)
                model_name ='Order {} Heterogeneous (at Order {})'.format(int(floor(lf_order)),hf_order)
                msg.print_n_send('Finished low fi hetero of order {} (at order {})'.format(lf_order, hf_order))
                os.remove('lofi.csv')
                

            
            with open('hetero.pkl','wb') as f: pickle.dump([hetero_models,sample_count],f)
    else: 
        with open('hetero.pkl','rb') as f: hetero_models, sample_count=pickle.load(f)
    print('Convergence assured! Here are all model statistics...')
    
    n_error_samples = 2000
    data = model_comparison(n_params, hetero_models, hifi_mesh, n_samples=n_error_samples)
    fig = plt.figure()
    ax = sns.ecdfplot(data,complementary=True,legend=True,palette="Paired",linewidth=3.0)
    ax.set_xlabel('Percentage Error in Temperature Prediction (%)')
    ax.set_title('Hifi Model on a constant 2500 Samples, Lofi model adaptively converged to 0.1% MAPE in relevant domains...')
    
    print(sample_count)
    #count_frame=pd.concat({k: pd.DataFrame(v).T for k, v in sample_count.items()}, axis=0)
    #print(count_frame)
    fig = plt.figure()
    ax2 = sns.barplot(sample_count,palette='Paired')
    plt.show()

    # time = 720
    # hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
    # lofi_callable = partial(callable_surrogate,hdmr_lf,outputs)
    # hdmr_disc=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
    # hdmr_disc_callable= partial(distance_surrogate,hdmr_lf,hdmr_disc,outputs)
    # hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
    # hdmr_scale_callable = partial(scale_surrogate,hdmr_lf,hdmr_scale,outputs)
    # # combo_func = partial(multifi_discrepancy, hifi_mesh,hdmr_disc_callable)
    # # hdmr_combo=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv')
    # # hdmr_combo_callable = partial(combo_surrogate, hdmr_disc_callable,hdmr_combo,outputs)
    # hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
    # hifi_callable = partial(callable_surrogate,hdmr_hf,outputs)
    # #plt.show()
    # print('############### RUNNING MULTIFI SURROGATE ###############')
    # outputs_mf_surr=double_loop_UQ(sampled_function=hdmr_scale_callable,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    # print('############### RUNNING LOFI SURROGATE ###############')
    # outputs_lf_surr=double_loop_UQ(sampled_function=lofi_callable,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    # print('############### RUNNING HIFI SURROGATE ###############')
    # outputs_hf_surr=double_loop_UQ(sampled_function=hifi_callable,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    # print('#################### RUNNING LOFI ####################')
    # outputs_lf = double_loop_UQ(sampled_function=lofi,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    # print('#################### RUNNING HIFI ####################')
    # #outputs_hf = double_loop_UQ(sampled_function=hifi_mesh,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    
    # output_array = [outputs_mf_surr[0],outputs_lf_surr[0],outputs_hf_surr[0],outputs_lf[0]]
    # try: 
    #     with open('output_array_timed.pkl','wb') as file: pickle.dump(output_array,file)
    # except:
    #     for i_output, output in enumerate(output_array):
    #         try:
    #             with open('output_array_'+str(i_output)+'_timed.pkl','wb') as file: pickle.dump(output,file)
    #         except:
    #             pass
    
    # tones = ['g','rb','r']#,'b','bg']
    # ax=''
    # for out, tone in zip(output_array,tones):
    #     ax = plot_pbox(out,tone,ax)
    # labels =[]
    # for name in ['Multifi HDMR ','Lofi HDMR ','Hifi HDMR']:#,'Lofi ','Hifi ']: 
    #     labels.append(name+' Minimum')
    #     labels.append(name+' Maximum')
    #     labels.append('_fill')
    # ax.legend(labels)

    # n_samples = 1000
    # #outputs_lf = double_loop_UQ(sampled_function=lofi,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    # #outputs_hf = double_loop_UQ(sampled_function=hifi_mesh,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    # outputs_mf_surr=double_loop_UQ(sampled_function=hdmr_scale_callable,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    # outputs_lf_surr=double_loop_UQ(sampled_function=lofi_callable,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    # outputs_hf_surr=double_loop_UQ(sampled_function=hifi_callable,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    # output_array = [outputs_mf_surr[0],outputs_lf_surr[0],outputs_hf_surr[0]]#,outputs_hf[0],outputs_lf[0]]
    # try: 
    #     with open('output_array_sample.pkl','wb') as file: pickle.dump(output_array,file)
    # except:
    #     for i_output, output in enumerate(output_array):
    #         try:
    #             with open('output_array_'+str(i_output)+'_sample.pkl','wb') as file: pickle.dump(output,file)
    #         except:
    #             pass

    # plt.show()
    # # plt.plot(list_of_errors)
    # # plt.show()
    # # HDMR_postprocessing(hdmr_lf,output_name='Mean_Heat_Flux',input_names=['Temperature','Density','Velocity','AoA','Slip','Roll']).plot_subdomains()
    # # HDMR_postprocessing(hdmr_mf,output_name='Mean_Heat_Flux',input_names=['Temperature','Density','Velocity','AoA','Slip','Roll']).plot_subdomains()


