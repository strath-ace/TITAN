from Uncertainty.HDMR import HDMR_sampling, HDMR_ROM, HDMR_postprocessing, HDMR_data_processing
from Uncertainty.UT import convert_to_ecef, convert_to_geodetic
from Uncertainty.double_loop import double_loop_UQ, plot_pbox
from Uncertainty.function_helpers import partial_helper
from Aerothermo import su2,aerothermo
from Configuration import configuration
from Freestream import mix_properties, atmosphere
from Dynamics.propagation import explicit_rk_N
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

def state_vector_dynamics(propagator,titan,options,dt,loc_scale_vector,db_name,parameter_vector):
    output = check_input(parameter_vector,db_name)
    if output is None:
        rebased_vector = np.zeros_like(parameter_vector)
        for i_value, rebase in enumerate(loc_scale_vector.items()):
            name, loc_scale = rebase
            rebased_vector[i_value] = loc_scale[0]+parameter_vector[i_value]*loc_scale[1]
        
        output, _ = propagator(convert_to_ecef(rebased_vector),None,None,dt,titan,options)
        output = convert_to_geodetic(output)
        add_output(parameter_vector,output,db_name)
    return output


def build_surrogates(n_params, outputs, lofi_callable, additive_multifi_callable, hf_order, lf_order, use_direct_error=False, output_epsilon='mean', error_criterion=1.0,plot_conv=False):
    # Note you should always pass an additive multifi callable, not a hifi or scale one
    # Note that this ^ is terrible and that Tommy should really fix it
    print('\nBeginning sampling and surrogate construction, reminder if you have changed... \n...meshes, \n...material parameters, \n...trajectory information, \n...model parameters \nYou need to delete your ORACLE files or you\'ll be using old results!\n')
    is_homo=True if lf_order%1==0 else False
    lf_order=int(floor(lf_order))
    sampler_add = HDMR_sampling(the_function=additive_multifi_callable,max_domain_order=hf_order,central_point=np.zeros(n_params),number_of_outputs=len(outputs),database_name='discrepancy.csv',parallel=True)
    sampler_lf = HDMR_sampling(the_function=lofi_callable,max_domain_order=lf_order,central_point=np.zeros(n_params),number_of_outputs=len(outputs),database_name='lofi.csv',parallel=True)
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
    recomp = HDMR_data_processing
    while n_hifi<n_hifi_enrichmented:
        DoE, subdomains, _ = sampler_add.enrich(rom=hdmr_add, samples_per_enrichment=n_procs,output_name=outputs[-1])
        #hdmr_add=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
        n_hifi = len(pd.read_csv('discrepancy.csv', header=None).index)
        print('{} / {}...'.format(n_hifi,n_hifi_enrichmented))
    try:    
        hdmr_scale = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recomp.recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
        hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recomp.recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
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
        hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recomp.recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
        hdmr_scale_callable = helper.construct_callable(type='corrected_surrogate',comparison_method='scale',
                                                        arguments=[helper.construct_callable(arguments=[hdmr_lf,outputs]),
                                                                helper.construct_callable(arguments=[hdmr_scale,outputs]),
                                                                outputs])
        hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recomp.recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
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

    hfpath = '/home/ckb18135/uq/hayabusa.cfg'

    configParser = configparser.RawConfigParser()   

    configParser.read(hfpath)
    options_in,titan_in=configuration.read_config_file(configParser,'')
    titan_in.assembly[0].state_vector = np.zeros(13)
    msg = messenger()

    location_scale_vector = {'alt' : [60000,120000],'lat' : [0,np.pi],'lon' :[np.pi,2*np.pi],'v' : [3950,7900],'fpa' : [0,np.pi],
                             'ha'  : [0, np.pi], 'w' : [0.5,1.0],'i' : [0.5,1.0],'j' : [0.5,1.0],'k':[0.5,1.0],'p' : [15,30],'q' : [15,30],'r' : [15,30]}
    n_params = 13
    outputs = location_scale_vector.keys()
    # Use partial to construct functions that only accept our parameter vector
    dt = 10.0
    lofi_prop = partial(explicit_rk_N,2)
    hifi_prop = partial(explicit_rk_N,4)
    lofi = partial(state_vector_dynamics,lofi_prop,titan_in,options_in,dt,location_scale_vector,'lofi_ORACLE.csv')
    hifi = partial(state_vector_dynamics,hifi_prop,titan_in,options_in,dt,location_scale_vector,'hifi_ORACLE.csv')


    multifi_dist = helper.construct_callable(type='model_discrepancies',comparison_method='distance',arguments=[hifi,lofi])
    multifi_scal = helper.construct_callable(type='model_discrepancies',comparison_method='scale',arguments=[hifi,lofi])
    
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
    data = model_comparison(n_params, hetero_models, hifi, n_samples=n_error_samples)
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


