from Uncertainty.HDMR import HDMR_sampling, HDMR_ROM, HDMR_postprocessing
from Uncertainty.atmosphere import pull_freestream_stats, mpp_solve_freestream
from Uncertainty.double_loop import double_loop_UQ, plot_pbox
from Aerothermo import su2,aerothermo
from Configuration import configuration
from Freestream import mix_properties, atmosphere
from Dynamics.dynamics import compute_quaternion
import concurrent.futures
import configparser
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform_direction
from filelock import FileLock
from matplotlib import pyplot as plt
import pickle, os
import seaborn as sns
import pandas as pd
is_freestream = ['temperature','density','velocity']
is_aleatory = ['density','temperature','velocity','aoa','slip','roll']
is_material = ['catalycity', 'specificHeatCapacity', 'emissivity']

## TODO Surrogates: Fix double loop to be single output, pass wrappers of certain outputs instead of vector wrappers
##                  Construct wrapper_function class to hold all wrapper funcs 

def distance_surrogate(hdmr_base,hdmr_dist,output_names,vector):
    output =[]
    for output_name in output_names:
        output.append(hdmr_base.call_global_surrogate(vector,output_name)-hdmr_dist.call_global_surrogate(vector,output_name))
    return output

def scale_surrogate(hdmr_base,hdmr_scale,output_names,vector):
    output =[]
    for output_name in output_names:
        output.append(hdmr_base.call_global_surrogate(vector,output_name)*hdmr_scale.call_global_surrogate(vector,output_name))
    return output

def combo_surrogate(mf_hdmr_base,hdmr_dist,output_names,vector):
    output =[]
    base = mf_hdmr_base(vector)
    for base_val, output_name in zip(base,output_names):
        output.append(base_val - hdmr_dist.call_global_surrogate(vector,output_name))
    return output

def callable_surrogate(hdmr,output_names,vector):
    output =[]
    for output_name in output_names:
        output.append(hdmr.call_global_surrogate(vector,output_name))
    return output

def recompose_multifi(in_hifi,in_lofi, input_form, output_form,n_out=1):
    # form can be hifi, distance, or scale
    if isinstance(in_hifi,str):
        hf = np.genfromtxt(in_hifi, delimiter=',')
    else: hf = in_hifi
    if isinstance(in_lofi,str):
        lf = np.genfromtxt(in_lofi,delimiter=',')
    else: lf = in_lofi
    new_db = None
    for i_row, row in enumerate(hf):
        find = np.where(np.all(np.isclose(lf[:,:-n_out],hf[i_row,:-n_out]),axis=1))[0]
        if len(find)<1: continue
        new_row = row
        lfrow = lf[find,:].flatten()
        # Firstly restore original hifi dataset...
        if input_form=='distance': new_row[-n_out:] = lfrow[-n_out:]-row[-n_out:]
        elif input_form == 'scale': new_row[-n_out:] = np.multiply(lfrow[-n_out:],row[-n_out:])
        
        if output_form=='distance': new_row[-n_out:] = lfrow[-n_out:]-row[-n_out:]
        elif output_form == 'scale': new_row[-n_out:] = np.divide(row[-n_out:],lfrow[-n_out:])
        if new_db is None: new_db=new_row
        else: new_db =np.vstack((new_db,new_row))
    return new_db
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
    output = None
    with FileLock(db_name+'.lock',timeout=30):
        if not os.path.exists(db_name) or os.stat(db_name).st_size == 0: return None
        try:
            database = np.genfromtxt(db_name,delimiter=',')
            if len(np.shape(database))==1:
                database = np.reshape(database,(1,len(database)))
            find = np.where(np.all(database[:,:len(parameter_vector)]==np.reshape(parameter_vector,[1,-1]),axis=1))[0]
            output = database[find,len(parameter_vector):,]
            if len(find)<1: output = None
            else: output=output.flatten()
        except:
            output = None
    return output

def add_output(parameter_vector,output_vector,db_name):
    with FileLock(db_name+'.lock',timeout=30):
        if not os.path.exists(db_name) or os.stat(db_name).st_size == 0:
            database = np.hstack((parameter_vector,output_vector))
            np.savetxt(db_name,[database],delimiter=',')
        else:
            try:
                database = np.genfromtxt(db_name,delimiter=',')
                if len(np.shape(database))==1:
                    database = np.reshape(database,(1,len(database)))
                database = np.vstack((database,np.hstack((parameter_vector,output_vector))))
                np.savetxt(db_name,database,delimiter=',')
            except (FileNotFoundError,OSError):
                print('Error writing {} > {}'.format(parameter_vector,output_vector))


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
        return output
    assem = rebase_parameters(parameter_vector,loc_scale_vector,assem)

    aerothermo.compute_low_fidelity_aerothermo([assem],options)

    q_integrated, body_temp = compute_fluxes(assembly=assem,dt=options.dynamics.time_step)
    # if output is not None:
    #     try: assert np.isclose(output[0],body_temp)
    #     except AssertionError:
    #         print('Error asserting that database ({}) = model!({})'.format(output[0],body_temp))
    add_output(parameter_vector,[body_temp],csv)
    return np.array([body_temp])


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

def multifi_discrepancy(func_reference,func,parameter_vector):

    discrepancies = []
    output_ref = np.reshape(func_reference(parameter_vector),-1)
    output = np.reshape(func(parameter_vector),-1)

    for out, out_ref in zip(output,output_ref):
        discrepancies.append(out-out_ref)

    return np.array(discrepancies)

def multifi_scaling(func_reference,func,parameter_vector):

    scalers = []
    output_ref = func_reference(parameter_vector)
    output = func(parameter_vector)

    for out, out_ref in zip(output,output_ref):
        scalers.append(out_ref/out)

    return np.array(scalers)

if __name__=='__main__':

    # csv_hf = 'discrepancy.csv'
    # csv_lf = 'lofi_ORACLE.csv'


    # out_hifi = recompose_multifi(in_hifi=csv_hf,in_lofi=csv_lf,input_form='distance',output_form='hifi')
    # out_scale = recompose_multifi(out_hifi,csv_lf,'hifi','scale')
    # assert out_scale.all() == recompose_multifi(csv_hf,csv_lf,'distance','scale').all()


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
    multifi_disc = partial(multifi_discrepancy, hifi_mesh, lofi)
    multifi_scal = partial(multifi_scaling, hifi_mesh, lofi)

    hf_order = 2 # be wary of changing this number lol
    lf_order = 3
    skip = True
    if not skip:
        print('\nBeginning sampling and surrogate construction, reminder if you have changed... \n...meshes, \n...material parameters, \n...trajectory information, \n...model parameters \nYou need to delete your ORACLE files or you\'ll be using old results!\n')
        sampler_mf = HDMR_sampling(the_function=multifi_disc,max_domain_order=hf_order,number_of_parameters=n_params,number_of_outputs=len(outputs),database_name='discrepancy.csv',parallel=True)
        sampler_lf = HDMR_sampling(the_function=lofi,max_domain_order=lf_order,number_of_parameters=n_params,number_of_outputs=len(outputs),database_name='lofi.csv',parallel=True)
        print('\nFinished sampling, constructing multifi distance surrogate...')
        hdmr_disc=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
        print('Finished constructing scale surrogate, constructing hifi and lofi surrogate...')
        hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
        hdmr_disc_callable= partial(distance_surrogate,hdmr_lf,hdmr_disc,outputs)
        

        #combo_func = partial(multifi_discrepancy, hifi_mesh,hdmr_scale_callable)
        #print('Converting scale samples to combo hdmr...\n')
        #sampler_combo = HDMR_sampling(the_function=combo_func,max_domain_order=hf_order,number_of_parameters=n_params,number_of_outputs=len(outputs),database_name='combo.csv',parallel=True)
        #print('Constructing combo hdmr...')
        #hdmr_combo = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv')
        #hdmr_combo_callable = partial(combo_surrogate, hdmr_scale_callable,hdmr_combo,outputs)

        n_lofi = max(np.shape(np.genfromtxt('lofi.csv',delimiter=',')))
        n_hifi = max(np.shape(np.genfromtxt('discrepancy.csv',delimiter=',')))
        mean_error = np.inf
        list_of_errors = []
        max_increment = 250
        n_increment = 0
        rolling_avg = 10
        n_hifi_enrichmented = 250
        print('Finished construction, enriching high fidelity from {} to {} samples...'.format(n_hifi,n_hifi_enrichmented))
        while n_hifi<n_hifi_enrichmented:
            DoE, subdomains = sampler_mf.enrich(rom=hdmr_disc, samples_per_enrichment=16,output_name=outputs[-1])
            hdmr_disc=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
            hdmr_disc_callable= partial(scale_surrogate,hdmr_lf,hdmr_disc,outputs)
            #combo_func = partial(multifi_discrepancy, hifi_mesh,hdmr_scale_callable)
            #hdmr_combo=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv')
            #sampler_combo.the_function = combo_func
            #sampler_combo.enrich(rom=hdmr_combo,sample=DoE,output_name=outputs[-1],subdomains=subdomains)

            #hdmr_combo=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv')
            #hdmr_combo_callable = partial(combo_surrogate, hdmr_scale_callable,hdmr_combo,outputs)
            #hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
            
            n_hifi+=16
            print('{} / {}...'.format(n_hifi,n_hifi_enrichmented))
        #HDMR_postprocessing(hdmr_mf,output_name=outputs[0],input_names=[name for name in location_scale_vector.keys()]).plot_subdomains()
        hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
        hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
        hdmr_scale_callable = partial(distance_surrogate,hdmr_lf,hdmr_scale,outputs)
        hifi_callable = partial(callable_surrogate,hdmr_hf,outputs)

        error_criterion = 2.5
        print('Finished hifi enrichment, doing lofi adaptive enrichment until ε≤{}%...'.format(error_criterion))
        while mean_error>error_criterion:
            if n_increment>=max_increment: break
            n_lofi += 16
            sampler_lf.enrich(rom=hdmr_lf, samples_per_enrichment=16,output_name=outputs[-1])
            hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv',epsilons=hdmr_lf.epsilons)#, ground_truth=lofi)
            errors = []
            region = 1.0
            inputs = region*uniform_direction(n_params).rvs(10)
            #hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv') # Why do I have to do this?
            #hdmr_combo=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv') # Why do I have to do this?
            for i,x in enumerate(inputs):
                lofi_val = hdmr_lf.call_global_surrogate(x=x,output_name=outputs[-1])
                #delta = hdmr_mf.call_global_surrogate(x=x,output_name=outputs[-1])
                #scale = hdmr_scale.call_global_surrogate(x=x,output_name=outputs[-1])
                #delta = hdmr_combo.call_global_surrogate(x=x,output_name=outputs[-1])
                #final = lofi_val+delta
                #mf = lofi_val*scale
                #final = mf - delta
                final = lofi_val
                ground_truth = lofi(x)[0]
                errors.append(abs(100*(final-ground_truth)/ground_truth))
                #if i%10==0: print('({} * {}) - {} = {} (actual is {}), {}% error'.format(lofi_val,scale,delta,final,ground_truth,errors[-1]))
                if i%50==0: print('Model predicted {} (actual is {}), {}% error'.format(final,ground_truth,errors[-1]))
            mean_error = np.mean(errors)
            std = np.std(errors)
            print('Mean error is... {}%, std of {}% Hifi {} Lofi {}'.format(mean_error,std,n_hifi,n_lofi))
            if mean_error<error_criterion:
                print('Surrogate is converging! Let\'s just double check that quickly...')
                for i,x in enumerate(region*uniform_direction(n_params).rvs(1000)):
                    lofi_val = hdmr_lf.call_global_surrogate(x=x,output_name=outputs[-1])
                    final = lofi_val
                    ground_truth = lofi(x)[0]
                    errors.append(abs(100*(final-ground_truth)/ground_truth))
                    if i%50==0: print('Model predicted {} (actual is {}), {}% error'.format(final,ground_truth,errors[-1]))
            list_of_errors.append(mean_error)
            n_increment+=1
        #HDMR_postprocessing(hdmr_lf,output_name=outputs[-1],input_names=[name for name in location_scale_vector.keys()],).plot_subdomains(plot_deltas=True)
        with open('error_history.pkl','wb') as file: pickle.dump(list_of_errors,file)
    hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
    lofi_callable = partial(callable_surrogate,hdmr_lf,outputs)
    hdmr_disc=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
    hdmr_disc_callable= partial(distance_surrogate,hdmr_lf,hdmr_disc,outputs)
    hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
    hdmr_scale_callable = partial(scale_surrogate,hdmr_lf,hdmr_scale,outputs)
    # combo_func = partial(multifi_discrepancy, hifi_mesh,hdmr_disc_callable)
    # hdmr_combo=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv')
    # hdmr_combo_callable = partial(combo_surrogate, hdmr_disc_callable,hdmr_combo,outputs)
    hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
    hifi_callable = partial(callable_surrogate,hdmr_hf,outputs)
    print('Convergence assured! Here are all model statistics...')
    
    region = 1.0
    lf_errors =[]
    combo_errors = []
    scale_errors =[]
    dist_errors =[]
    hf_errors = []
    samord_errors=[]
    gt = []
    for i,x in enumerate(region*uniform_direction(n_params).rvs(20)):
        lofi_val = hdmr_lf.call_global_surrogate(x=x,output_name=outputs[-1])
        lofi_samord = hdmr_lf.call_global_surrogate(x=x,output_name=outputs[-1],max_order=2)
        dist = hdmr_disc.call_global_surrogate(x=x,output_name=outputs[-1])
        #delta = hdmr_combo.call_global_surrogate(x=x,output_name=outputs[-1])
        scale = hdmr_scale.call_global_surrogate(x=x,output_name=outputs[-1])
        hf_syr = hifi_callable(x)[0]
        disted = lofi_val-dist
        scaled = lofi_val*scale
        #final = scaled - delta
        ground_truth = hifi_mesh(x)[0]

        #combo_errors.append(abs(final-ground_truth))
        lf_errors.append(abs(lofi_val-ground_truth))
        samord_errors.append(abs(lofi_val-ground_truth))
        scale_errors.append(abs(scaled-ground_truth))
        dist_errors.append(abs(disted-ground_truth))
        hf_errors.append(abs(hf_syr-ground_truth))
        gt.append(ground_truth)
        #ce_pct = 100 * combo_errors[-1]/ground_truth
        lf_pct = 100 * lf_errors[-1]/ground_truth
        sc_pct = 100 * scale_errors[-1]/ground_truth
        ds_pct = 100 * dist_errors[-1]/ground_truth
        hf_pct = 100 * hf_errors[-1]/ground_truth
        so_pct = 100* samord_errors[-1]/ground_truth

        if i%100==0: print('Error (%) | Lofi {} | Scaled {} | Distanced {} | Hifi HDMR {}'.format(lf_pct,sc_pct,ds_pct,hf_pct))
        print(                 'MEAN VALS | Lofi {} | Scaled {} | Distanced {} | Hifi HDMR {}'.format(np.mean(lf_errors),np.mean(scale_errors),np.mean(dist_errors),np.mean(hf_errors)))
        print('Finished lofi adaptive enrichment, doing the actual UQ now...')
        errs = [lf_errors,scale_errors,dist_errors,hf_errors]
        errs_pct = np.transpose([100*np.divide(err,gt) for i, err in enumerate(errs)])
        import matplotlib
        cmap=matplotlib.colormaps['magma']
        labels = ['Low Fidelity Surrogate','Multiplicative Multifidelity Surrogate','Additive Multifidelity Surrogate','High Fidelity Surrogate']
        data = pd.DataFrame(errs_pct,columns=labels)
        with open('error_compare.pkl','wb') as file: pickle.dump(data,file)
        #for err, lab in zip(errs,labels):
        ax = sns.ecdfplot(data,complementary=True,legend=True,palette='dark')
        # for i_color, color in enumerate([0.0,0.25,0.5,0.75]):
        #     ax.lines[i_color].set_color(cmap(color))
        ax.set_xlabel('Percentage Error in Temperature Prediction (%)')
        ax.set_title('Model Comparison on Test Points (2000 Samples)')

    time = 720
    hdmr_lf=HDMR_ROM(order=lf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='lofi.csv')
    lofi_callable = partial(callable_surrogate,hdmr_lf,outputs)
    hdmr_disc=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='discrepancy.csv')
    hdmr_disc_callable= partial(distance_surrogate,hdmr_lf,hdmr_disc,outputs)
    hdmr_scale=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','scale'))
    hdmr_scale_callable = partial(scale_surrogate,hdmr_lf,hdmr_scale,outputs)
    # combo_func = partial(multifi_discrepancy, hifi_mesh,hdmr_disc_callable)
    # hdmr_combo=HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',read_data_filename='combo.csv')
    # hdmr_combo_callable = partial(combo_surrogate, hdmr_disc_callable,hdmr_combo,outputs)
    hdmr_hf = HDMR_ROM(order=hf_order,output_names=outputs,surrogate_type='kriging',initial_data=recompose_multifi('discrepancy.csv','lofi_ORACLE.csv','distance','hifi'))
    hifi_callable = partial(callable_surrogate,hdmr_hf,outputs)
    #plt.show()
    print('############### RUNNING MULTIFI SURROGATE ###############')
    outputs_mf_surr=double_loop_UQ(sampled_function=hdmr_scale_callable,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    print('############### RUNNING LOFI SURROGATE ###############')
    outputs_lf_surr=double_loop_UQ(sampled_function=lofi_callable,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    print('############### RUNNING HIFI SURROGATE ###############')
    outputs_hf_surr=double_loop_UQ(sampled_function=hifi_callable,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    print('#################### RUNNING LOFI ####################')
    outputs_lf = double_loop_UQ(sampled_function=lofi,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    print('#################### RUNNING HIFI ####################')
    #outputs_hf = double_loop_UQ(sampled_function=hifi_mesh,aleatory_flags=aleatory_flags,threshold=time,output_names=outputs,mode='Time')
    
    output_array = [outputs_mf_surr[0],outputs_lf_surr[0],outputs_hf_surr[0],outputs_lf[0]]
    try: 
        with open('output_array_timed.pkl','wb') as file: pickle.dump(output_array,file)
    except:
        for i_output, output in enumerate(output_array):
            try:
                with open('output_array_'+str(i_output)+'_timed.pkl','wb') as file: pickle.dump(output,file)
            except:
                pass
    
    tones = ['g','rb','r']#,'b','bg']
    ax=''
    for out, tone in zip(output_array,tones):
        ax = plot_pbox(out,tone,ax)
    labels =[]
    for name in ['Multifi HDMR ','Lofi HDMR ','Hifi HDMR']:#,'Lofi ','Hifi ']: 
        labels.append(name+' Minimum')
        labels.append(name+' Maximum')
        labels.append('_fill')
    ax.legend(labels)

    n_samples = 1000
    #outputs_lf = double_loop_UQ(sampled_function=lofi,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    #outputs_hf = double_loop_UQ(sampled_function=hifi_mesh,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    outputs_mf_surr=double_loop_UQ(sampled_function=hdmr_scale_callable,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    outputs_lf_surr=double_loop_UQ(sampled_function=lofi_callable,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    outputs_hf_surr=double_loop_UQ(sampled_function=hifi_callable,aleatory_flags=aleatory_flags,threshold=n_samples,output_names=outputs,mode='Samples')
    output_array = [outputs_mf_surr[0],outputs_lf_surr[0],outputs_hf_surr[0]]#,outputs_hf[0],outputs_lf[0]]
    try: 
        with open('output_array_sample.pkl','wb') as file: pickle.dump(output_array,file)
    except:
        for i_output, output in enumerate(output_array):
            try:
                with open('output_array_'+str(i_output)+'_sample.pkl','wb') as file: pickle.dump(output,file)
            except:
                pass

    plt.show()
    # plt.plot(list_of_errors)
    # plt.show()
    # HDMR_postprocessing(hdmr_lf,output_name='Mean_Heat_Flux',input_names=['Temperature','Density','Velocity','AoA','Slip','Roll']).plot_subdomains()
    # HDMR_postprocessing(hdmr_mf,output_name='Mean_Heat_Flux',input_names=['Temperature','Density','Velocity','AoA','Slip','Roll']).plot_subdomains()


