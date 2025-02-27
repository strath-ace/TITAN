from TITAN import loop
from Configuration.configuration import read_config_file
from Uncertainty.UT import create_distribution
from Dynamics.propagation import update_dynamic_attributes
from messaging import messenger as msg
import configparser
import concurrent.futures
import copy
import os

def run(filename,n_samples):
    n_samples = int(n_samples)
    configParser = configparser.RawConfigParser()   
    configFilePath = filename.lstrip()
    configParser.read(configFilePath)

    base_options, base_titan = read_config_file(configParser,'')
    base_distri = create_distribution(base_titan.assembly[0],base_options,is_Library=False)

    initial_states = base_distri.rvs(n_samples)
    messenger = msg(threshold=30)
    with concurrent.futures.ProcessPoolExecutor(base_options.uncertainty.n_procs) as executor:
        output_futures = [executor.submit(wrapper,base_titan,base_options,i_sample,state) for i_sample, state in enumerate(initial_states)]

        for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
            if f._exception:
                messenger.print_n_send('Error on result number {}: {}'.format(i_sim,f.exception()))
            else:
                messenger.print_n_send('Finished sim: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/n_samples,4))+'%)')
        concurrent.futures.wait(output_futures)

def wrapper(titan, options, i_sample, state):
    print('Titan {} Options {} i_sample {}, state {}'.format(titan,options,i_sample,state))
    options.output_folder += '/MC_' + str(i_sample)
    options.clean_up_folders()
    options.create_output_folders()
    options.uncertainty.plot = True if i_sample % round(2*options.uncertainty.n_procs) == 0 else False
    titan.assembly[0].state_vector = state
    titan.assembly[0].state_vector_prior = []
    titan.assembly[0].derivs_prior = []

    update_dynamic_attributes(assembly=titan.assembly[0],state_vector=state,options=options)
    loop(options,titan)
