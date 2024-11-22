import concurrent.futures, multiprocessing
from scipy.optimize import minimize, shgo
from scipy.stats import truncnorm, uniform_direction
from functools import partial
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import psutil
import datetime as dt
import os
def double_loop_UQ(sampled_function,aleatory_flags,mode='Samples',threshold=660,outputs_indices = 0, plots = False,n_processes = 'Auto',output_names='Auto',discrepancy=None):
    try: n_outputs = len(outputs_indices)
    except:
        outputs_indices = [outputs_indices]
        n_outputs = len(outputs_indices)

    n_processes = psutil.cpu_count(logical=False) if n_processes == 'Auto' else n_processes
    if output_names=='Auto': output_names = ['Output '+str(i) for i in outputs_indices]
    if n_processes<=1:
        serial_UQ(sampled_function,aleatory_flags,mode,threshold,outputs_indices)
    
    scale = (1/5)
    a, b = (-1 - 0) / scale, (1 - 0) / scale

    aleatory_rvs = [truncnorm(a=a,b=b, loc=0,scale=scale) for _ in range(aleatory_flags.count(1))]
    

    if mode == 'Samples':
        outputs_all=call_sampler(threshold,outputs_indices,n_processes,sampled_function,aleatory_flags,aleatory_rvs)

    elif mode=='Time':
        outputs_all = None
        finishtime = dt.datetime.now()+dt.timedelta(seconds=threshold)
        now = dt.datetime.now()
        while now<finishtime:
            outputs = call_sampler(n_processes,outputs_indices,n_processes,sampled_function,aleatory_flags,aleatory_rvs)[0]
            if outputs_all is not None: 
                outputs_all = np.vstack((outputs_all,outputs))
            else: outputs_all = outputs
            now = dt.datetime.now()
            print(np.shape(outputs_all))
    outputs_all=[outputs_all]

    if plots:
        for i_output, output_of_interest in enumerate(outputs_indices):
            outputs = plot_pbox(outputs_all[i_output])

    
    return outputs_all
    

def opt_wrapper(the_function,aleatory_flags,sample,output_of_interest,direction,epistemic_vector):
    input_vector = []
    i_aleatory = 0
    i_epistemic = 0
    for is_aleatory in aleatory_flags:
        if is_aleatory:
            input_vector.append(sample[i_aleatory])
            i_aleatory+=1
        else:
            input_vector.append(epistemic_vector[i_epistemic])
            i_epistemic+=1
    return direction*np.reshape(the_function(input_vector),-1)[0]

def in_loop_optimise(sample,optimisation_parameters):
    sampled_function,aleatory_YN,output_of_interest = optimisation_parameters
    np.random.RandomState(os.getpid())
    min_func = partial(opt_wrapper,sampled_function,aleatory_YN,sample,output_of_interest,1)
    max_func = partial(opt_wrapper,sampled_function,aleatory_YN,sample,output_of_interest,-1)
    bounds = [(-1,1) for _ in range(aleatory_YN.count(0))]
    x0 = uniform_direction(aleatory_YN.count(0))
    max_opt=shgo(func=max_func,bounds=bounds)#,options={'disp':False})
    #min_opt=minimize(fun=min_func,x0=x0.rvs(),bounds=bounds,options={'disp':False})
    min_opt=shgo(func=min_func,bounds=bounds)
    print('Found min {} at {} and max {} at {}'.format(min_opt.fun,min_opt.x,-max_opt.fun,max_opt.x))
    # Some wires crossed here, idk why but this is correct
    return [min_opt.fun, -max_opt.fun]

def serial_UQ(sampled_function,aleatory_YN,mode='Samples',threshold=660,outputs_indices = [0], ):
    scale = (1/5)
    a, b = (-1 - 0) / scale, (1 - 0) / scale

    aleatory_rvs = [truncnorm(a=a,b=b, loc=0,scale=scale) for _ in range(aleatory_YN.count(1))]
    outputs_all = []
    if mode == 'Samples':
        for output_of_interest in outputs_indices:
            outputs = [0,0]
            samples = np.transpose([norm.rvs(threshold) for norm in aleatory_rvs])
            optimisation_parameters = [sampled_function,aleatory_YN,output_of_interest]
            outputs = np.array([in_loop_optimise(sample,optimisation_parameters) for sample in samples])
            outputs_all.append(outputs)

def plot_pbox(outputs,tone,ax=''):
        color_bright = [0.0,0.0,0.0]
        color_mid = [0.0,0.0,0.0]
        color_low = [0.0,0.0,0.0]
        for i_channel, channel in enumerate(['r','g','b']):
            if channel in tone:
                color_bright[i_channel]+=0.75
                color_mid[i_channel]+=0.5
                color_low[i_channel]+=0.25
        if ax=='':
            ax = sns.ecdfplot(outputs,color=color_low)
        else: ax = sns.ecdfplot(outputs,ax=ax,color=color_low)
        x1, y1 = ax.lines[-2].get_data()
        x2, y2 = ax.lines[-1].get_data()
        ax.lines[-2].set_color(color_mid)
        ax.lines[-1].get_data(color_low)
        ax.fill_betweenx(y1,x1,x2,step='pre',alpha=0.25,color=color_bright,label='_nolabel')
        return ax

def call_sampler(n_samples,outputs_indices,n_processes,sampled_function,aleatory_YN,aleatory_rvs):
    outputs_all = []
    for output_of_interest in outputs_indices:
            outputs = [0,0]
            samples = np.transpose([norm.rvs(n_samples) for norm in aleatory_rvs])
            optimisation_parameters = [sampled_function,aleatory_YN,output_of_interest]
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
                output_futures = [executor.submit(in_loop_optimise,sample,optimisation_parameters) for sample in samples]

            for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
                #print('Evaluated sample: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/max(np.shape(samples)),4))+'%)')
                pass
            concurrent.futures.wait(output_futures)

            for i_future, future in enumerate(output_futures):
                if future._exception: # Error handling
                    raise Exception('Error on result number {}: {}'.format(i_future,future.exception()))
                else:
                    outputs=np.vstack((outputs,future.result()))
            outputs = outputs[1:,:]

            outputs_all.append(outputs)
    return outputs_all

