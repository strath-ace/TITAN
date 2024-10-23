import configparser
import os
import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import concurrent.futures
from argparse import ArgumentParser, RawTextHelpFormatter, BooleanOptionalAction
import datetime as dt
from Uncertainty.uncertainty import uncertaintySupervisor, extractQoI, plot_demise
from Uncertainty.dynamics_tools import deterministicImpulse
import sys
from TITAN import load_and_run_cfg as runTITAN
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import pickle
import pathlib
from Configuration.configuration import Uncertainty

from matplotlib import pyplot as plt
from messaging import messenger

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def wrapper(i_sim, params):

    base_cfg, UQ, oracle, verbose, keep = params

    cfg = build_cfg(base_cfg, UQ, verbose)
    pid = os.getpid()
    base_dir = cfg['Options']['Output_folder']
    directory = base_dir + '/Sample_PID_'+str(pid)
    cfg.set('Options','Output_folder',directory)
    cleanup=False
    loadOracleData(directory,oracle)
    if len(next(os.walk(base_dir))[1])>keep:
        cleanup = True
        cfg.set('Options','Write_solutions','False')

    if verbose: runTITAN(cfg,'')
    else:
        with HiddenPrints(): runTITAN(cfg,'')
    with open(directory+'/Data/QoI.pkl','rb') as file: quants = pickle.load(file)
    if cleanup: shutil.rmtree(directory)
    return quants, UQ.inputVector

def createOracle(cfg):
    print('Generating Oracle...')
    if os.path.exists('TempOracle'): shutil.rmtree('TempOracle')
    os.mkdir('TempOracle')

    oracle_cfg = configparser.ConfigParser()
    oracle_cfg.read_dict(cfg)
    oracle_cfg.set('Options', 'Output_folder', 'TempOracle')
    oracle_cfg.set('Options','Load_mesh','False')
    oracle_cfg.set('Options','Fidelity','Low')

    with open(args.yamlfile,'r') as file: uncertainties = yaml.load(file,Loader)
    if 'deorbit' in uncertainties:
        burndata = uncertainties['deorbit'].copy()
        deterministicImpulse(oracle_cfg,burndata)


    if not cfg.getboolean('GRAM', 'Uncertain'):
        print('...pregenerating mesh')
        oracle_cfg.set('Options', 'num_iters', '2')
    else:
        print('...uncertain GRAM selected, running full sim')
        oracle_cfg.set('GRAM','Uncertain','False')


    runTITAN(oracle_cfg,'')
    runTITAN(oracle_cfg,'wind')
    # mass = extractQoI(oracle_cfg,'TempOracle/Data/data.csv')[0]


    # if 'deorbit' in uncertainties:
    #     uncertainties['deorbit']['vehicle_mass']=float(mass)

    # with open(args.yamlfile, 'w') as file: yaml.dump(uncertainties,file)

def loadOracleData(directory,oracledirectory):
    subfolders =['Data','GRAM','Restart']

    if os.path.exists(directory): shutil.rmtree(directory)

    os.mkdir(directory)
    [os.mkdir(directory + '/' + sub) for sub in subfolders]
    [copy_tree(oracledirectory+ '/' + sub, directory + '/' + sub) for sub in subfolders]


def sampler(cfg,yamlfile,oracle='',n_procs=multiprocessing.cpu_count(),verbose=False,keep=0,seed='Auto',overrideUQ = None):
    msg = messenger(threshold=300)

    if oracle=='':
        createOracle(cfg)
        oracle='TempOracle'
    cfg.set('Options','Load_mesh','True')

    if os.path.exists('QoI.pkl'):
        qoi = pathlib.Path('QoI.pkl')
        qoi.unlink()

    if not os.path.exists(cfg['Options']['Output_folder']): os.mkdir(cfg['Options']['Output_folder'])

    if overrideUQ: UQ = overrideUQ
    else:
        UQ = uncertaintySupervisor(isActive=True,rngSeed=seed)
        UQ.constructInputs(yamlfile)

    outputs = []
    n_samples = cfg.getint('Options','Num_runs')

    params = [cfg, UQ, oracle, verbose, keep]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
        output_futures = [executor.submit(wrapper,i_sample,params) for i_sample in range(n_samples)]

        for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
            msg.print_n_send('Finished sim: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/n_samples,4))+'%)')
        concurrent.futures.wait(output_futures)

    # with open('QoI.pkl', 'rb') as file: qoi=pickle.load(file)

    qoi = Uncertainty()
    qoi.objects = cfg['QoI']['Objects']
    qoi.outputs = cfg['QoI']['Outputs']
    qoi.build_quantities(cfg['Assembly']['Path'])
    
    inputs = []
    for i_future, future in enumerate(output_futures):
        try: 
            for obj, output in future.result()[0].items():
                for name, value in output.items():
                    qoi.quantities[obj][name] = np.concatenate((qoi.quantities[obj][name],value))
            inputs=future.result()[1] if len(inputs)==0 else np.vstack((inputs,future.result()[1]))
        except Exception as e: 
            msg.print_n_send('Error on result number {}: {}'.format(i_future,e))
    
    if verbose: 
        print('Plotting your demise...')
        plot_demise(qoi.quantities,qoi.outputs,qoi.objects)
    
    return qoi, inputs

def build_cfg(cfg, UQ, verbose):
    new_cfg = configparser.ConfigParser()
    new_cfg.read_dict(cfg)
    if not verbose: 
        with HiddenPrints(): new_cfg=UQ.sampleConfig(new_cfg)
    else: 
         new_cfg=UQ.sampleConfig(new_cfg)
    return new_cfg

if __name__ == "__main__":
    # To run TITAN, it requires the user to specify a configuration
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("-N", "--num",
                        dest="n_processes",
                        type=int,
                        help="number of processes to run, leaving unspecified will attempt to use all available processes",
                        metavar="n")
    parser.add_argument("-c", "--config",
                        dest="cfgfile",
                        type=str,
                        help="filename of base config file",
                        metavar="cfg")
    parser.add_argument("-u", "--uqfile",
                        dest="yamlfile",
                        type=str,
                        help="filename of .yaml uncertainty file",
                        metavar="yaml")
    parser.add_argument("-O", "--oracle",
                        dest="oracle",
                        type=str,
                        help="path to \"Oracle\" folder containing prior TITAN run, necessary to run uncertain GRAM,if needed one will be generated when not specified",
                        metavar="path")
    parser.add_argument("-p", "--print",
                        dest="show_prints",
                        action=BooleanOptionalAction,
                        help="bool to toggle whether TITAN processes print text to terminal")
    parser.add_argument("-K", "--keep-data",
                        dest="keep",
                        type=int,
                        help="number of processes to keep data from, leave blank to keep all")

    args=parser.parse_args()

    if args.show_prints: print('Running in verbose mode...')

    if not args.cfgfile or not args.yamlfile: raise Exception('The user needs to provide a reference .cfg and a .yaml uncertainty file.\n')

    cfg = configparser.ConfigParser()
    cfg.read(args.cfgfile)

    oracle = '' if not args.oracle else args.oracle

    n_procs = multiprocessing.cpu_count() if not args.n_processes else args.n_processes

    keep = n_procs if not args.keep else args.keep

    outputs, _ = sampler(cfg=cfg,yamlfile=args.yamlfile,oracle=oracle,n_procs=n_procs,verbose=args.show_prints,keep=keep,seed='Auto')


