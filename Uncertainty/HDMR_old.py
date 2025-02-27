# Based on the work of Dr Martin Kubicek
# Developed in Python by Dr Bryn Jones (thanks Bryn)
# Modifications by Tommy Williamson
import numpy as np
import pandas as pd
import itertools
import openturns as ot
import matplotlib.pyplot as plot
from scipy.interpolate import RBFInterpolator
import concurrent.futures
import pathlib
from openturns import viewer
from functools import partial
from copy import copy
import psutil
import glob, os
import warnings

class HDMR_ROM():

    def __init__(self,order,initial_data='read',read_data_filename='DataSave.csv',output_names=['output'],surrogate_type='kriging',subdomain_blacklist=[]):

        self.order = order
        self.database = initial_data
        self.output_names = output_names
        self.surrogate_type = surrogate_type
        self.subdomain_blacklist = subdomain_blacklist
        self.db_name = read_data_filename
        ot.Log.Show(ot.Log.NONE)

        if type(self.database) is str:
            if self.database=='read':
                self.database = pd.read_csv(self.db_name,header=None).to_numpy()

        ##########################################
        ### CLASSIFY THE DATA INTO SUB-DOMAINS ###
        ##########################################

        self.subdomains = []

        for current_max_order in reversed(range(self.order+1)):

            for i_sample in range(len(self.database[:,0])):

                current_subdimensions = self.classify_subdomain(self.database[i_sample,:-len(output_names)])

                if len(current_subdimensions)==current_max_order:

                    added = False
                    for i_subdomain in range(len(self.subdomains)):
                        if np.array_equal(current_subdimensions,self.subdomains[i_subdomain].subdimensions):
                            added_data = self.database[i_sample,:]
                            self.subdomains[i_subdomain].add_data(added_data)
                            added = True
                            continue
                        
                        match = True
                        for subdomain_ID in current_subdimensions:
                            if subdomain_ID not in self.subdomains[i_subdomain].subdimensions:
                                match = False
                                break

                        if match == False:
                            continue
                        else:
                            self.subdomains[i_subdomain].add_data(self.database[i_sample,:])

                    if not(added):
                        should_append = True
                        if self.subdomain_blacklist is not None:
                            for blacklisted_domain in self.subdomain_blacklist:
                                if np.any(blacklisted_domain==current_subdimensions) and not np.array_equal(blacklisted_domain,current_subdimensions):
                                    should_append=False
                        if should_append: 
                            self.subdomains.append(self.SubDomain(current_subdimensions,self,self.output_names,self.surrogate_type))
                            added_data = self.database[i_sample,:]
                            self.subdomains[-1].add_data(added_data)

        ################################
        ### BUILD THE TREE STRUCTURE ###
        ################################

        for current_max_order in range(self.order+1):

            for parental_subdomain in range(len(self.subdomains)):

                if not(len(self.subdomains[parental_subdomain].subdimensions)==current_max_order):
                    continue

                for check_subdomain in self.subdomains:

                    if not(len(self.subdomains[parental_subdomain].subdimensions) < (len(check_subdomain.subdimensions))):
                        continue

                    match = True
                    matched_subdomains = []
                    for subdomain_ID in self.subdomains[parental_subdomain].subdimensions:
                        if subdomain_ID not in check_subdomain.subdimensions:
                            match = False
                            break
                        else:
                            matched_subdomains.append(np.where(check_subdomain.subdimensions==subdomain_ID))

                    if match == False:
                        continue
                    
                    matched_subdomains = np.sort(np.asarray(matched_subdomains).flatten())

                    if parental_subdomain not in check_subdomain.parents:

                        check_subdomain.parents.append({'subdomain':parental_subdomain,'subdimensions':matched_subdomains})
                        if len(matched_subdomains) == 0:
                            matched_subdomains = [0]

                        for i_data_point in range(len(check_subdomain.data_dict['x'])):
                            current_row = np.asarray(check_subdomain.data_dict['x'][i_data_point])
                            if np.any(check_subdomain.data_dict['x'][i_data_point]==0.):
                                exact_samples = True
                            else:
                                exact_samples = False

                            for output_name in self.output_names:
                                check_subdomain.data_dict[output_name][i_data_point] -= self.subdomains[parental_subdomain].call_surrogate(current_row[matched_subdomains],output_name,exact_samples)
                            check_subdomain.is_datapoint_adjusted[i_data_point] = True

                        for output_name in self.output_names:
                            check_subdomain.data_dict[output_name] = np.asarray(check_subdomain.data_dict[output_name]).flatten()

        ### Collect children
        for check_subdomain in self.subdomains:
            if len(check_subdomain.subdimensions)==0: # 0th order subdomain has all others as children
                check_subdomain.children = [sub for sub in self.subdomains if sub is not check_subdomain]
                continue

            for candidate_child in self.subdomains:
                for childs_parent in candidate_child.parents[:]: 
                    if len(childs_parent['subdimensions'])==0: continue # Skip first parent as its 0th order
                    is_my_mummy = np.all(childs_parent['subdimensions'] == check_subdomain.subdimensions) # Are you my mummy?
                    if is_my_mummy: 
                        check_subdomain.children.append(self.subdomains.index(candidate_child))
                        break # Don't talk to me or my son ever again

        for subdomain in self.subdomains:
            for name in self.output_names:
                if name not in subdomain.surrogate.keys():
                    subdomain.create_surrogate(name)

    ############################
    ### KRIGING FIT FUNCTION ###
    ############################

    def fitKriging(self, coordinates, observations, covarianceModel, basis, subdomain, ground_truth):
        """
        Fit the parameters of a Kriging metamodel.
        """

        lower = 50.0
        upper = 1000.0

        # Define the Kriging algorithm.
        algo = ot.KrigingAlgorithm(coordinates, observations, covarianceModel, basis)
        #algo.setNoise([1e-12]*len(observations))

        # Set the optimization bounds for the scale parameter to sensible values
        # given the data set.
        scale_dimension = covarianceModel.getScale().getDimension()
        algo.setOptimizationBounds(ot.Interval([lower] * scale_dimension, [upper] * scale_dimension))

        # Run the Kriging algorithm and extract the fitted surrogate model.
        algo.run()
        krigingResult = algo.getResult()
        krigingMetamodel = krigingResult.getMetaModel()
        if ground_truth is not None:
            self.plot_kriging_in_progress(coordinates,observations,krigingResult,subdomain,ground_truth)
        return krigingResult, krigingMetamodel

    def plot_kriging_in_progress(self,coordinates,observations,result,subdomain, ground_truth):
        n_inputs = len(self.database[0,:]) - len(self.output_names)

        metamodel = result.getMetaModel()
        dim = metamodel.getParameterDimension()
        n_obs = observations.getSize()
        if dim>2 or n_obs<0: return None
        n_grids = np.floor(250**(1/dim))
        gridpoints = construct_ot_grid(dim=dim,n_points=n_grids )
        y_kriging = metamodel(gridpoints) #+ ground_truth(np.zeros(n_inputs))
        observations = np.array(observations)
        coordinates = np.array(coordinates)
        #observations += ground_truth(np.zeros(n_inputs))
        samplepoints = np.zeros((len(gridpoints),n_inputs))
        samplepoints[:,subdomain.subdimensions] = gridpoints
        if len(self.output_names)>1:
            y_ground_truth = np.reshape([ground_truth(sam)[-1] for sam in samplepoints],(-1,1))
        else: y_ground_truth = np.reshape([ground_truth(sam) for sam in samplepoints],(-1,1))

        for parent_data in subdomain.parents:
            parental_dimensions = parent_data['subdimensions']
            parent = self.subdomains[parent_data['subdomain']]
            for i_gt, gt in enumerate(y_ground_truth):
                datapoint_in_parent=np.zeros(n_inputs)
                if len(parental_dimensions)>0:
                    datapoint_in_parent[parental_dimensions]=samplepoints[i_gt,:][parental_dimensions]
                y_ground_truth[i_gt]-=ground_truth(datapoint_in_parent)
                #y_ground_truth[i_gt]+=ground_truth(np.zeros(n_inputs))

        sqrt = ot.SymbolicFunction(['x'],['sqrt(abs(x))'])
        #epsilon = ot.Sample(n_grids, [1.0e-8])
        conditionalVariance = result.getConditionalMarginalVariance(gridpoints) #+ epsilon
        conditionalSigma = sqrt(conditionalVariance)
        if dim==1: #use ot builtin for 1-D Kriging...
            graph = ot.Graph("Kriging in progress with {} observations".format(n_obs), "", "", True, "")
            kri = ot.Curve(gridpoints,y_kriging)
            kri.setLegend("Kriging")
            gt = ot.Curve(gridpoints,y_ground_truth)
            gt.setLegend("Ground truth")
            obs = ot.Cloud(coordinates,observations)
            obs.setLegend("Observations")
            boundsPoly = ot.Polygon.FillBetween(gridpoints,y_kriging-3*conditionalSigma, y_kriging+3*conditionalSigma)
            boundsPoly.setColor('#43ff6499')
            boundsPoly.setLegend('+/- 3 sigma')
            graph.add(boundsPoly)
            graph.add(kri)
            graph.add(obs)
            graph.add(gt)
            graph.setLegendPosition("upper right")
            view = viewer.View(graph)
        elif dim==2: # Must construct our own graph for 2-D plots
            x = np.array(gridpoints)[:,0]
            y = np.array(gridpoints)[:,1]
            z_ground_truth = np.array(y_ground_truth).flatten()
            z_kriging = np.array(y_kriging).flatten()
            z_sigma = np.array(conditionalSigma).flatten()
            z_sigma_plus = [kr + 3.0*z_sigma[i_kr] for i_kr, kr in enumerate(z_kriging)]
            z_sigma_minus = [kr - 3.0*z_sigma[i_kr] for i_kr, kr in enumerate(z_kriging)]
            sample_data = np.hstack((coordinates,observations))
            figure = plot.figure()
            ax = figure.add_subplot(projection='3d')
            ax.plot_trisurf(x,y,z_kriging,label='Kriging',color='b',alpha=0.5)
            #ax.plot_trisurf(x,y,z_ground_truth, label='Ground Truth',color='r',alpha=0.5)
            ax.plot_trisurf(x,y,z_sigma_plus, label='+/- 3.0 Sigma (so it fits)',color='g',alpha=0.25)
            ax.plot_trisurf(x,y,z_sigma_minus, label='_no_sigma_label',color='g',alpha=0.25)
            ax.scatter(sample_data[:,0],sample_data[:,1],sample_data[:,2],label='Observations',color='y',marker='*')
            ax.legend()
            figure.suptitle("Kriging in progress with {} observations".format(n_obs))
        plot.show()
    ###########################
    ### DEFINE A SUB-DOMAIN ###
    ###########################

    def classify_subdomain(self,input_vector):
        return input_vector.nonzero()[0]

    class SubDomain:
        def __init__(self,subdimensions,outer_class,output_names,surrogate_type='kriging'):
            self.subdimensions = subdimensions
            self.data_dict = {}
            self.data_dict['x'] = []
            self.output_names = output_names

            for output_name in output_names:
                self.data_dict[output_name] = []
            
            self.parents = []
            self.children = []
            self.surrogate = {}
            self.outer_class = outer_class
            self.surrogate_type = surrogate_type
            self.variance_lookup = {}
            self.is_datapoint_adjusted = []

        def add_data(self,data):

            data_shape = np.shape(data)

            if len(data_shape) == 1:
                self.append_data(data)
            elif len(data_shape) == 2:
                number_of_rows = data_shape[0]
                for a in range(number_of_rows):
                    self.append_data(data[a,:])
            else:
                raise ValueError('Subdomain data entry shape too long')

        def append_data(self,data_row):
            

            if len(self.data_dict['x']) == 0:
                self.data_dict['x'] = np.reshape(data_row[self.subdimensions],(1,len(self.subdimensions)))
                pointer = len(self.output_names)
                for output_name in self.output_names:
                    self.data_dict[output_name] = np.reshape(data_row[-pointer],(1,1))
                    pointer -= 1
            elif np.any(np.all(self.data_dict['x']==data_row[self.subdimensions],axis=1)): return
            else:
                self.data_dict['x'] = np.append(self.data_dict['x'],[data_row[self.subdimensions]],0)
                pointer = len(self.output_names)
                for output_name in self.output_names:
                    self.data_dict[output_name] = np.append(self.data_dict[output_name],data_row[-pointer])
                    pointer -= 1
            if len(self.subdimensions)>0: self.is_datapoint_adjusted.append(False)
            else: self.is_datapoint_adjusted.append(True)

        def subtract_parents(self,datapoint):
            datapoint_in_self = datapoint[self.subdimensions]
            i_dict = np.where(np.all(self.data_dict['x']==datapoint_in_self,axis=1))[0][0]
            if self.is_datapoint_adjusted[i_dict]: return
            for parent_data in self.parents:
                parental_dimensions = parent_data['subdimensions']
                parent = self.outer_class.subdomains[parent_data['subdomain']]
                for name in self.outer_class.output_names: parent.create_surrogate(name)
                datapoint_in_parent=self.data_dict['x'][i_dict][parental_dimensions] if len(parental_dimensions)>0 else 0.0
                for output_name in self.outer_class.output_names: self.data_dict[output_name][i_dict] -= parent.call_surrogate(datapoint_in_parent,output_name)[0][0]
            self.is_datapoint_adjusted[i_dict]=True
                #checked_domains.append(self.outer_class.subdomains.index(parent_data['subdomain']))
            #return checked_domains

        def call_surrogate(self,x,output_name,return_exact_samples=True):

            x = np.asarray(x)

            if output_name not in self.surrogate.keys():
                self.create_surrogate(output_name)

            if return_exact_samples:
                random_perturbation = 0.
            else:
                random_perturbation = 1.23456789e-12

            while len(np.shape(x)) < 2:
                x = np.asarray([x])

            if (self.surrogate_type == 'kriging') or self.surrogate_type == 'no_surrogate':

                return self.surrogate[output_name](ot.Sample(x+random_perturbation))
            
            elif self.surrogate_type == 'RBF':
                return self.surrogate[output_name](x+random_perturbation)

            else:

                raise ValueError("No valid surrogate selected for subdomain model")

        def create_surrogate(self,output_name,ground_truth=None):
            if self.surrogate_type == 'kriging':
                dimension = len(self.subdimensions)
                if dimension == 0:
                    self.surrogate[output_name] = partial(invariant_func,ot.Sample(self.data_dict[output_name]))
                else:

                    x_train = np.asarray(self.data_dict['x'])
                    y_train = np.asarray(self.data_dict[output_name])

                    nan_entries = np.isnan(y_train)

                    x_train = np.delete(x_train,nan_entries,axis=0)
                    y_train = np.delete(y_train,nan_entries)

                    x_train = [x for x in x_train]
                    y_train = [[x] for x in y_train]

                    x_train = ot.Sample(x_train)
                    y_train = ot.Sample(y_train)

                    basis = ot.ConstantBasisFactory(dimension).build()
                    #covarianceModel = ot.GeneralizedExponential(np.ones((dimension))*1.e-3,2-1e-8)
                    covarianceModel = ot.MaternModel([1.0] * dimension, 1.5)
                    krigingResult, krigingMetamodel = self.outer_class.fitKriging(x_train, y_train, covarianceModel, basis,self,ground_truth)
                    self.surrogate[output_name] = krigingMetamodel
                    fuzz = np.random.rand()*1e-3
                    self.variance_lookup[output_name] = construct_ot_grid(len(self.subdimensions),n_points=int(100/len(self.subdimensions)),bounds=[-1+fuzz,1])
                    self.variance_lookup[output_name] = np.hstack((self.variance_lookup[output_name],krigingResult.getConditionalMarginalVariance(self.variance_lookup[output_name])))

            elif self.surrogate_type == 'no_surrogate':
                self.surrogate[output_name] = partial(invariant_func,ot.Sample([[np.squeeze(self.data_dict[output_name][0]).tolist()]]))#self.data_dict[output_name][0])

            elif self.surrogate_type == 'RBF':
                
                dimension = len(self.subdimensions)
                if dimension == 0:
                    self.surrogate[output_name] = partial(invariant_func,ot.Sample(self.data_dict[output_name]))
                else:
                    rbf_surrogate = RBFInterpolator(self.data_dict['x'],self.data_dict[output_name])
                    #self.surrogate[output_name] = lambda x: ot.Sample([rbf_surrogate(x).flatten().tolist()])

            else:

                raise ValueError("No valid surrogate selected for subdomain model")
            
        def select_uncertain_points(self, n_points, output_name='',ground_truth=None):
            dim = len(self.subdimensions)
            points = np.zeros([n_points,dim])

            variance_checker = self.variance_lookup[output_name]

            metamodel_checker = self.surrogate[output_name]

            y_existing_points = np.asarray(self.data_dict[output_name])

            nan_indices = np.isnan(y_existing_points)

            x_isting_points = np.reshape(np.delete(np.asarray(self.data_dict['x']),nan_indices,axis=0),[-1,dim])
            y_existing_points = np.reshape(np.delete(y_existing_points,nan_indices),[-1,len(self.output_names)])

            basis = ot.ConstantBasisFactory(dim).build()
            covarianceModel = ot.MaternModel([1.0] * dim, 1.5)

            candidate = variance_checker[np.argsort(variance_checker[:,-1])[-1],:-1]
            variance = variance_checker[np.argsort(variance_checker[:,-1])[-1],-1]
            for i_point in range(n_points):
                while candidate in points:
                    
                    prediction_at_candidate = metamodel_checker(candidate)[0]
                    x_isting_points = np.vstack((x_isting_points,candidate))
                    y_existing_points = np.vstack((y_existing_points,prediction_at_candidate))

                    adapted_kriging_result, metamodel_checker = self.outer_class.fitKriging(ot.Sample(x_isting_points), ot.Sample(y_existing_points), covarianceModel, basis,self.subdimensions,ground_truth)
                    fuzz = np.random.rand()*1e-3
                    variance_grid = construct_ot_grid(len(self.subdimensions),n_points=10*n_points,bounds=[-1+fuzz,1])
                    variance_checker = np.hstack((variance_grid,adapted_kriging_result.getConditionalMarginalVariance(variance_grid)))

                    candidate = variance_checker[np.argsort(variance_checker[:,-1])[-1],:-1]
                    variance = variance_checker[np.argsort(variance_checker[:,-1])[-1],-1]
                points[i_point] = candidate
            return points, variance





    ###############################
    ### DEFINE GLOBAL SURROGATE ###
    ###############################

    def call_global_surrogate(self,x,output_name,max_order=np.inf):

        x_shape = np.shape(x)

        if len(x_shape) == 1:
            x = np.asarray([x])
            x_shape = np.shape(x)

        output = 0.

        for a in range(x_shape[0]):
            string = ''
            for subdomain in self.subdomains:
                if len(subdomain.subdimensions) > max_order:
                    continue
                contribution = subdomain.call_surrogate(x[a,subdomain.subdimensions],output_name)[0][0]
                output += contribution
                string+= ' +'+ str(contribution) if contribution>=0 else ' ' + str(contribution)
            #print(string + ' = {}'.format(output))

        return output

class HDMR_sampling():

    def __init__(self,the_function,max_domain_order,number_of_parameters,number_of_outputs=1,parallel=False,num_cores='auto',load_database=True,database_name='DataSave.csv',blacklist=[]):

        self.the_function = the_function
        self.max_domain_order = max_domain_order
        self.number_of_parameters = number_of_parameters
        self.parallel = parallel
        self.number_of_outputs = number_of_outputs
        self.db_name = database_name
        self.subdomain_blacklist = blacklist
        if parallel:
            self.n_procs = psutil.cpu_count(logical=False) if num_cores=='auto' else num_cores

        if not load_database: pathlib.Path(self.db_name).unlink()
        try:
            self.database = pd.read_csv(self.db_name,header=None).to_numpy()

            if len(np.shape(self.database))==1:
                self.database = np.reshape(self.database,(1,len(self.database)))

        except (FileNotFoundError,OSError):
            print('No existing file \'{}\' found, initialising from scratch...'.format(self.db_name))
            self.database = np.zeros((1,number_of_parameters+self.number_of_outputs))
            initial_DOE = np.zeros((1,number_of_parameters))
            self.database[0,number_of_parameters:] = np.squeeze(self.the_function(initial_DOE[0,:]))
            np.savetxt(self.db_name,self.database,delimiter=',')

        initial_DOE = np.zeros((1,number_of_parameters))

        for order in range(1,max_domain_order+1):

            combinations = list(itertools.combinations(range(number_of_parameters), order)) 
            
            for combination in combinations:
                do_DoE = True
                for blacklisted_domain in self.subdomain_blacklist: # Make sure we're not sampling blacklisted domains
                    if np.all([bl in combination for bl in blacklisted_domain]) and not np.array_equal(blacklisted_domain,combination):
                        do_DoE = False

                if do_DoE:
                    new_rows = np.full((1,number_of_parameters),0.)
                    
                    for index in combination:
                        new_rows[0,index] = -1.

                    for index in combination:
                        new_rows = np.append(new_rows,new_rows,0)
                        new_rows[:int(np.shape(new_rows)[0]/2),index] = 1.

                    initial_DOE = np.append(initial_DOE,new_rows,0) # This results in a full factorial up to specified order

        self.parallel_generate(initial_DOE[1:,:]) if self.parallel else self.generate_samples(initial_DOE[1:,:])

    def generate_samples(self,sample_DOE):
        for row in sample_DOE:
            if not np.any(np.all((self.database[:,:-self.number_of_outputs] == row),axis=1)): # Only continue if line does not exist in db
                self.database = np.append(self.database,np.reshape(np.append(row,self.the_function(row)),(1,len(row)+self.number_of_outputs)),0)
                np.savetxt(self.db_name,self.database,delimiter=',')

    def parallel_generate(self,sample_DOE):
        # lock_files = glob.glob('*_HDMR.lock')
        # for lock_file in lock_files: os.remove(lock_file)
        pruned_DOE = np.zeros((1,self.number_of_parameters))
        outputs = np.zeros((1,self.number_of_outputs))
        for row in sample_DOE:
            row = [row.flatten()]
            if not np.any(np.all((self.database[:,:-self.number_of_outputs] == row),axis=1)): pruned_DOE=np.vstack((pruned_DOE,row))
        sample_DOE = pruned_DOE[1:,:]
        n_workers = np.shape(sample_DOE)[0] if np.shape(sample_DOE)[0] < self.n_procs else self.n_procs
        n_workers = 1 if n_workers<1 else n_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            output_futures = [executor.submit(self.the_function,row) for row in sample_DOE]

        for i_sim, f in enumerate(concurrent.futures.as_completed(output_futures)):
            print('Evaluated sample: '+str(i_sim+1)+' ('+str(round(100*(i_sim+1)/max(np.shape(sample_DOE)),4))+'%)')
            # outputs=np.vstack((outputs,f.result()))
        concurrent.futures.wait(output_futures)

        for i_future, future in enumerate(output_futures):
            if future._exception: # Error handling
                raise Exception('Error on result number {}: {}'.format(i_future,future.exception()))
            else:
                outputs=np.vstack((outputs,future.result()))
        outputs = outputs[1:,:]
        # Stacking in many different directions, I know it's hideous but it should be robust 
        self.database=np.vstack((self.database,np.hstack((sample_DOE,outputs)))) 
        np.savetxt(self.db_name,self.database,delimiter=',')

    def select_enrichment_target(self,rom,force_order=0,use_kriging_var=True,stochastic=False, n_subdomains_to_pick=1):
        combined_subdomain_fitness, combined_subdomain_data_sizes, _ = self.combined_datametric_s(rom, force_order, use_kriging_var)

        combined_subdomain_data_sizes = np.asarray(combined_subdomain_data_sizes)
        small_subdomains = combined_subdomain_data_sizes<=2
        selected_subdimensions = []

        for i_subdomain in range(n_subdomains_to_pick):
            if np.sum(small_subdomains)>0:
                print('Enriching small subdomains, remaining {}...'.format(np.sum(small_subdomains)))
                subdomain_candidates = np.argwhere(small_subdomains)
                choice = np.random.randint(len(subdomain_candidates)) if stochastic else 0
                selected_subdimensions.append(rom.subdomains[subdomain_candidates[choice][0]].subdimensions)
                small_subdomains[subdomain_candidates[choice]]=False

            else:
                combined_subdomain_fitness*=1/np.sum(combined_subdomain_fitness)
                pointer=0
                if stochastic:
                    geneseed = np.random.rand()
                    pointer = 0
                    cumsum = 0.
                    while cumsum<geneseed:
                        cumsum += combined_subdomain_fitness[pointer]
                        pointer += 1
                    pointer-=1
                else:
                    pointer = np.argmax(combined_subdomain_fitness)

                selected_subdimensions.append(rom.subdomains[pointer].subdimensions)
                combined_subdomain_fitness[pointer] *= 0.5
        return selected_subdimensions

    def combined_data_metrics(self, rom, force_order, use_kriging_var):
        output_names = rom.subdomains[-1].output_names
        number_of_subdomains = len(rom.subdomains)

        subdomain_fitness = {}
        subdomain_data_sizes = {}
        subdomain_vsrr = {} # Variance to squared range ratio 

        for i_name, name in enumerate(output_names):
            subdomain_fitness[name] = []
            subdomain_data_sizes[name] = []
            subdomain_vsrr[name] = []
            overall_range = np.max(self.database[:,self.number_of_parameters+i_name])-np.min(self.database[:,self.number_of_parameters+i_name])

            for subdomain in rom.subdomains:
                if (force_order==0 or len(subdomain.subdimensions)==force_order) and len(subdomain.subdimensions) != 0:
                    which_data = ~np.any(subdomain.data_dict['x']==0,1)
                    not_nan_data = subdomain.data_dict[name][which_data]
                    not_nan_data = not_nan_data[~np.isnan(not_nan_data)]
                    subdomain_data_sizes[name].append(len(not_nan_data))
                    if use_kriging_var: subdomain_fitness[name].append(np.max(subdomain.variance_lookup[name]))

                    else:
                        subdomain_fitness[name].append(np.var(not_nan_data)/(len(which_data)**2))
                        subdomain_vsrr[name].append(np.var(not_nan_data)/(overall_range**2))
                    
                else:
                    subdomain_fitness[name].append(0.0)
                    subdomain_data_sizes[name].append(np.inf)
                    subdomain_vsrr[name].append(0.0)

            subdomain_fitness[name] = (subdomain_fitness[name]-np.min(subdomain_fitness[name]))/(np.max(subdomain_fitness[name])-np.min(subdomain_fitness[name]))
            subdomain_data_sizes[name] = np.asarray(subdomain_data_sizes[name])

        combined_subdomain_fitness = np.zeros((number_of_subdomains))
        combined_subdomain_vsrr = np.zeros((number_of_subdomains))

        for name in output_names:
            subdomain_fitness[name] = np.asarray(subdomain_fitness[name])
            subdomain_fitness[name] /= np.sum(subdomain_fitness[name])
            combined_subdomain_fitness += subdomain_fitness[name]

            # subdomain_vsrr[name] = np.asarray(subdomain_vsrr[name])
            # subdomain_vsrr[name] /= np.sum(subdomain_vsrr[name])
            combined_subdomain_vsrr += subdomain_vsrr[name]
        
        combined_subdomain_fitness /= len(output_names)
        combined_subdomain_vsrr /= len(output_names)
        


        combined_subdomain_data_sizes = []
        for a in range(number_of_subdomains):
            min_subdomain_size = np.inf
            for name in output_names:
                if subdomain_data_sizes[name][a] < min_subdomain_size:
                    min_subdomain_size = subdomain_data_sizes[name][a]
            combined_subdomain_data_sizes.append(min_subdomain_size)
        return combined_subdomain_fitness,combined_subdomain_data_sizes, combined_subdomain_vsrr
    
    def enrich(self,rom,force_order=0,samples_per_enrichment = 1,output_name='',sample=None,list_of_subdimensions=None,ground_truth=None,method='combined'):
        if method != 'combined': ## Deprecated sampling method
            if list_of_subdimensions is None:
                print('Subdomain selection...')
                list_of_subdimensions = self.select_enrichment_target(rom,force_order,use_kriging_var=True, n_subdomains_to_pick=samples_per_enrichment,stochastic=False)

            subdomains =[]
            count = []
            for sub_dim in list_of_subdimensions:
                for candidate_subdomain in rom.subdomains:
                    matching_dims = candidate_subdomain.subdimensions==sub_dim
                    should_add = np.all(matching_dims)
                    if should_add and candidate_subdomain not in subdomains: 
                        subdomains.append(candidate_subdomain)
                        count.append(1)
                        break
                    elif should_add:
                        count[subdomains.index(candidate_subdomain)]+=1
                        break

            if sample is None:
                print('Sample selection...')
                for i_subd, sub in enumerate(subdomains):
                    sample = sub.select_uncertain_points(count[i_subd], output_name=output_name,ground_truth=ground_truth)
        else:
            sample, subdomains, sampled_domains = self.variance_GP_selection(rom, samples_per_enrichment, output_name, list_of_subdimensions)
        predictions ={}
        epsilons = {}

        for name in rom.output_names:

            predictions[name] = [rom.call_global_surrogate(row,name) for row in sample]
            epsilons[name] =[]
            self.parallel_generate(sample) if self.parallel else self.generate_samples(sample)
        
        for i_subd, sampled_subdomain in enumerate(sampled_domains):
            sampled_subdomain.add_data(self.database[-samples_per_enrichment+i_subd,:])
            sampled_subdomain.subtract_parents(self.database[-samples_per_enrichment+i_subd,:-self.number_of_outputs])
            for name in rom.output_names: sampled_subdomain.create_surrogate(name,ground_truth)
            
            
            for i_child in sampled_subdomain.children:
                    child_subdomain = rom.subdomains[i_child]
                    child_subdomain.add_data(self.database[-samples_per_enrichment+i_subd,:])
                    child_subdomain.subtract_parents(self.database[-samples_per_enrichment+i_subd,:-self.number_of_outputs])
                    for name in rom.output_names: child_subdomain.create_surrogate(name,ground_truth)

            for i_name, name in enumerate(rom.output_names):
                #print('Data min of {}, max of {}'.format(min(sampled_subdomain.data_dict[name]),max(sampled_subdomain.data_dict[name])))
                    #print('Data min of {}, max of {}'.format(min(child_subdomain.data_dict[name]),max(child_subdomain.data_dict[name])))

                new_data = self.database[-samples_per_enrichment+i_subd,-len(rom.output_names)+i_name]
                current_prediction = predictions[name][i_subd]
                pct_error = 100*abs(current_prediction-new_data)/new_data
                #print('Ground truth of {}, predicted {} ({}% error)'.format(new_data,current_prediction,pct_error))
                epsilons[name].append(pct_error)
            
        for subd in rom.subdomains:
            if not np.all(subd.is_datapoint_adjusted): warnings.warn('Uh oh subdomain {} has unadjusted datapoints!'.format(sampled_subdomain.subdimensions))
        return sample,subdomains,epsilons

    def variance_GP_selection(self, rom, samples_per_enrichment, output_name, list_of_subdimensions=None):
        # This function using the variance of subdomain GPs (kriging) to select optimal subdomains and samples from those subdomains
        if list_of_subdimensions is None: 
            subdomains =[]
            for subd in rom.subdomains:
                if len(subd.subdimensions)>0: subdomains.append(subd)
        else: # Ugly code repeat, but better for readability than making a new function imo
            subdomains =[]
            for sub_dim in list_of_subdimensions:
                for candidate_subdomain in rom.subdomains:
                    if len(candidate_subdomain.subdimensions)>0: # Did you know an np.all([])=True? I didn't!
                        matching_dims = candidate_subdomain.subdimensions==sub_dim
                        should_add = np.all(matching_dims)
                        if should_add and candidate_subdomain not in subdomains: 
                            subdomains.append(candidate_subdomain)
                            break

            # This little dohickey iterates over a set of subdomains and allocates samples based upon highest variance across all subdomains
        samples_per_subdomain = np.zeros(len(subdomains),dtype=np.int8)
        old_sample_count = -1*np.ones(len(subdomains))
        subdomain_max_variance = np.zeros(len(subdomains)) 
            
        while np.sum(samples_per_subdomain)<samples_per_enrichment: # Iterate until we reach max samples
            for i_subd, count in enumerate(samples_per_subdomain): 
                if not count == old_sample_count[i_subd]: # Note that variances from select_points are predicted as if new data points were added
                    subdomain_max_variance[i_subd] = subdomains[i_subd].select_uncertain_points(count+1, output_name)[1]
            old_sample_count = copy(samples_per_subdomain)
            samples_per_subdomain[np.argmax(subdomain_max_variance)]+=1
            
        sample = np.zeros(self.number_of_parameters)
        list_of_sampled_domains =[]
        for i_subd, count in enumerate(samples_per_subdomain):
            new_row = np.zeros([count,self.number_of_parameters])
            new_row[:,subdomains[i_subd].subdimensions] = subdomains[i_subd].select_uncertain_points(count, output_name)[0]
            sample = np.vstack((sample,new_row))
            for _ in range(count): list_of_sampled_domains.append(subdomains[i_subd])
        sample = sample[1:,:]
        return sample,subdomains,list_of_sampled_domains

    def all_subdomains_at_point(self,the_point):

        def combinatorics_wrapper(number_of_parameters=self.number_of_parameters):

            def recursive_combinatorics(list_one_combo,list_two_combo):
                output_list_of_combinations = []
                for a in range(len(list_one_combo)):
                    for b in range(a+1,len(list_two_combo)):
                        output_list_of_combinations.append(np.append(list_one_combo[a],list_two_combo[b]))
                        recursive_combinations = recursive_combinatorics([output_list_of_combinations[-1]],list_two_combo[b:])
                        for recursive_combination in recursive_combinations:
                            output_list_of_combinations.append(recursive_combination)

                return output_list_of_combinations

            the_domains = np.arange(number_of_parameters)
            combined_domains = recursive_combinatorics(the_domains,the_domains)
            for domain in the_domains:
                combined_domains.append(domain)
            combined_domains.append([])

            return combined_domains

        combinations = combinatorics_wrapper()

        for row in the_point:
            for combination in combinations:
                the_sample = row.copy()
                the_sample[combination] = 0.
                self.generate_samples(the_sample[None,:])

    def increase_order(self,rom,cull_number=0,n_sigma_tolerance=0,percentage_tolerance=0.0):
        # Can cull either by selecting the n subdomains wth the least variance (by setting cull_number=n) or can set a tolerance level of
        # n_sigma, which corresponds to a VSRR (Variance to Squared Range Ratio) of 1/(2*n_sigma)^2.
        # Can also specify a percentage which corresponds to the allowable population not captured by the samples
        # Note that since the mean is always very close to 0 we can't use Coefficient of Variation
        if cull_number>0 or n_sigma_tolerance>0 or percentage_tolerance>0:
            vsrr_threshold = 1/(2*n_sigma_tolerance)**2 if n_sigma_tolerance>0 else 0.0
            _, _, combined_data_vsrr = self.combined_data_metrics(rom=rom, force_order=0, use_kriging_var=False)
            while cull_number>0:
                minvsrr = np.inf
                cull_index = None
                for i_subd, vsrr in enumerate(combined_data_vsrr):
                    subdims = rom.subdomains[i_subd].subdimensions # The 2nd place award for clunkiest condition goes to...
                    if len(subdims)==rom.order and vsrr<minvsrr and not np.any([np.all(subdims==bl) for bl in self.subdomain_blacklist]):
                        minvsrr=vsrr
                        cull_index = i_subd
                self.subdomain_blacklist.append(rom.subdomains[cull_index].subdimensions)
                cull_number-=1
            for i_subd, vsrr in enumerate(combined_data_vsrr):
                subdims = rom.subdomains[i_subd].subdimensions # And now the main event...
                if (vsrr<vsrr_threshold or vsrr<0.01*percentage_tolerance) and not np.any([np.all(subdims==bl) for bl in self.subdomain_blacklist]) and len(subdims)==rom.order:
                    self.subdomain_blacklist.append(subdims)
                    
        if not os.path.exists(self.db_name): np.savetxt(self.db_name,self.database,delimiter=',')
        new_sampler = HDMR_sampling(self.the_function,rom.order+1,self.number_of_parameters,number_of_outputs=self.number_of_outputs,
                                parallel=self.parallel,num_cores=self.n_procs,load_database=True,database_name=self.db_name,
                                blacklist=self.subdomain_blacklist)
        new_rom = HDMR_ROM(order=rom.order+1,read_data_filename=self.db_name,output_names=rom.output_names, surrogate_type=rom.surrogate_type, 
                       subdomain_blacklist=self.subdomain_blacklist)
        return new_rom, new_sampler
class HDMR_postprocessing():

    def __init__(self,the_rom,output_name='output',input_names=None,):

        self.the_rom = the_rom
        self.output_name = output_name
        self.input_names = input_names

    def plot_subdomains(self,orders={1,2},plot_deltas=False):

        subdomain_fitness = []
        for subdomain in self.the_rom.subdomains:
            which_data = ~np.any(subdomain.data_dict['x']==0,1)
            not_nan_data = subdomain.data_dict[self.output_name][which_data]
            not_nan_data = not_nan_data[~np.isnan(not_nan_data)]
            subdomain_fitness.append(np.var(not_nan_data))

        subdomain_order = np.argsort(-np.asarray(subdomain_fitness))

        for which_subdomain in self.the_rom.subdomains:

            if len(which_subdomain.subdimensions) == 0:
                central_values = which_subdomain.call_surrogate([],self.output_name)[0]
                break

        for which_subdomain in np.asarray(self.the_rom.subdomains)[subdomain_order]:

            number_of_dimensions = len(which_subdomain.subdimensions)

            if number_of_dimensions not in orders:
                continue

            number_of_points = 500

            the_linspace = np.linspace(-1.,1.,number_of_points)

            if number_of_dimensions == 1:

                x = the_linspace
                y = np.squeeze(which_subdomain.call_surrogate(np.reshape(x,(number_of_points,1)),self.output_name))

                if plot_deltas==False:
                    delta = central_values
                    prefix = ''
                else:
                    delta = 0.
                    prefix = 'Δ'

                plot.plot(x,y+delta,marker=None, color='b',label='HDMR Surrogate')
                plot.scatter(which_subdomain.data_dict['x'],which_subdomain.data_dict[self.output_name]+delta,color='k',label='Samples')
                plot.legend(loc="upper left")
                if self.input_names is None:
                    plot.xlabel('Input '+str(which_subdomain.subdimensions[0]))
                else:
                    plot.xlabel(self.input_names[which_subdomain.subdimensions[0]])
                plot.ylabel(prefix+self.output_name)

                plot.show()

            elif number_of_dimensions == 2:

                if plot_deltas==False:
                    delta = central_values
                    prefix = ''
                else:
                    delta = 0.
                    prefix = 'Δ'

                x, y = np.meshgrid(the_linspace,the_linspace)
                z = np.zeros((number_of_points,number_of_points))

                for a in range(number_of_points):
                    for b in range(number_of_points):
                        z[a,b] = which_subdomain.call_surrogate(np.asarray([[x[a,b],y[a,b]]]),self.output_name)[0][0]

                sample_x = np.asarray(which_subdomain.data_dict['x'])

                figure = plot.figure()
                ax = figure.add_subplot(projection='3d')
                ax.plot_wireframe(np.reshape(x,(500,500)),np.reshape(y,(500,500)),z+delta,label='Surrogate')
                ax.scatter(sample_x[:,0],sample_x[:,1],which_subdomain.data_dict[self.output_name]+delta,color='r',label='Samples')
                if self.input_names is None:
                    ax.set_xlabel('Input '+str(which_subdomain.subdimensions[0]))
                    ax.set_ylabel('Input '+str(which_subdomain.subdimensions[1]))
                else:
                    ax.set_xlabel(self.input_names[which_subdomain.subdimensions[0]])
                    ax.set_ylabel(self.input_names[which_subdomain.subdimensions[1]])
                ax.set_zlabel(prefix+self.output_name)
                plot.show()


def construct_ot_grid(dim = 1,n_points = 50,bounds = [-1,1]):
    xmin = bounds[0]
    xmax = bounds[1]
    if dim==1:
        step = (xmax - xmin) / (n_points - 1)
        grid = ot.RegularGrid(xmin, step, int(n_points))
        gridpoints = grid.getVertices()
    elif dim>1:
        interval = [int(n_points) for _ in range(dim)]
        grid = ot.IntervalMesher(interval)
        mins = [xmin for _ in range(dim)]
        maxs = [xmax for _ in range(dim)]
        gridpoints = grid.build(ot.Interval(mins,maxs)).getVertices()
    return gridpoints

def invariant_func(returner, x): # Neede
    return returner

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