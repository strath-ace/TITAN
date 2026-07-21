import numpy as np
import mutationpp as mpp
import pathlib, cea
from itertools import combinations
import multiprocessing
class Byproducts():
    """Class to handle byproducts generation of assembly using Mutation++
    """

    def __init__(self, n_faces : int, is_emitting=False, rho = {}, mf = {}, mass = {}, species = [], n_free_species=5, cutoff=1e-6):
        """

        Args:
            n_faces (int): Number of facets of the assembly mesh
            is_emitting (bool, optional): Whether the assembly is currently generating emissions. Defaults to False.
            rho (dict, optional): Per-facet ablated species densities. Defaults to {}.
            mf (dict, optional): Per-facet ablated species mass fractions. Defaults to {}.
            mass (dict, optional): Per-facet ablated species masses. Defaults to {}.
            species (list, optional): Per-facet ablated species emissions (kg/km). Defaults to [].
            n_free_species (int, optional): Number of species in freestream mixture. Defaults to 5 (for air_5).
            cutoff (float, optional): Mass fraction cutoff for reporting byproducts, values below this will be treated as 0. Defaults to 1e-6.
        """
        #: [bool] Whether the assembly is currently generating emissions
        self.is_emitting = is_emitting

        #: [Dict] Per-facet ablated species densities
        self.rho = {}

        #: [Dict] Per-facet ablated species mass fractions
        self.mf = {}

        #: [Dict] Per-facet ablated species masses
        self.mass ={}

        #: [Dict] Per-facet ablated species emissions (kg/km)
        self.emission = {}

        #: [array] Per-facet density for mixtures
        self.rho_mix = np.zeros(n_faces)

        #: [array] Per-facet mass fractions for mixtures
        self.c_i_mix = np.zeros([n_faces, n_free_species])

        #: [array] Per-facet pressure for mixtures
        self.P_mix = np.zeros(n_faces)

        #: [array] Per-facet temperature for mixtures
        self.T_mix = np.zeros(n_faces)

        #: [list] List of species names across all component mixtures
        self.species = species

        #: [float] Mass fraction cutoff
        self.cutoff = cutoff

        #: [Mutationpp Mixture] Ablation mix
        self.mix = None

        #: [float] Oxygen content of surrounding air
        self.oxy_content = 1.0

    def get_species_list(self,assembly, thermo_method = 'cea'):
        """Retrieves the list all species from assembly components 

        Args:
            assembly (assembly.Assembly): The parent assembly
        """
        
        for component in assembly.objects:
            if component.mixture is not None:
                match thermo_method:
                    case 'cea':
                        mix_species = component.species
                        mix_species.append('N')
                        mix_species.append('O')
                        ablated_mix = cea.Mixture(mix_species,products_from_reactants=True)
                        species_names = ablated_mix.species_names
                    case 'mpp':
                        mix = mpp.Mixture(mpp.MixtureOptions(component.mixture))
                        species_names = [mix.speciesName(i) for i in range(mix.nSpecies())]
                for name in species_names:    
                    if name not in self.species: self.species.append(name)
        for speci in self.species:
            self.rho[speci] = np.zeros_like(self.rho_mix)
            self.mf[speci] = np.zeros_like(self.rho_mix)
            self.mass[speci] = np.zeros_like(self.rho_mix)
            self.emission[speci] = np.zeros_like(self.rho_mix)

    def mix_excess(self, assembly, options, delta_t=1):
        """Performs an air-in-excess equilibriation of each ablating facet of the assembly to compute emitted byproducts

        Args:
            assembly (assembly.Assembly): The parent assembly
            options (configuration.Options): TITAN options 
            delta_t (int, optional): TITAN time step. Defaults to 1.
        """
        mx = mpp.Mixture(mpp.MixtureOptions(options.aerothermo.mixture))
        free_species_names = [mx.speciesName(i_spec) for i_spec in range(mx.nSpecies())]

        for speci in self.species:
            self.rho[speci] = np.zeros_like(self.rho_mix)
            self.mf[speci] = np.zeros_like(self.rho_mix)
            self.mass[speci] = np.zeros_like(self.rho_mix)
            self.emission[speci] = np.zeros_like(self.rho_mix)

        if options.thermal.byproducts_method.lower()=='mpp': q = multiprocessing.Queue()
        for component in assembly.objects:
            if component.mixture is None: continue
            stoich_mult = None
            if not hasattr(component,'stoichiometric_mult'): 
                try:
                    component.stoichiometric_mult = get_stoichiometric_ratio(mpp.Mixture(component.mixture))
                except: 
                    make_mixfile_if_needed(component.mixture, component.species)
                    component.stoichiometric_mult = get_stoichiometric_ratio(mpp.Mixture(component.mixture))
            stoich_mult = component.stoichiometric_mult 

            
            if options.thermal.byproducts_method.lower()=='cea':
                mf, rho = cea_excess_air(self.P_mix, self.T_mix, self.c_i_mix, free_species_names, component.mass_fraction, component.species, self.oxy_content*stoich_mult)
                for name, value in mf.items():
                    if name not in self.species:
                            self.species.append(name)
                            self.rho[name] = np.zeros_like(self.rho_mix)
                            self.mf[name] = np.zeros_like(self.rho_mix)
                            self.mass[name] = np.zeros_like(self.rho_mix)
                            self.emission[name] = np.zeros_like(self.rho_mix)
                    if value>self.cutoff:
                        try:
                            self.rho[name][component.facet_index] = rho * value
                            self.mf[name][component.facet_index] = value
                            mass = component.facet_dm * value
                            self.mass[name][component.facet_index] = mass

                            self.emission[name][component.facet_index] = 1000*np.abs(mass/(delta_t*assembly.trajectory.velocity*np.sin(assembly.trajectory.gamma)))
                        except: pass
            elif options.thermal.byproducts_method.lower()=='mpp':
                # For memory safety it's nicer to run mpp as a separate process
                p = multiprocessing.Process(target=air_in_excess_mpp,args=(self.P_mix,
                                                                    self.T_mix,
                                                                    self.c_i_mix,
                                                                    component.mixture,
                                                                    component.species,
                                                                    component.mass_fraction,
                                                                    free_species_names,
                                                                    q,
                                                                    options.thermal.excess_mult,
                                                                    options.verbose,
                                                                    stoich_mult,
                                                                    self.oxy_content
                                                                    ))
                p.start()
                p.join(5.0)
                try:
                    mf, rhos, stoich_mult = q.get(timeout=5.0)
                except Exception as e: 
                    print('Error solving mixture for {} at P={} T={}!'.format(component.name, self.P_mix, self.T_mix))
                    if p.is_alive(): 
                        p.terminate()
                        p.kill()
                    print(e)
                    continue
                try:
                    
                    if not hasattr(component,'stoichiometric_mult'): component.stoichiometric_mult = stoich_mult
                    # if options.verbose: 
                    #     for spec, mfr in zip(self.species, mf): print(spec, mfr)
                    notable_species = np.argwhere(mf>self.cutoff).flatten()

                    rhos = np.tile(np.array(rhos), (len(component.facet_dm), 1))
                    mf = np.tile(np.array(mf), (len(component.facet_dm), 1))

                    per_species_mass = mf * component.facet_dm[:,np.newaxis]
                    for i_nz in notable_species:
                        name = self.species[i_nz]
                        self.rho[name][component.facet_index]  = rhos[:,i_nz]
                        self.mf[name][component.facet_index]   = mf[:,i_nz]
                        mass = per_species_mass[:,i_nz]
                        self.mass[name][component.facet_index] = mass
                        ## Emission as kg of byproduct/kilometre of altitude = m/dt / [ v sin(gamma)]
                        self.emission[name][component.facet_index] = 1000*np.abs(mass/(delta_t*assembly.trajectory.velocity*np.sin(assembly.trajectory.gamma)))
                except Exception as e:
                    print(e)
            else: raise Exception('Byproducts method must be either mpp or cea!')

def cea_excess_air(P : float, T : float, c_i_mix : np.ndarray, free_mix_species_names : list, mass_fractions : np.ndarray, elements : list, free_ratio : float = None):
    """_summary_

    Args:
        P (float): _description_
        T (float): _description_
        c_i_mix (np.ndarray): _description_
        free_mix_species_names (list): _description_
        mass_fractions (np.ndarray): _description_
        elements (list): _description_
        free_ratio (float, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    mix_species = elements
    mix_species.append('N')
    mix_species.append('O')
    ablated_mix = cea.Mixture(mix_species,products_from_reactants=True)
    solver = cea.EqSolver(ablated_mix)
    solution = cea.EqSolution(solver)

    n_species = ablated_mix.num_species
    cea_mass_fractions = np.zeros(n_species)

    for i_mf, mf in enumerate(mass_fractions): 
        cea_mass_fractions[ablated_mix.species_names.index(elements[i_mf])] = mf

    for mf_a, name in zip(c_i_mix, free_mix_species_names):
        i_species = ablated_mix.species_names.index(name)
        cea_mass_fractions[i_species] = free_ratio*mf_a
    
    solver.solve(solution, cea.TP, T, P, cea_mass_fractions / np.sum(cea_mass_fractions))
    print(solution.density)
    return solution.mass_fractions, solution.density, 

def air_in_excess_mpp(P :float, T :float, c_i_mix : np.ndarray, component_mixture : str, component_species : list, component_mass_fraction : np.ndarray, 
                    free_mix_species_names : list, queue, excess_mult=2.5, verbose=False, stoichiometric_mult=None, oxy_content=1):
    """Equilibriates the component mixture with the specified excess ratio using Mutation++

    Args:
        component (component.Component): Target component
        free_mix_species_names (list): List of species names in the freestream mixture 
        excess_mult (float, optional): Multiplier of the stoichiometric ratio. Defaults to 2.5.
    """
    
    ablated_mix_opts = mpp.MixtureOptions(component_mixture)
    ablated_mix_opts.setStateModel("Equil")
    if verbose: print('Creating mix {}...'.format(component_mixture))
    mix = mpp.Mixture(ablated_mix_opts)
    species_map = [mix.speciesIndex(name) for name in component_species]

    if np.any(np.array(species_map)<0): 
        raise Exception('Could not find all of {} in mixture {} on component!'.format(component_species,component_mixture))

    if stoichiometric_mult is None: stoichiometric_mult = get_stoichiometric_ratio(mix)
    component_mass_fraction /= np.sum(component_mass_fraction) # No harm in renormalising this
    elem_list = get_composition(mix, free_mix_species_names, component_mass_fraction, 
                                        species_map, c_i_mix, excess_mult*stoichiometric_mult/oxy_content)

    mix.addComposition(elem_list, True)
    if verbose:
        print('Composition,P,T')
        
        print(elem_list)
        print(P)
        print(T)

        print('Setting state...')
    mix.setState(P, T, 1)
    queue.put([mix.Y(), mix.densities(), stoichiometric_mult])


def get_composition(ablated_mix, free_mix_species_names : list, component_mass_fraction : list, 
                    species_map : list, c_i_mix : np.ndarray, freestream_ratio = 1.0) -> str:
    """Generates a comma-separated string of elements and mole fractions to assign as a compostion to a Mutation++ mixture

    Args:
        ablated_mix (mutationpp.Mixture): Target mixture
        free_mix_species_names (list): List of species names in the freestream mixture 
        component_mass_fraction (list): Mass fraction of component species released into the flow
        species_map (list): Indices of component species in full mixture 
        freestream_ratio (float): Ratio of freestream mixture to ablated species. Defaults to 1.0.

    Returns:
        str: Elemental composition string
    """

    n_species = ablated_mix.nSpecies()
    mass_fractions = np.zeros(n_species)

    for i_mf, mf in enumerate(component_mass_fraction): mass_fractions[species_map[i_mf]] = mf

    for mf_a, name in zip(c_i_mix, free_mix_species_names):
        i_species = ablated_mix.speciesIndex(name)
        if i_species<0: raise Exception('Could not find {} in mixture!'.format(name))
        mass_fractions[i_species] = freestream_ratio*mf_a

    mass_fractions/=np.sum(mass_fractions)
    species_x = ablated_mix.convert_y_to_x(mass_fractions)
    elements_x = ablated_mix.convert_x_to_xe(species_x)
    elements_x = np.array(elements_x)/np.sum(elements_x)

    elem_list = ''
    for i_elem, x_elem in enumerate(elements_x):
        elem_list+=ablated_mix.elementName(i_elem)+':'+str(x_elem)+','

    return elem_list
    
def get_stoichiometric_ratio(mix : mpp.Mixture, oxy_content : float = 1.0) -> float:
    """Returns the maximal stoichiometric ratio of any oxide compound in a Mutation++ mixture

    Args:
        mix (mpp.Mixture): Target mixture
        oxy_content (float, optional): Oxygen content of the mixture. Defaults to 1.0.

    Returns:
        float: Maximal stoichiometric ratio of the mixture
    """
    list_of_stoichs = []
    matrix = mix.elementMatrix()
    i_Oxygen = mix.elementIndex('O')
    m_Oxygen = mix.atomicMass(i_Oxygen)
    matrix = matrix[np.nonzero(matrix[:,i_Oxygen])[0],:]

    for elem in range(mix.nElements()):
        if mix.elementName(elem)=='O': continue

        ratios = (matrix[:,elem]/matrix[:,i_Oxygen])
        ratios = ratios[np.nonzero(ratios)[0]]

        if len(ratios)<1: continue
        stoich = (m_Oxygen/mix.atomicMass(elem))/np.min(ratios)

        list_of_stoichs.append(stoich)

    return np.max(list_of_stoichs)/oxy_content

def make_mixfile_if_needed(name,mix_elems:list) ->str:
    if not 'N' in mix_elems: mix_elems.append('N')
    if not 'O' in mix_elems: mix_elems.append('O')

    mpp_data_dir = mpp.GlobalOptions.dataDirectory()
    with open(mpp_data_dir+'/mixtures/'+name+'.xml', 'w') as f:
        f.write('<mixture thermo_db="NASA-9">\n')
        f.write('    <species>\n')
        f.write('        {all with '+','.join(mix_elems)+'}\n')
        f.write('    </species>\n')
        f.write('</mixture>')
    
    mix = mpp.Mixture(mpp.MixtureOptions(name))

    return mix

def test_mix_wrapper(combo,i_mix):
    mpp_data_dir = mpp.GlobalOptions.dataDirectory()
    name = 'candidate_'.join(i_mix)
    with open(mpp_data_dir+'/'+name+'.xml', 'w') as f:
        f.write('<mixture thermo_db="NASA-9">\n')
        f.write('    <species>\n')
        f.write('        '+' '.join(combo)+'\n')
        f.write('    <\\species>\n')
        f.write('<\\mixture>')
    mixO = mpp.MixtureOptions(name)
    mixO.setStateModel('Equil')
    mix = mpp.Mixture(mixO)
