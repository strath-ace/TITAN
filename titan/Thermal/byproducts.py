"""byproducts module."""
import numpy as np
import mutationpp as mpp

class Byproducts():
    """Class to handle byproducts generation of assembly using Mutation++
    """

    def __init__(self, n_faces : int, is_emitting=False, rho = {}, mf = {}, mass = {}, species = [], n_free_species=5, cutoff=1e-6):
        """Args

:param n_faces:
:type n_faces:
:param is_emitting:
:type is_emitting:
:param rho:
:type rho:
:param mf:
:type mf:
:param mass:
:type mass:
:param species:
:type species:
:param n_free_species:
:type n_free_species:
:param cutoff:
:type cutoff:"""
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

        #: [array] Per-facet column height for mixtures
        self.column_height_mix = np.zeros(n_faces)

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

    def get_species_list(self,assembly):
        """Retrieves the list all species from assembly components 

:param assembly:
:type assembly:"""
        for component in assembly.objects:
            if component.mixture is not None:
                mix = mpp.Mixture(mpp.MixtureOptions(component.mixture))
                for i_spec in range(mix.nSpecies()):
                    name = mix.speciesName(i_spec)
                    if name not in self.species: self.species.append(name)
        for speci in self.species:
            self.rho[speci] = np.zeros_like(self.column_height_mix)
            self.mf[speci] = np.zeros_like(self.column_height_mix)
            self.mass[speci] = np.zeros_like(self.column_height_mix)
            self.emission[speci] = np.zeros_like(self.column_height_mix)

    def mix_excess(self, assembly, options, delta_t=1):
        """Performs an air-in-excess equilibriation of each ablating facet of the assembly to compute emitted byproducts

:param assembly:
:type assembly:
:param options:
:type options:
:param delta_t:
:type delta_t:"""
        mx = mpp.Mixture(mpp.MixtureOptions(options.aerothermo.mixture))
        free_species_names = [mx.speciesName(i_spec) for i_spec in range(mx.nSpecies())]

        for speci in self.species:
            self.rho[speci] = np.zeros_like(self.column_height_mix)
            self.mf[speci] = np.zeros_like(self.column_height_mix)
            self.mass[speci] = np.zeros_like(self.column_height_mix)
            self.emission[speci] = np.zeros_like(self.column_height_mix)

        for component in assembly.objects:
            if component.mixture is None: continue
            self.air_in_excess(component, free_species_names, options.thermal.excess_mult)
            
            mf = self.mix.Y()
            rhos = self.mix.densities()
            
            if options.verbose: 
                for spec, mfr in zip(self.species, mf): print(spec, mfr)
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
    

    def air_in_excess(self, component, free_mix_species_names : list, excess_mult=2.5):
        """Equilibriates the component mixture with the specified excess ratio

:param component:
:type component:
:param free_mix_species_names:
:type free_mix_species_names:
:param excess_mult:
:type excess_mult:"""

        ablated_mix_opts = mpp.MixtureOptions(component.mixture)
        ablated_mix_opts.setStateModel("Equil")
        self.mix = mpp.Mixture(ablated_mix_opts)
        species_map = [self.mix.speciesIndex(name) for name in component.species]

        if np.any(np.array(species_map)<0): raise Exception('Could not find all species!')

        if not hasattr(component, 'stoichiometric_mult'): component.stoichiometric_mult = get_stoichiometric_ratio(self.mix)
        elem_list = self.get_composition(self.mix, free_mix_species_names, component.mass_fraction, 
                                         species_map, excess_mult*component.stoichiometric_mult/self.oxy_content)

        self.mix.addComposition(elem_list, True)
        self.mix.setState(self.P_mix, self.T_mix[0], 1)


    def get_composition(self, ablated_mix, free_mix_species_names : list, component_mass_fraction : list, 
                        species_map : list, freestream_ratio = 1.0) -> str:
        """Generates a comma-separated string of elements and mole fractions to assign as a compostion to a Mutation++ mixture

:param ablated_mix:
:type ablated_mix:
:param free_mix_species_names:
:type free_mix_species_names:
:param component_mass_fraction:
:type component_mass_fraction:
:param species_map:
:type species_map:
:param freestream_ratio:
:type freestream_ratio:

:return:
:rtype:"""

        n_species = ablated_mix.nSpecies()
        mass_fractions = np.zeros(n_species)

        for i_mf, mf in enumerate(component_mass_fraction): mass_fractions[species_map[i_mf]] = mf

        for mf_a, name in zip(self.c_i_mix, free_mix_species_names):
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

:param mix:
:type mix:
:param oxy_content:
:type oxy_content:

:return:
:rtype:"""
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
