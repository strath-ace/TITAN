"""byproducts module."""
import numpy as np
import mutationpp as mpp

class Byproducts():
    """Track species released by ablating components and their emissions.

    Per-facet mixture properties and species-resolved mass, density,
    composition, and emission data are stored for post-processing.
    """

    def __init__(self, n_faces : int, is_emitting=False, rho = {}, mf = {}, mass = {}, species = [], n_free_species=5, cutoff=1e-6):
        """Initialize per-facet storage for ablation byproducts.

    :param n_faces: Number of surface facets represented by the arrays.
    :type n_faces: int
    :param is_emitting: Initial state indicating whether emissions are active.
    :type is_emitting: bool
    :param rho: Optional initial species-density mapping (storage is initialized internally).
    :type rho: dict
    :param mf: Optional initial species-mass-fraction mapping (storage is initialized internally).
    :type mf: dict
    :param mass: Optional initial species-mass mapping (storage is initialized internally).
    :type mass: dict
    :param species: Initial species-name list.
    :type species: list[str]
    :param n_free_species: Number of freestream species tracked in each facet mixture.
    :type n_free_species: int
    :param cutoff: Minimum species mass fraction retained in output.
    :type cutoff: float
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
        """Collect mixture species from components and initialize their arrays.

    :param assembly: Assembly whose component mixtures are inspected.
    :type assembly: Assembly
    """
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
        """Equilibrate ablated component mixtures with excess air.

        Species properties and altitude-normalized emissions are accumulated
        on the facets associated with each component.

    :param assembly: Assembly containing the ablating components.
    :type assembly: Assembly
    :param options: Simulation settings, including freestream-mixture options.
    :type options: Options
    :param delta_t: Elapsed time represented by the ablated mass, in seconds.
    :type delta_t: float
    """
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
        """Equilibrate a component's ablated mixture with freestream species.

    :param component: Component defining the ablated mixture and composition.
    :type component: Component
    :param free_mix_species_names: Species names available in the freestream mixture.
    :type free_mix_species_names: list[str]
    :param excess_mult: Multiplier applied to the component's stoichiometric ratio.
    :type excess_mult: float
    """

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
        """Build a Mutation++ composition string from mass fractions.

        Component species and freestream species are combined, converted to
        elemental mole fractions, and serialized as comma-separated entries.

    :param ablated_mix: Mutation++ mixture representing the ablated material.
    :type ablated_mix: mpp.Mixture
    :param free_mix_species_names: Species names in the freestream mixture.
    :type free_mix_species_names: list[str]
    :param component_mass_fraction: Mass fractions for the component species.
    :type component_mass_fraction: list[float]
    :param species_map: Indices mapping component species into ``ablated_mix``.
    :type species_map: list[int]
    :param freestream_ratio: Relative amount of freestream material to include.
    :type freestream_ratio: float

    :return: Element names and normalized mole fractions in Mutation++ format.
    :rtype: str
    """

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
    """Find the largest fuel-to-oxygen stoichiometric ratio in a mixture.

:param mix: Mutation++ mixture to inspect.
:type mix: mpp.Mixture
:param oxy_content: Oxygen availability scaling factor.
:type oxy_content: float

:return: Maximum stoichiometric ratio adjusted by oxygen availability.
:rtype: float
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
