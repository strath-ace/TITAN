import numpy as np
import mutationpp as mpp

class Byproducts():
    """ Class Byproducts

        A class to store the emission byproducts of each assembly
    """

    def __init__(self, n_faces, is_emitting=False, rho = {}, mf = {}, mass = {}, species = [], n_free_species=5, cutoff=1e-6):
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

    def get_species_list(self,assembly):
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
        mx = mpp.Mixture(mpp.MixtureOptions(options.aerothermo.mixture))
        species_names = [mx.speciesName(i_spec) for i_spec in range(mx.nSpecies())]

        for speci in self.species:
            self.rho[speci] = np.zeros_like(self.column_height_mix)
            self.mf[speci] = np.zeros_like(self.column_height_mix)
            self.mass[speci] = np.zeros_like(self.column_height_mix)
            self.emission[speci] = np.zeros_like(self.column_height_mix)

        for component in assembly.objects:
            if component.mixture is None: continue
            rhos, mf = self.air_in_excess(component, [mx.speciesName(i_spec) for i_spec in range(mx.nSpecies())])
            notable_species = np.argwhere(mf>self.cutoff).flatten()
            rhos = np.tile(np.array(rhos), (len(component.facet_dm), 1))
            mf = np.tile(np.array(mf), (len(component.facet_dm), 1))

            per_species_mass = mf * component.facet_dm[:,np.newaxis]
            for i_nz in notable_species:
                name = self.species[i_nz]
                self.rho[name]  = rhos[:,i_nz]
                self.mf[name]   = mf[:,i_nz]
                mass = per_species_mass[:,i_nz]
                self.mass[name] = mass
                ## Emission as kg of byproduct/kilometre of altitude = m/dt / [ v sin(gamma)]
                self.emission[name] = 1000*np.abs(mass/(delta_t*assembly.trajectory.velocity*np.sin(assembly.trajectory.gamma)))

    def mix_deprec(self, assembly, options, vol_method='excess', delta_t=1):

        match vol_method:
            case 'ble'   : assem_vol_thickness = self.column_height_mix
            case 'excess': assem_vol_thickness = 100000*self.column_height_mix

        mx = mpp.Mixture(mpp.MixtureOptions(options.aerothermo.mixture))
        species_names = [mx.speciesName(i_spec) for i_spec in range(mx.nSpecies())]

        for speci in self.species:
            self.rho[speci] = np.zeros_like(self.column_height_mix)
            self.mf[speci] = np.zeros_like(self.column_height_mix)
            self.mass[speci] = np.zeros_like(self.column_height_mix)
            self.emission[speci] = np.zeros_like(self.column_height_mix)

        for component in assembly.objects:
            if component.mixture is None: continue
            #facet_volume = component.mesh.facet_area * assem_vol_thickness[component.facet_index]
            #facet_mass =  facet_volume * self.rho_mix
            
        
            
            ablated_mass = []
            facet_ablation_mass = component.facet_dm
            if options.dynamics.augmented_state: facet_ablation_mass/=delta_t

            for mf in component.mass_fraction:
                ablated_mass.append(-1*mf*facet_ablation_mass)


            ablated_mass = np.array(ablated_mass)
            augmented_species_mass = np.hstack([species_mass, ablated_mass])
            augmented_total_mass = np.sum(augmented_species_mass, axis=1, keepdims=True)

            nonzero_facets = np.where(facet_volume>0)[0]
            new_mass_densities = np.zeros_like(augmented_species_mass)
            new_mass_densities[nonzero_facets] = augmented_species_mass[nonzero_facets]
            new_mass_densities[nonzero_facets] /= np.full_like(augmented_species_mass[nonzero_facets].T, facet_volume[nonzero_facets].T).T

            ablated_mix_opts = mpp.MixtureOptions(component.mixture)
            ablated_mix_opts.setStateModel("Equil")
            ablated_mix = mpp.Mixture(ablated_mix_opts)
            [species_names.append(name) for name in component.species] 
            species_map = [ablated_mix.speciesIndex(name) for name in species_names]
            n_species = ablated_mix.nSpecies()


            for i_facet, density in enumerate(new_mass_densities):
                if np.all(density>0):
                    density_state = np.zeros(n_species)
                    for rho_spec, i_species in zip(density, species_map): density_state[i_species] = rho_spec
                    spec_x = ablated_mix.convert_rho_to_x(density_state)
                    elem_x = ablated_mix.convert_x_to_xe(spec_x)
                    elem_list = ''
                    for i_elem, x_elem in enumerate(elem_x):
                        elem_list+=ablated_mix.elementName(i_elem)+':'+str(x_elem)+','
                    ablated_mix.addComposition(elem_list, True)
                    ablated_mix.equilibrate(self.P_mix, self.T_mix)
                    mf   = ablated_mix.Y()
                    rhos = ablated_mix.densities()

                    densities_dict = {}
                    mfs_dict = {}
                    mass_dict = {}
                    for i_nz in np.argwhere(mf>self.cutoff).flatten():
                        name = ablated_mix.speciesName(i_nz)
                        self.rho[name][i_facet]  = rhos[i_nz]
                        self.mf[name][i_facet]   = mf[i_nz]
                        mass = mf[i_nz] * augmented_total_mass[i_facet][0]
                        self.mass[name][i_facet] = mass
                        ## Emission as kg of byproduct/kilometre of altitude = m/dt / [ v sin(gamma)]
                        self.emission[name][i_facet] = 1000*np.abs(mass/(delta_t*assembly.trajectory.velocity*np.sin(assembly.trajectory.gamma)))


            del ablated_mix, ablated_mix_opts
    

    def air_in_excess(self, component, air_mix_species_names, excess_ratio=2.5, tol=1e-6):
        '''
            Function to generate stoichiometric mixture of air and ablated components
        '''
        print('Getting mix opts')
        ablated_mix_opts = mpp.MixtureOptions(component.mixture)
        ablated_mix_opts.setStateModel("Equil")
        print('Setting mix opts')
        ablated_mix = mpp.Mixture(ablated_mix_opts)
        species_map = [ablated_mix.speciesIndex(name) for name in component.species]
        oxygen_species = [ablated_mix.speciesIndex('O2')], 
                        #   ablated_mix.speciesIndex('O'), 
                        #   ablated_mix.speciesIndex('NO'),
                        #   ablated_mix.speciesIndex('NO2')]
        if np.any(np.array(species_map)<0): raise Exception('Could not find all species!')
        
        
        eps = np.inf

        stoichiometric_mult = 1.0

        ub = 7.5
        lb = 0.5
        n_iter = 0
        equil_prev = 1
        # while abs(eps)>tol:
            
        #     stoichiometric_mult = np.mean([lb,ub])

        #     elem_list = self.get_composition(ablated_mix, air_mix_species_names, component.mass_fraction, 
        #                                      species_map, stoichiometric_mult)

        #     ablated_mix.addComposition(elem_list, True)
        #     ablated_mix.equilibrate(self.P_mix, self.T_mix)

        #     O2_equil = np.max([ablated_mix.X()[index] for index in oxygen_species])

        #     eps = 1-abs(O2_equil/equil_prev)

        #     if eps>0: # Mixture tending rich ()
        #         lb = stoichiometric_mult
        #     else: # Mixture tending lean
        #         ub = stoichiometric_mult
        #     if n_iter>1e6: break
        #     n_iter +=1
        #     equil_prev = O2_equil
        stoichiometric_mult = 3.384
        elem_list = self.get_composition(ablated_mix, air_mix_species_names, component.mass_fraction, 
                                         species_map, excess_ratio*stoichiometric_mult)

        ablated_mix.addComposition(elem_list, True)
        print('Setting state...')
        ablated_mix.setState(self.P_mix, self.T_mix[0], 1)

        mf   = ablated_mix.Y()
        rhos = ablated_mix.densities()
        return mf, rhos

    def get_composition(self, ablated_mix, air_mix_species_names, component_mass_fraction, species_map, stoichiometric_mult):
        n_species = ablated_mix.nSpecies()
        mass_fractions = np.zeros(n_species)
        for i_mf, mf in enumerate(component_mass_fraction): mass_fractions[species_map[i_mf]] = mf
        for mf_a, name in zip(self.c_i_mix, air_mix_species_names):
            i_species = ablated_mix.speciesIndex(name)
            if i_species<0: raise Exception('Could not find {} in mixture!'.format(name))
            mass_fractions[i_species] = stoichiometric_mult*mf_a
        mass_fractions/=np.sum(mass_fractions)
        
        species_x = ablated_mix.convert_y_to_x(mass_fractions)
        elements_x = ablated_mix.convert_x_to_xe(species_x)
        elements_x = np.array(elements_x)/np.sum(elements_x)
            #O2_base = species_x[ablated_mix.speciesIndex('O2')]
        elem_list = ''
        for i_elem, x_elem in enumerate(elements_x):
            elem_list+=ablated_mix.elementName(i_elem)+':'+str(x_elem)+','
        return elem_list
