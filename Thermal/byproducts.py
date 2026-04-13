import numpy as np
import mutationpp as mpp
import copy
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

        #: [Mutationpp Mixture] Ablation mix
        self.mix = None

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
            import faulthandler
            faulthandler.enable()
            if component.mixture is None: continue
            self.air_in_excess(component, [mx.speciesName(i_spec) for i_spec in range(mx.nSpecies())])
            
            mf = self.mix.Y()
            rhos = self.mix.densities()
            
            for spec, mfr in zip(self.species, mf): print(spec, mfr)
            notable_species = np.argwhere(mf>self.cutoff).flatten()

            rhos = np.tile(np.array(rhos), (len(component.facet_dm), 1))
            mf = np.tile(np.array(mf), (len(component.facet_dm), 1))

            per_species_mass = mf * component.facet_dm[:,np.newaxis]
            for i_nz in range(len(self.species)):#notable_species:
                name = self.species[i_nz]
                self.rho[name]  = rhos[:,i_nz]
                self.mf[name]   = mf[:,i_nz]
                mass = per_species_mass[:,i_nz]
                self.mass[name] = mass
                ## Emission as kg of byproduct/kilometre of altitude = m/dt / [ v sin(gamma)]
                self.emission[name] = 1000*np.abs(mass/(delta_t*assembly.trajectory.velocity*np.sin(assembly.trajectory.gamma)))
    

    def air_in_excess(self, component, air_mix_species_names, excess_ratio=2.5, tol=1e-6):
        '''
            Function to generate stoichiometric mixture of air and ablated components
        '''
        print('Getting mix opts')
        ablated_mix_opts = mpp.MixtureOptions(component.mixture)
        ablated_mix_opts.setStateModel("Equil")
        print('Setting mix opts')
        self.mix = mpp.Mixture(ablated_mix_opts)
        species_map = [self.mix.speciesIndex(name) for name in component.species]
        oxygen_species = [self.mix.speciesIndex('O2')], 
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
        elem_list = self.get_composition(self.mix, air_mix_species_names, component.mass_fraction, 
                                         species_map, excess_ratio*stoichiometric_mult)

        self.mix.addComposition(elem_list, True)
        print('Setting state...')
        self.mix.setState(self.P_mix, self.T_mix[0], 1)

        #mf   = np.array(copy.copy(ablated_mix.Y()))
        ## Mutation is doing some weird stuff which causes all sorts of errors, we fix it but just taking the data
        ## This is obviously an ugly hack but what else is there to do
        #mf = self.mix.Y()
        #rhos = np.zeros_like(self.mix.densities())
        #del ablated_mix
        #for spec, mfr in zip(self.species, mf): print(spec, mfr)
        
       # return rhos, mf

    def get_composition(self, ablated_mix, air_mix_species_names, component_mass_fraction, species_map, stoichiometric_mult):
        n_species = ablated_mix.nSpecies()
        mass_fractions = np.zeros(n_species)
        for i_mf, mf in enumerate(component_mass_fraction): mass_fractions[species_map[i_mf]] = mf
        for mf_a, name in zip(self.c_i_mix, air_mix_species_names):
            i_species = ablated_mix.speciesIndex(name)
            if i_species<0: raise Exception('Could not find {} in mixture!'.format(name))
            mass_fractions[i_species] = stoichiometric_mult*mf_a
        mass_fractions/=np.sum(mass_fractions)
        #for el, mfr in zip(range(ablated_mix.nElements()),ablated_mix.convert_y_to_ye(mass_fractions)): #print(ablated_mix.elementName(el), mfr)
        species_x = ablated_mix.convert_y_to_x(mass_fractions)
        elements_x = ablated_mix.convert_x_to_xe(species_x)
        elements_x = np.array(elements_x)/np.sum(elements_x)
            #O2_base = species_x[ablated_mix.speciesIndex('O2')]
        elem_list = ''
        for i_elem, x_elem in enumerate(elements_x):
            elem_list+=ablated_mix.elementName(i_elem)+':'+str(x_elem)+','
        return elem_list
