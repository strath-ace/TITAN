#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN 
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""aerothermo module."""
import numpy as np
from ..Freestream import mix_properties
from ..Dynamics.frames import *
from scipy import special
from copy import copy
from ..Aerothermo import su2, switch, sparta
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RigidTransform as Trans
from scipy.spatial import KDTree
from scipy.optimize import brentq, fsolve
from functools import partial

import trimesh, pathlib
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
except:
    print('PyEmbree/Embreex library not set up')
    from trimesh.ray.ray_triangle import RayMeshIntersector

try:
    import mutationpp as mpp
except:
    print("Mutationpp library not set up")

def mixture_mpp(mixture = "air5") -> mpp.Mixture:
    """Retrieve the mixture object of the Mutation++ library

    :param mixture: String specifying the mixture.
    :type mixture: str
    :return: Mixture object.
    :rtype: mpp.Mixture

    """


    opts = mpp.MixtureOptions(mixture)
    opts.setThermodynamicDatabase("RRHO")
    opts.setStateModel("ChemNonEq1T")
    opts.setViscosityAlgorithm("Gupta-Yos")

    mix = mpp.Mixture(opts)
    
    return mix

### Stagnation Equations
def stagnation_P(P:float, gamma:float, M:float)->float:
    """Compute stagnation pressure

    :param P: Free-stream pressure
    :type P: float
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param M: Mach number.
    :type M: float
    :return: Stagnation pressure.
    :rtype: float"""
    P_0 = P * (1 + ((gamma - 1.0)/2.0)*(M**2))**(gamma / (gamma - 1))
    return P_0

def stagnation_T(T:float, gamma:float, M:float)->float:
    """Compute stagnation temperature

    :param T: Temperature.
    :type T: float
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param M: Mach number.
    :type M: float
    :return: Stagnation temperature.
    :rtype: float"""
    T_0 = T * (1 + ((gamma - 1.0)/2.0)*(M**2))
    return T_0

### Normal Shock Equations
def normal_shock_P(P:float, gamma:float, M:float) -> float:
    """Compute isentropic normal shock pressure

    :param P: Pre-shock pressure.
    :type P: float
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param M: Mach number.
    :type M: float
    :return: Post-shock pressure.
    :rtype: float"""
    P_post = P*((2.0 * gamma * (M**2)) - (gamma - 1.0)) / (gamma + 1.0)

    return P_post

def normal_shock_T(T:float, gamma:float, M:float) -> float:
    """Compute isentropic normal shock temperature

    :param T: Temperature.
    :type T: float
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param M: Mach number.
    :type M: float
    :return: Post-shock temperature.
    :rtype: float"""
    T_post = T*(((2.0 * gamma * (M**2.0)) - (gamma - 1.0)) * (((gamma - 1.0) * (M**2.0)) + 2.0)) / (((gamma + 1.0)**2.0) * (M**2.0))
    return T_post
    

def normal_shock_M(gamma: float, M:float) -> float:
    """Compute isentropic normal shock Mach

    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param M: Mach number.
    :type M: float
    :return: Post-shock Mach number.
    :rtype: float"""
    M_post = np.sqrt((((gamma - 1.0) * (M**2.0)) + 2.0) / ((2.0 * gamma * (M**2.0)) - (gamma - 1.0)))
    return M_post
    

def normal_shock_rho(rho:float, gamma:float, M:float)->float:
    """Compute isentropic normal shock density

    :param rho: Pre-shock density.
    :type rho: float
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param M: Mach number.
    :type M: float
    :return: Post-shock density.
    :rtype: float"""
    rho_post = rho*(((gamma + 1.0) * (M**2.0)) / (((gamma - 1.0) * (M**2.0)) + 2.0))
    return rho_post

def energy_loop_obj_func(mix: mpp.Mixture, P_eq: float, h_ref: float, T_eq: float) -> float:
    """Objective function for the energy loop to ensure conservation of 
    total enthalpy at target equilibrium conditions

    :param mix: Freestream mixture.
    :type mix: Mutationpp.Mixture
    :param P_eq: Equilibrium pressure.
    :type P_eq: float
    :param h_ref: Reference enthalpy.
    :type h_ref: float
    :param T_eq: Equilibrium temperature.
    :type T_eq: float
    :return: Enthalpy discrepancy.
    :rtype: float"""
    mix.equilibrate(T_eq, P_eq)
    h_eq = mix.mixtureHMass()
    return h_eq-h_ref

def energy_loop_brent(mix : mpp.Mixture, T_eq : float, P_eq : float, h_ref : float, T_max = 1e5) -> mpp.Mixture:
    """Root-finding loop to ensure conservation of total enthalpy at target equilibrium conditions.
    (Uses the Brent method)

    :param mix: Freestream mixture.
    :type mix: Mutationpp.Mixture
    :param T_eq: Initial guess for equilibrium temperature.
    :type T_eq: float
    :param P_eq: Initial guess for equilibrium pressure.
    :type P_eq: float
    :param h_ref: Reference enthalpy.
    :type h_ref: float
    :param T_max: Temperature upper limit.
    :type T_max: float
    :return: Equilibrium mixture.
    :rtype: Mutationpp.Mixture"""
    f = partial(energy_loop_obj_func, mix, P_eq, h_ref)

    T_eq = brentq(f, 10.0, T_max)
    mix.equilibrate(T_eq, P_eq)
    return mix

class stagnation_line_per_facet():
    """
    Class to store the flow conditions at freestream, stagnation, BLE and wall for every facet
    """

    def __init__(self, Tfree : float, Pfree : float, Mfree : np.ndarray, Twall : np.ndarray, mixname = None):
        """Solve stations of the stagnation line (Free, Post-Shock, BLE, Wall) 
        from freestream and wall conditions

        :param Tfree: Freestream temperature.
        :type Tfree: float
        :param Pfree: Freestream pressure.
        :type Pfree: float
        :param Mfree: Vector of Freestream Mach numbers.
        :type Mfree: np.ndarray
        :param Twall: Vector of Wall temperatures.
        :type Twall: np.ndarray
        :param mixname: Name of the mixture.
        :type mixname: str"""
        if mixname == None: 
            print('No mix given to stagnation_line()! Defaulting to air5...')
            self.mix = mixture_mpp("air5")
        else: self.mix = mixture_mpp(mixname)

        self.Tfree = Tfree
        self.Pfree = Pfree
        self.Mfree = Mfree
        self.Twall = Twall
        self.n_facets = len(self.Mfree)
        #Equilibrate the mix with the freesteam conditions:
        self.mix.equilibrate(self.Tfree, self.Pfree)        
        self.gammafree = self.mix.mixtureFrozenGamma()
        self.ufree = self.Mfree*self.mix.frozenSoundSpeed()
        self.mufree = self.mix.viscosity()
        self.rhofree = self.mix.density()
        self.c_i_free = self.mix.Y()
        self.oxygen_mf = self.mix.convert_y_to_ye(self.c_i_free)[self.mix.elementIndex('O')]
        #molecular weight
        self.MW_free = self.mix.mixtureMw()
        
        self.T0_free = stagnation_T(self.Tfree, self.gammafree, self.Mfree)
        self.P0_free = stagnation_P(self.Pfree, self.gammafree, self.Mfree)
        #Total enthalpy at freestream
        self.H0_free = self.mix.mixtureHMass() + (self.Mfree*self.mix.frozenSoundSpeed())**2/2.0

        #Post-shock conditions:
        self.T_post = normal_shock_T(self.Tfree, self.gammafree, self.Mfree)
        self.P_post = normal_shock_P(self.Pfree, self.gammafree, self.Mfree)
        self.rho_post = normal_shock_rho(self.rhofree, self.gammafree, self.Mfree)
        self.M_post = normal_shock_M(self.gammafree, self.Mfree)
        self.u_post = self.M_post*np.sqrt((self.gammafree*self.P_post)/self.rho_post)

        self.T0_post = stagnation_T(self.T_post, self.gammafree, self.M_post)
        self.P0_post = stagnation_P(self.P_post, self.gammafree, self.M_post)
        self.rho0_post = self.rho_post*(1+(self.gammafree - 1) / 2.0 * self.M_post**2)**(1/(self.gammafree - 1))

        #Boundary layer edge conditions
        #Assuming mixture at equilibrium
        self.Te = self.T0_post
        self.Pe = self.P0_post


        self.rhoe = np.zeros(self.n_facets)
        self.mue = np.zeros(self.n_facets)
        self.He = np.zeros(self.n_facets)
        self.ce_i = np.zeros([self.n_facets,self.mix.nSpecies()])
        self.xe_i = np.zeros([self.n_facets,self.mix.nSpecies()])
        self.MWe = np.zeros(self.n_facets)
        self.mu_orig_e = np.zeros(self.n_facets)            
        self.Hd = np.zeros(self.n_facets)
        self.Pwall = np.zeros(self.n_facets)
        self.rhow = np.zeros(len(self.Twall))
        self.muw = np.zeros(len(self.Twall))
        self.Hw = np.zeros(len(self.Twall))

        # Assume facets adjacent in the list have similar temperature

        best_guess_T = self.Te[0]
        for i_facet in range(self.n_facets):
            conditions_dict = ble_conditions(self.mix, best_guess_T, self.Pe[i_facet], self.H0_free[i_facet], self.Twall[i_facet])
            best_guess_T = conditions_dict['Te']
            self.assign_facet_conditions(i_facet, conditions_dict)

        #Adimensional numbers
        #At the moment these values are hardcoded according to several literature sources
        self.Pr = 0.71
        self.Le = 1.0

    def assign_facet_conditions(self, i_facet : int, conditions_dict : dict):
        """Assigns flow conditions to the respective indices in flow property arrays.

        :param i_facet: Index of the facet.
        :type i_facet: int
        :param conditions_dict: Dictionary of conditions.
        :type conditions_dict: dict"""
        self.Te[i_facet] = conditions_dict['Te']
        self.Pe[i_facet] = conditions_dict['Pe']
        self.rhoe[i_facet] = conditions_dict['rhoe']
        self.mue[i_facet] = conditions_dict['mue']
        self.He[i_facet] = conditions_dict['He']
        self.ce_i[i_facet, :] = conditions_dict['ce_i']
        self.xe_i[i_facet, :] = conditions_dict['xe_i']
        self.MWe[i_facet] = conditions_dict['MWe']
        self.mu_orig_e[i_facet] = conditions_dict['mu_orig_e']
        self.Hd[i_facet] = conditions_dict['Hd']
        self.Pwall[i_facet] = conditions_dict['Pwall']
        self.rhow[i_facet] = conditions_dict['rhow']
        self.muw[i_facet] = conditions_dict['muw']
        self.Hw[i_facet] = conditions_dict['Hw']

def ble_conditions(mix : mpp.Mixture, best_guess_T : float, Pe : float, H0_free : float, Twall : float) -> dict:
    """Compute equilibrium conditions at boundary layer edge based upon post-shock stagnation enthalpy

    :param mix: Mutationpp mixture object.
    :type mix: mpp.Mixture
    :param best_guess_T: Value for best guess temperature.
    :type best_guess_T: float
    :param Pe: Initial pressure.
    :type Pe: float
    :param H0_free: Post-shock stagnation enthalpy.
    :type H0_free: float
    :param Twall: Wall temperature.
    :type Twall: float
    :return: Flow conditions dictionary.
    :rtype: dict"""
    #mix = mixture_mpp(mixname)
    mix = energy_loop_brent(mix, best_guess_T, Pe, H0_free)

    best_guess_T = mix.T()

    Te = mix.T()
    Pe = mix.P()
    rhoe = mix.density()
    mue = mix.viscosity()
    He = mix.mixtureHMass()

    #N O NO N2 O2 according to air_5 from Mutationpp
    ce_i = np.zeros(mix.nSpecies())
    xe_i = np.zeros(mix.nSpecies())
    ce_i[:] = mix.Y()
    xe_i[:] = mix.X()
    MWe = mix.mixtureMw()

    mix.setState(mix.densities(), Te, 1)
    mu_orig_e = mix.viscosity()

    #N - 33867025.2 J/Kg heat of formation
    #O - 15432544.8 J/Kg Heat of formation

    #Heat of dissociation                 
    Hd = 33867025.2*ce_i[0] + 15432544.8 *ce_i[1]

    #Wall conditions
    #Assuming mixture at equilibrium
    Pwall = Pe
    mix.equilibrate(Twall, Pwall)
    rhow =mix.density()
    muw = mix.viscosity()
    Hw = mix.mixtureHMass()

    return {'Te' : Te, 'Pe': Pe, 'rhoe' : rhoe, 'mue' : mue, 'He' : He, 'ce_i' : ce_i, 'xe_i' : xe_i, 'MWe' : MWe, 'mu_orig_e' : mu_orig_e, 'Hd' : Hd, 'Pwall' : Pwall, 'rhow' : rhow, 'muw' : muw, 'Hw' : Hw}

# Some minor code duplication, preserved here for robustness
class stagnation_line():
    """
    Class to store the flow conditions at freestream, stagnation, BLE and wall, global for the whole assembly
    """

    def __init__(self, Tfree : float, Pfree : float, Mfree : float, Twall : float, mix : mpp.Mixture = None):
        """Solve stations of the stagnation line (Free, Post-Shock, BLE, Wall) from freestream and wall conditions
        :param Tfree: Freestream temperature.
        :type Tfree: float
        :param Pfree: Freestream pressure.
        :type Pfree: float
        :param Mfree: Freestream Mach number.
        :type Mfree: float
        :param Twall: Wall temperature.
        :type Twall: float
        :param mix: Mixture object.
        :type mix: mutationpp.Mixture"""
        if mix == None: 
            print('No mix given to stagnation_line()! Defaulting to air5...')
            self.mix = mixture_mpp("air5")
        else: self.mix = mix

        self.Tfree = Tfree
        self.Pfree = Pfree
        self.Mfree = Mfree
        self.Twall = Twall
        #Equilibrate the mix with the freesteam conditions:

        self.mix.equilibrate(self.Tfree, self.Pfree)        
        self.gammafree = self.mix.mixtureFrozenGamma()
        self.ufree = self.Mfree*self.mix.frozenSoundSpeed()
        self.mufree = self.mix.viscosity()
        self.rhofree = self.mix.density()
        self.c_i_free = self.mix.Y()
        self.oxygen_mf = self.mix.convert_y_to_ye(self.c_i_free)[self.mix.elementIndex('O')]
        #molecular weight
        self.MW_free = self.mix.mixtureMw()
        
        self.T0_free = stagnation_T(self.Tfree, self.gammafree, self.Mfree)
        self.P0_free = stagnation_P(self.Pfree, self.gammafree, self.Mfree)
        #Total enthalpy at freestream
        self.H0_free = self.mix.mixtureHMass() + (self.Mfree*self.mix.frozenSoundSpeed())**2/2.0

        #Post-shock conditions:
        self.T_post = normal_shock_T(self.Tfree, self.gammafree, self.Mfree)
        self.P_post = normal_shock_P(self.Pfree, self.gammafree, self.Mfree)
        self.rho_post = normal_shock_rho(self.rhofree, self.gammafree, self.Mfree)
        self.M_post = normal_shock_M(self.gammafree, self.Mfree)
        self.u_post = self.M_post*np.sqrt((self.gammafree*self.P_post)/self.rho_post)

        self.T0_post = stagnation_T(self.T_post, self.gammafree, self.M_post)
        self.P0_post = stagnation_P(self.P_post, self.gammafree, self.M_post)
        self.rho0_post = self.rho_post*(1+(self.gammafree - 1) / 2.0 * self.M_post**2)**(1/(self.gammafree - 1))

        #Boundary layer edge conditions
        #Assuming mixture at equilibrium
        self.Te = self.T0_post
        self.Pe = self.P0_post

        #Energy loop (Need to match the Total enthalpy)
        self.mix = energy_loop_brent(self.mix, self.Te, self.Pe, self.H0_free)

        self.Te = self.mix.T()
        self.Pe = self.mix.P()
        self.rhoe = self.mix.density()
        self.mue = self.mix.viscosity()
        self.He = self.mix.mixtureHMass()

        #N O NO N2 O2 according to air_5 from Mutationpp
        self.ce_i = self.mix.Y()
        self.xe_i = self.mix.X()
        self.MWe = self.mix.mixtureMw()

        self.mix.setState(self.mix.densities(), self.Te, 1)
        self.mu_orig_e = self.mix.viscosity()

        #N - 33867025.2 J/Kg heat of formation
        #O - 15432544.8 J/Kg Heat of formation

        #Heat of dissociation                 
        self.Hd = 33867025.2*self.ce_i[0] + 15432544.8 *self.ce_i[1]

        #Wall conditions
        #Assuming mixture at equilibrium
        self.Pwall = self.Pe
        self.rhow = np.zeros(len(Twall))
        self.muw = np.zeros(len(Twall))
        self.Hw = np.zeros(len(Twall))

        for index, T in enumerate(Twall):
            self.mix.equilibrate(T, self.Pwall)
            self.rhow[index] = self.mix.density()
            self.muw[index] = self.mix.viscosity()
            self.Hw[index] = self.mix.mixtureHMass()

        #Adimensional numbers
        #At the moment these values are hardcoded according to several literature sources
        self.Pr = 0.71
        self.Le = 1.0

def compute_aerothermo(titan, options):
    """Fidelity selection for aerothermo computation

    :param titan: TITAN simulation object.
    :type titan: AssemblyList
    :param options: Options object.
    :type options: Options"""

    atmo_model = options.freestream.model
    
    for assembly in titan.assembly:
        #Compute the freestream properties and stagnation quantities
        mix_properties.compute_freestream(atmo_model, assembly.trajectory.altitude, assembly.trajectory.velocity, assembly.Lref, assembly.freestream, assembly, options)
        mix_properties.compute_stagnation(assembly.freestream, options.freestream)

    if options.fidelity.lower() == 'low':
        titan.groups = compute_low_fidelity_aerothermo(titan.assembly, options)
    elif options.fidelity.lower() == 'high':

        if  (assembly.freestream.knudsen <= options.aerothermo.knc_pressure):
            if options.cfd.cfd_restart: su2.restart_cfd_aerothermo(titan, options)
            else: su2.compute_cfd_aerothermo(titan, options)
        else:
            sparta.compute_dsmc_aerothermo(titan, options)

    elif options.fidelity.lower() == 'multi':
        switch.compute_aerothermo(titan, options)
    else:
        raise Exception("Select the correct fidelity options : (Low, High, Multi)")

def compute_aerodynamics(assembly, index : list, flow_direction : np.ndarray, options):
    """Low-fidelity computation of the aerodynamics (pressure, friction)

    :param assembly: Assembly object to process.
    :type assembly: object
    :param index: Integer value for index.
    :type index: int
    :param flow_direction: Unit vector pointing in the flow direction.
    :type flow_direction: np.ndarray
    :param options: Options object.
    :type options: Options"""

    Kn_cont_pressure = options.aerothermo.knc_pressure
    Kn_free = options.aerothermo.knf

    #Pressure calculation only if Drag model is False
    if (not options.vehicle) or (options.vehicle and not options.vehicle.Cd):
        if  (assembly.freestream.knudsen <= Kn_cont_pressure):
            assembly.aerothermo.pressure[index] += aerodynamics_module_continuum(assembly, index, flow_direction) \
                * assembly.aerothermo.partial_factor[index] * options.aerothermo.CP_mult

        elif (assembly.freestream.knudsen >= Kn_free): 
            pressure, shear = aerodynamics_module_freemolecular(assembly, index, flow_direction)
            assembly.aerothermo.pressure[index] += pressure * assembly.aerothermo.partial_factor[index]  * options.aerothermo.CP_mult
            assembly.aerothermo.shear[index] += shear * assembly.aerothermo.partial_factor[index,None] * options.aerothermo.CTau_mult

        else: 
            aerobridge = bridging(assembly.freestream, Kn_cont_pressure, Kn_free )
            pressures, shears = aerodynamics_module_bridging(assembly, index, aerobridge, flow_direction)
            assembly.aerothermo.pressure[index] += pressures * assembly.aerothermo.partial_factor[index] * options.aerothermo.CP_mult
            assembly.aerothermo.shear[index] += shears * assembly.aerothermo.partial_factor[index,None] * options.aerothermo.CTau_mult


def compute_aerothermodynamics(assembly, index : list, flow_direction : np.ndarray, options):
    """Low-fidelity computation of the aerothermodynamics (heat-flux)

    :param assembly: Assembly object to process.
    :type assembly: object
    :param index: Integer value for index.
    :type index: int
    :param flow_direction: Unit vector pointing in the flow direction.
    :type flow_direction: np.ndarray
    :param options: Options or configuration object.
    :type options: Options"""

    Kn_cont_heatflux = options.aerothermo.knc_heatflux       
    Kn_free = options.aerothermo.knf

    StConst = assembly.freestream.density*assembly.freestream.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect    

    # Heatflux calculation for Earth
    if options.planet.name == "earth":
        if  (assembly.freestream.knudsen <= Kn_cont_heatflux):
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_continuum(assembly, index, options)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] * options.aerothermo.CH_mult 

        elif (assembly.freestream.knudsen >= Kn_free): 
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_freemolecular(assembly, index)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] * options.aerothermo.CH_mult 

        else: 
            #atmospheric model for the aerothermodynamics bridging needs to be the NRLSMSISE00
            atmo_model = "NRLMSISE00"
            aerobridge = bridging(assembly.freestream, Kn_cont_heatflux, Kn_free )
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_bridging(assembly, index, flow_direction, atmo_model, Kn_cont_heatflux, Kn_free, options)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] * options.aerothermo.CH_mult 


    elif options.planet.name == "neptune" or options.planet.name == "uranus":
        #https://sci.esa.int/documents/34923/36148/1567260384517-Ice_Giants_CDF_study_report.pdf        
        assembly.aerothermo.heatflux[index] = aerothermodynamics_module_ice_giants(assembly, index, options)


def compute_low_fidelity_aerothermo(assemblies, options) -> list:
    """Low-fidelity aerothermo computation
    
    :param assemblies: List of assembly objects
    :type assemblies: list[object]
    :param options: Options object.
    :type options: Options
    :return: Groups of assemblies as defined by Sphere of Influence.
    :rtype: list[list]"""
    for _assembly in assemblies: del _assembly.aero_index

    #Number of subdivisions
    n = options.aerothermo.subdivision_triangle
    flow_directions, groups, group_map = SoI_assembly_groups(assemblies, options.aerothermo.SoI_rad)
    
    for it, _assembly in enumerate(assemblies):
        _assembly.aerothermo.heatflux *= 0
        _assembly.aerothermo.pressure *= 0
        _assembly.aerothermo.pressure += _assembly.freestream.pressure
        _assembly.aerothermo.shear    *= 0
        _assembly.aerothermo.he       *= 0
        _assembly.aerothermo.hw       *= 0
        _assembly.aerothermo.Te       *= 0
        _assembly.aerothermo.rhoe     *= 0
        _assembly.aerothermo.ue       *= 0
        _assembly.aerothermo.ce_i     *= 0

        _assembly.quaternion_prev = _assembly.quaternion #to be used in thermal model
        flow_direction = flow_directions[group_map[it]]
        if not hasattr(_assembly, 'aero_index'):
            ray_trace(groups[group_map[it]],n, options)
        else: 
            pass

        index = _assembly.aero_index
        compute_aerothermodynamics(_assembly, index, flow_direction, options)
        compute_aerodynamics(_assembly, index, flow_direction, options)
        #if options.pato.flag and options.pato.Ta_bc == "ablation": compute_equilibrium_chemistry(_assembly, options.aerothermo.mixture, index)
        #if options.pato: compute_frozen_chemistry(_assembly, options.aerothermo.mixture)
    return groups


def edge_subdivision(v0 : np.ndarray,v1 : np.ndarray,v2 : np.ndarray, n : int) -> np.ndarray:
    """Each subdivision level divides the triangle into 4 parts with equal areas

    :param v0: Array of first vertex positions.
    :type v0: np.ndarray
    :param v1: Array of second vertex positions.
    :type v1: np.ndarray
    :param v2: Array of third vertex positions.
    :type v2: np.ndarray
    :param n: Number of subdivision levels.
    :type n: int
    :return: Array of centroid positions.
    :rtype: np.ndarray"""

    def COG_subdivision(v0,v1,v2, COG, start, n, i = 1):
        """Recursively subdivides a triangle based upon centroid

        :param v0: Array of first vertex positions.
        :type v0: np.ndarray
        :param v1: Array of second vertex positions.
        :type v1: np.ndarray
        :param v2: Array of third vertex positions.
        :type v2: np.ndarray
        :param COG: Array of centroid positions.
        :type COG: np.ndarray
        :param start: Starting index for the current subdivision.
        :type start: int
        :param n: Number of subdivision levels.
        :type n: int
        :param i: Current subdivision level.
        :type i: int
        :return: Resultant subdivision level.
        :rtype: int"""
    
        v0v1 = (v0 + v1) / 2.0
        v0v2 = (v0 + v2) / 2.0
        v1v2 = (v1 + v2) / 2.0
    
        if i == n:
    
            COG[start+0::4**n,:] = (v0v1 + v0v2 + v0)/3.0
            COG[start+1::4**n,:] = (v0v1 + v1v2 + v1)/3.0
            COG[start+2::4**n,:] = (v0v2 + v1v2 + v2)/3.0
            COG[start+3::4**n,:] = (v0v1 + v0v2 + v1v2)/3.0
    
            return start + 4
    
        else:
            start = COG_subdivision(v0v1,v0v2, v0, COG, start, n, i+1)
            start = COG_subdivision(v0v1,v1, v1v2, COG, start, n, i+1)
            start = COG_subdivision(v0v2,v1v2, v2, COG, start, n, i+1)
            start = COG_subdivision(v0v1,v1v2, v0v2, COG, start, n, i+1)


    if n == 0:
        COG = (v0+v1+v2)/3.0

    else:
        COG = np.zeros((len(v0)*4**n,3))
        COG_subdivision(v0,v1,v2,COG, 0, n)

    return COG

def ray_trace(assembly_group : list, n : int, options, output_rays : str = None):
    """Calls the ray-tracing engine to assign the aero_index parameter across an assembly ensemble

    :param assembly_group: List of assembly objects.
    :type assembly_group: list
    :param n: Number of subdivision levels.
    :type n: int
    :param options: Options object.
    :type options: object
    :param output_rays: Whether to output debug ray geometry, either "leading" or "trailing".
    :type output_rays: str, optional"""

    # Prefilter our raytracing by flow-facing facets
    theta = [0]
    v0 = [[0,0,0]]
    v1 = [[0,0,0]]
    v2 = [[0,0,0]]
    pointers = [0]
    flow_dirs = [[0,0,0]]
    for i_assem, _assembly in enumerate(assembly_group):
        assem_body_flow_vec = -Rot.from_quat(_assembly.quaternion).inv().apply(_assembly.velocity)
        assem_body_flow_vec /= np.linalg.norm(assem_body_flow_vec)
        facet_normals = _assembly.mesh.facet_normal
        length_normals = np.linalg.norm(facet_normals, axis = 1, ord = 2)
        _assembly.aerothermo.theta = np.pi/2 - np.arccos(np.clip(np.sum(- assem_body_flow_vec * facet_normals/length_normals[:,None] , axis = 1), -1.0, 1.0))
        _assembly.freestream.per_facet_mach = np.full_like(length_normals,_assembly.freestream.mach)
        theta = np.hstack([theta,_assembly.aerothermo.theta])
        v0 = np.vstack([v0, _assembly.mesh.v0])
        v1 = np.vstack([v1,_assembly.mesh.v1])
        v2 = np.vstack([v2, _assembly.mesh.v2])
        pointers.append(pointers[-1]+len(length_normals))
        flow_dirs = np.vstack([flow_dirs,[_assembly.velocity for _ in range(len(length_normals))]])
    
    theta     = theta[1:]
    v0        = v0[1:,:]
    v1        = v1[1:,:]
    v2        = v2[1:,:]
    flow_dirs = flow_dirs[1:,:]
    filtered_facet_indices = np.where(theta>0)[0]
    
    flow_dirs = flow_dirs[filtered_facet_indices,:]

    base_assembly = assembly_group[0]
    meshlist = []

    main_Translate_ECEF = trimesh.transformations.translation_matrix(-_assembly.position)
    main_Translate_CoG  = trimesh.transformations.translation_matrix(Rot.from_quat(base_assembly.quaternion).apply(_assembly.COG))
    flow_dirs  = -flow_dirs
    flow_len   = np.linalg.norm(flow_dirs, axis = 1)
    flow_dirs /= flow_len[:,np.newaxis]

    for i_assem, _assembly in enumerate(assembly_group):
        new_mesh = trimesh.Trimesh(vertices=_assembly.mesh.nodes, faces=_assembly.mesh.facets)
        quaternion = np.append([_assembly.quaternion[3]], _assembly.quaternion[0:3])
        R_B_ECEF = trimesh.transformations.quaternion_matrix(quaternion)
        Translate_COG = trimesh.transformations.translation_matrix(-_assembly.COG)
        Translate_ECEF = trimesh.transformations.translation_matrix(_assembly.position)
        #
        Matrix = main_Translate_CoG@main_Translate_ECEF@Translate_ECEF@R_B_ECEF@Translate_COG
        new_mesh.apply_transform(Matrix)
        TransMatrix = Trans.from_matrix(Matrix)
        v0[pointers[i_assem]:pointers[i_assem+1],:] = TransMatrix.apply(
            v0[pointers[i_assem]:pointers[i_assem+1],:]
            )
        v1[pointers[i_assem]:pointers[i_assem+1],:] = TransMatrix.apply(
            v1[pointers[i_assem]:pointers[i_assem+1],:]
            )
        v2[pointers[i_assem]:pointers[i_assem+1],:] = TransMatrix.apply(
            v2[pointers[i_assem]:pointers[i_assem+1],:]
            )
        meshlist.append(new_mesh)

    
    mesh = trimesh.util.concatenate(meshlist)

    ray = RayMeshIntersector(mesh)

    facet_centroids = edge_subdivision(v0[filtered_facet_indices], 
                                       v1[filtered_facet_indices], 
                                       v2[filtered_facet_indices], n)
    for _ in range(n):
        flow_dirs = np.repeat(flow_dirs,4, axis=0)

    ray_origins = facet_centroids - 1e-4*flow_dirs
    ray_directions = -flow_dirs

    ray_directions.shape = (-1,3)
    facet_sees_flow =  np.zeros_like(theta, dtype=np.int16)

    match output_rays:
        case 'leading':
            ray_ends   = facet_centroids - 10*flow_dirs
        case 'trailing':
            ray_ends   = facet_centroids + 10*flow_dirs 
        

    if output_rays is not None:
        if not pathlib.Path(options.output_folder+'/Rays').exists():
            pathlib.Path(options.output_folder+'/Rays').mkdir(parents=True)
        if not hasattr(options, 'n_debug'): options.n_debug = 0
        else: options.n_debug+=1
        mesh.export(options.output_folder+'/Rays/debug_{}.stl'.format(options.n_debug))
        write_rays_to_vtk(options.output_folder+'/Rays/debug_rays_{}.vtk'.format(options.n_debug),ray_origins, ray_ends)

    hits  = ~ray.intersects_any(ray_origins = ray_origins, ray_directions = ray_directions)
    hits.shape = (-1, 4**n)
    hits = np.sum(hits, axis = 1)
    facet_sees_flow[filtered_facet_indices] = hits

    for i_assembly, _assembly in enumerate(assembly_group):

        per_assem_see_flow = facet_sees_flow[pointers[i_assembly]:pointers[i_assembly+1]]
        _assembly.aerothermo.partial_factor = per_assem_see_flow/(4**n)

        per_assem_see_flow = np.arange(len(_assembly.mesh.facets))[per_assem_see_flow != 0]

        _assembly.aerothermo.proj_area = 0
        #proj_facet_areas = _assembly.mesh.facet_area[per_assem_index] * np.dot(_assembly.mesh.facet_normal[per_assem_index],flow_direction)
        #_assembly.aerothermo.proj_area=np.sum(proj_facet_areas)
        _assembly.aero_index = per_assem_see_flow

def shock_angle(M:float, theta_array:np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the shock wave angle beta for an oblique shock for an array of flow turn angles.

    :param M: Mach number.
    :type M: float
    :param theta_array: Array of facet incidence angles in radians.
    :type theta_array: np.ndarray
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :return: Array of shock wave angles in radians.
    :rtype: np.ndarray"""

    # Define the equation for the shock wave angle beta
    def equation(beta : np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Documentation for the function.
        :param beta: Array of shock wave angles in radians.
        :type beta: np.ndarray
        :param theta: Array of facet incidence angles in radians.
        :type theta: np.ndarray
        :return: Return value.
        :rtype: np.ndarray"""
        return np.tan(theta) - 2 * (1 / np.tan(beta)) * ((M**2 * np.sin(beta)**2 - 1) / 
                                                         (M**2 * (gamma + np.cos(2 * beta)) + 2))

    # Array to store the resulting beta angles
    beta_deg_array = []

    # Loop over each theta and solve for beta
    for theta in theta_array:
        # Initial guess for beta (in radians), typically starts just above theta
        beta_initial_guess = theta + np.radians(5)  # A guess slightly larger than theta

        # Solve the equation for beta
        beta_solution = fsolve(equation, beta_initial_guess, args=(theta))[0]
        
        # Convert beta to degrees and append to the result list
        beta_deg_array.append(beta_solution)

    return np.array(beta_deg_array)


def aerodynamics_module_continuum(assembly, p : np.ndarray, flow_direction : np.ndarray) -> np.ndarray:
    """Pressure computation for continuum regime
    :param assembly: Assembly object to process.
    :type assembly: object
    :param p: Value for p.
    :type p: Any
    :param flow_direction: Value for flow direction.
    :type flow_direction: str
    :return: Return value.
    :rtype: Any"""
    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream

    length_normal = np.linalg.norm(facet_normal, axis = 1, ord = 2)

    p = p*(length_normal[p] != 0)

    #assembly.aerothermo.theta[p] =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))
    P0_s = free.P1_s
    Cpmax= (2.0/(free.gamma*free.per_facet_mach[p]**2.0))*((P0_s/free.pressure-1.0))

    #TODO
    if free.mach <= 1.0: Cpmax = 1

    Theta = assembly.aerothermo.theta[p]

    Cp = Cpmax*np.sin(Theta)**2
    Cp[Theta < 0] = 0

    Pressure = Cp[:,None] * 0.5  *free.density * free.velocity**2

    Pressure[np.isnan(Pressure)] = 0
    Pressure.shape = (-1)

    return Pressure


def aerothermodynamics_module_ice_giants(assembly, p : np.ndarray, options) -> np.ndarray:
    """Low-fidelity computation of the aerothermodynamics in the Ice giants planet
:param assembly: Assembly object to process.
:type assembly: object
:param p: Value for p.
:type p: Any
:param options: Options or configuration object.
:type options: object
:return: Return value.
:rtype: Any"""

    length_normal = np.linalg.norm(assembly.mesh.facet_normal, axis = 1, ord = 2)
    p = p*(length_normal[p] != 0)
    
    Theta = assembly.aerothermo.theta[p]

    if options.vehicle:
        nose_radius = options.vehicle.noseRadius
        assembly.aerothermo.qconvstag = 10000*9.08 * np.sqrt(1/(2*nose_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
        assembly.aerothermo.qradstag = 10000*0.091 * nose_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)
        assembly.aerothermo.qstag = assembly.aerothermo.qconvstag+assembly.aerothermo.qradstag

    facet_radius = assembly.mesh.facet_radius[p]
    Qstag = 10000*9.08 * np.sqrt(1/(2*facet_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
    Qradstag = 10000*0.091 * facet_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)

    K = 0.1
    Q = Qstag + Qradstag
    Q = Q*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution
    Q[Q<0] = 0

    return Q

def aerothermodynamics_module_continuum(assembly, p : np.ndarray, options) -> np.ndarray:
    """Heatflux computation for continuum regime
:param assembly: Assembly object to process.
:type assembly: object
:param p: Value for p.
:type p: Any
:param options: Options or configuration object.
:type options: object
:return: Return value.
:rtype: Any"""

    facet_normal = assembly.mesh.facet_normal
    facet_radius = assembly.mesh.facet_radius
    free = assembly.freestream
    body_temperature = assembly.aerothermo.temperature


    def FR(flow, vel_grad):
        """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:return: Return value.
:rtype: Any"""
        q = 0.94*(flow.rhow*flow.muw)**0.1*(flow.rhoe*flow.mue)**0.4*(flow.He - flow.Hw)*np.sqrt(vel_grad)
        return q

    def FR_non_cat(flow, vel_grad):
        """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:return: Return value.
:rtype: Any"""
        q = 0.94*(flow.rhow*flow.muw)**0.1*(flow.rhoe*flow.mue)**0.4*(flow.He - flow.Hw)*np.sqrt(vel_grad)*(1-flow.Hd/flow.He)
        return q

    def VD(flow, vel_grad):
        """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:return: Return value.
:rtype: Any"""
        q = 0.94*(flow.rhoe*flow.mue)**0.5*(flow.He - flow.Hw)*np.sqrt(vel_grad)
        return q

    def SCARAB(flow, radius):
        """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param radius: Numeric value for radius.
:type radius: float
:return: Return value.
:rtype: Any"""

        ## In TITAN is 2*radius because fostrad assumes SCARAB uses diameter ?
        # In addition, Scarab uses that viscosity at stagnation point is given by the power law
        # And chemistry is not accounted for in this scarab formulation 

        # The equation
        #    Re = flow.rhofree * flow.ufree/flow.mue * (2* radius)
        # is replaced by
        Re = flow.rhofree * flow.ufree/(flow.mufree*(flow.T0_post/flow.Tfree)**0.75)
        Re0 = Re * (2* radius)
        St = 2.1/np.sqrt(Re0)
        q = St * 0.5*flow.rhofree*flow.ufree**3
        return q

    def SG(flow, radius):
        """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param radius: Numeric value for radius.
:type radius: float
:return: Return value.
:rtype: Any"""
        #K retrieved from Sutton graves paper
        q =  0.1117*np.sqrt(flow.Pe/radius)*(1/np.sqrt(101325))*(flow.He - flow.Hw)
        return q

    hf_model = options.aerothermo.heat_model

    if options.aerothermo.cat_method.lower() == 'constant':
        cat_rate = options.aerothermo.cat_rate
    elif options.aerothermo.cat_method.lower() == 'material':
        cat_rate = np.ones(len(facet_normal))
        for obj in assembly.objects:
            if obj.material.catalycity != None:
                cat_rate[obj.facet_index] = obj.material.catalycity

        cat_rate = cat_rate[p]
    else:
        raise ValueError("Error in catalicity method (constant or material)")


    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    Theta = assembly.aerothermo.theta[p]

    T0s  = free.T1_s
    P02  = free.P1_s
    h0s  = free.h1_s
    rhos = free.rho_s

    Pr = free.prandtl
    mu_T0s = free.mu_s

    dudx = 1.0/facet_radius* np.sqrt(2*(P02-free.pressure)/rhos)
    flow_ble = None

    StConst = free.density*free.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect (as in Fostrad)

    if free.mach < 1: hf_model = 'vd'

    if hf_model == 'sc': #Scarab formulation and Lees distribution
        # (OLD Fostrad equation)
        Re0norm = free.density * free.velocity / (free.mu *(T0s/free.temperature)**free.omega)
        Re0 = 2.0*facet_radius[p]*Re0norm
        Stc = 2.1/np.sqrt(Re0)
    
    if hf_model == 'vd': #Van Driest
        # (Old Fostrad equation)
        #This Van Driest formula is considering non-reacting flow, thus not accounting for changes in the mixture for the BLE
        Stc = 0.763*(Pr**(-0.6))*(rhos*mu_T0s)**0.5*np.sqrt(dudx[p])*(h0s-free.cp*body_temperature[p])/StConst 

    if hf_model == 'fr': #Fay Riddell
        mix = mixture_mpp(options.aerothermo.mixture)
        flow_ble = stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'fr')
        Stc = q/StConst

    if hf_model == 'fr_noncat': #Fay Riddell
        mix = mixture_mpp(options.aerothermo.mixture)
        flow_ble = stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'fr_noncat')        
        Stc = q/StConst

    if hf_model == 'fr_parcat': #Fay Riddell
        do_pfm = False
        if do_pfm:
            Mfree = free.mach * np.sin(Theta)
            Mfree[Mfree<1] = 1
            flow_ble = stagnation_line_per_facet(Tfree = free.temperature, Pfree = free.pressure, Mfree = Mfree, Twall = body_temperature[p], mixname = options.aerothermo.mixture)
        else: 
            mix = mixture_mpp(options.aerothermo.mixture)
            flow_ble = stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'fr_parcat', cat_rate)       
        Stc = q/StConst

    if hf_model == 'sg': #Sutton_graves
        mix = mixture_mpp(options.aerothermo.mixture)
        flow_ble = stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'sg')
        Stc = q/StConst

    K = 0.1
    Stc = Stc*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution 

    Stc[Stc < 0] = 0
    Stc.shape = (-1)
    if options.thermal.ablation_mode=='byproducts':
        if flow_ble is not None:
            assembly.byproducts.column_height_mix[p] = 0.99*flow_ble.u_post/vel_grad
            assembly.byproducts.rho_mix = flow_ble.rhofree#flow_ble.rhoe
            assembly.byproducts.c_i_mix = flow_ble.c_i_free#flow_ble.ce_i
            assembly.byproducts.T_mix = flow_ble.Twall#flow_ble.Te
            assembly.byproducts.P_mix = flow_ble.Pfree#flow_ble.Pe
            assembly.byproducts.oxy_content = flow_ble.oxygen_mf
        else:
            assembly.byproducts.column_height_mix[p] = np.zeros_like(assembly.aerothermo.theta)
    # if not pathlib.Path('./stagline_stations.csv').resolve().exists():
    #     header = True
    # else: header = False
    # ble_data_columns = ['Altitude']
    # for station in ['free', 'post', 'BLE']:
    #     for data_type in ['Ma_','P_','T_','rho_','%N_','%O_','%NO_','%N2_','%O2_']
    #         ble_data_columns.append(data_type+station)
    #ble_data = None
    # ble_data = [assembly.trajectory.altitude, flow_ble.Mfree, flow_ble.Pfree, flow_ble.Tfree, flow_ble.rhofree]
    # [ble_data.append(c_i) for c_i in flow_ble.c_i_free]
    # [ble_data.append(dat) for dat in [flow_ble.M_post, flow_ble.P_post, flow_ble.T_post, flow_ble.rho_post]]
    # [ble_data.append(c_i) for c_i in energy_loop(.c_pos).Y()]
    # for i_facet, f_id in enumerate(p):
    #     if ble_data is None: ble_data = [[assembly.trajectory.altitude,f_id, flow_ble.Mfree, flow_ble.Pfree, flow_ble.Tfree, rho_free]]
    #with open(str(pathlib.Path('./stagline_stations.csv').resolve()),'w') as f:

    return Stc

def aerothermodynamics_module_freemolecular(assembly, p : np.ndarray) -> np.ndarray:
    """Heatflux computation for free-molecular regime
:param assembly: Assembly object to process.
:type assembly: object
:param p: Value for p.
:type p: Any
:return: Return value.
:rtype: Any"""

    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream
    Wall_Temperature = assembly.aerothermo.temperature

    StConst = free.density*free.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect (as in Fostrad)

    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    #assembly.aerothermo.theta[p] =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    Theta = assembly.aerothermo.theta[p]

    AccCoeff = 1.0 #TODO Wall molecular diffusive accomodation coefficient
    SR = np.sqrt(0.5*free.gamma)*free.per_facet_mach[p]
    
    Q_fm = AccCoeff * free.pressure*np.sqrt(0.5*free.R*free.temperature/np.pi) * \
           ((SR**2 + free.gamma/(free.gamma - 1.0) - (free.gamma + 1.0)/(2 * (free.gamma - 1)) * Wall_Temperature[p] / free.temperature ) * \
           (np.exp(-(SR*np.sin(Theta))**2) + np.sqrt(np.pi) * (SR * np.sin(Theta)) * (1 + special.erf(SR*np.sin(Theta)))) - 0.5 * np.exp(-(SR*np.sin(Theta))**2))


    Stfm = Q_fm/StConst
    Stfm.shape = (-1)

    return Stfm

def aerodynamics_module_freemolecular(assembly, p : np.ndarray, flow_direction : np.ndarray):
    """Pressure computation for Free-molecular regime
:param assembly: Assembly object to process.
:type assembly: object
:param p: Value for p.
:type p: Any
:param flow_direction: Value for flow direction.
:type flow_direction: str
:return: Return value.
:rtype: Any"""

    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream
    body_temperature = assembly.aerothermo.temperature


    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    #assembly.aerothermo.theta[p] =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    Theta = assembly.aerothermo.theta[p]

    SR = np.sqrt(0.5*free.gamma)*free.per_facet_mach[p]
    SN = 1.0 #TODO 0.93
    ST = 1.0 #TODO

    pfm1 = ((2 - SN)/np.sqrt(np.pi)*(SR*np.sin(Theta)) + 0.5*SN*np.sqrt(body_temperature[p]/free.temperature))*np.exp(-(SR*np.sin(Theta))**2.0)
    pfm2 = ((2 - SN)*(SR**2*np.sin(Theta)**2 + 0.5) + 0.5 * SN * np.sqrt(np.pi) * np.sqrt(body_temperature[p]/free.temperature) * (SR*np.sin(Theta)))*(1 + special.erf(SR*np.sin(Theta)))
    pfm = (1/SR**2)*(pfm1+pfm2)
    
    Pressure = pfm[:,None]*(0.5*free.density*free.velocity**2 )
    Pressure[np.isnan(Pressure)] = 0

    tfm = (ST*np.cos(Theta)/SR/np.sqrt(np.pi)) * (np.exp(-(SR*np.sin(Theta))**2.0) + np.sqrt(np.pi) * SR * np.sin(Theta) * (1 + special.erf(SR*np.sin(Theta))))
    Shear = tfm[:,None]*(0.5*free.density*free.velocity**2 )
    Shear[np.isnan(Shear)] = 0

    direction = np.copy(flow_direction)
    direction[1] += 1e-8 # To prevent "bang-on" zero-norm tangent vectors
    direction /= np.linalg.norm(direction)
    direction.shape = (-1)
    direction=np.tile(direction,(len(facet_normal[p]),1))

    tangent_vector = direction - ((direction*facet_normal[p]).sum(axis = 1))[:,None]*facet_normal[p]/(facet_normal[p]*facet_normal[p]).sum(axis=1)[:,None]
    tangent_vector = tangent_vector/np.sqrt((tangent_vector*tangent_vector).sum(axis=1)[:,None])
    
    Pressure.shape = (-1)
    Shear.shape = (-1)

    Shear = Shear[:,None]*tangent_vector
    return Pressure, Shear

def bridging(free, Kn_cont : float, Kn_free: float) -> float:
    """Computation of the bridging factor for the aeordynamic computation
:param free: Value for free.
:type free: Any
:param Kn_cont: Value for kn cont.
:type Kn_cont: Any
:param Kn_free: Value for kn free.
:type Kn_free: Any
:return: Return value.
:rtype: Any"""

    CF_ratiolow  = 0.1508
    CF_ratiohigh = 1e-6
    Kn_trans_R = (np.log(free.knudsen)-np.log(Kn_cont))/(np.log(Kn_free)- np.log(Kn_cont))

    BridgeCF = Kn_trans_R/((1+special.erf(Kn_trans_R*4.0-2.0))/2.0)
    if   (BridgeCF > 1): BridgeCF=(BridgeCF-1)*CF_ratiolow   + 1
    elif (BridgeCF < 1): BridgeCF=(BridgeCF-1)*CF_ratiohigh  + 1

    AeroBridge = (1+special.erf(Kn_trans_R*4-2.0))/2.0*BridgeCF
    return AeroBridge

def aerodynamics_module_bridging(assembly, p : np.ndarray, aerobridge : float, flow_direction : np.ndarray):
    """Pressure computation for Transitional regime
:param assembly: Assembly object to process.
:type assembly: object
:param p: Value for p.
:type p: Any
:param aerobridge: Value for aerobridge.
:type aerobridge: Any
:param flow_direction: Value for flow direction.
:type flow_direction: str
:return: Return value.
:rtype: Any"""

    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream
    wall_temperature = assembly.aerothermo.temperature



    Pcont = aerodynamics_module_continuum(assembly, p, flow_direction)
    Pfree, Sfree = aerodynamics_module_freemolecular(assembly, p, flow_direction)

    Pressure = Pcont + (Pfree - Pcont)* aerobridge
    Shear = 0 + (Sfree - 0)* aerobridge

    return Pressure, Shear

def aerothermodynamics_module_bridging(assembly, p, flow_direction, atm_data, Kn_cont, Kn_free, options):
    """Heatflux computation for the heat-flux regime
:param assembly: Assembly object to process.
:type assembly: object
:param p: Value for p.
:type p: Any
:param flow_direction: Value for flow direction.
:type flow_direction: str
:param atm_data: Value for atm data.
:type atm_data: Any
:param Kn_cont: Value for kn cont.
:type Kn_cont: Any
:param Kn_free: Value for kn free.
:type Kn_free: Any
:param options: Options or configuration object.
:type options: object
:return: Return value.
:rtype: Any"""

    lref = assembly.Lref
    free = assembly.freestream
    facet_radius = assembly.mesh.facet_radius
    facet_normal = assembly.mesh.facet_normal

    #Computes the altitude of which the transition between flow regimes occur
    alt_cont, alt_free = bridging_altitudes(atm_data, Kn_cont, Kn_free, lref)
    
    free_cont = copy(free)
    free_free = copy(free)

    #Computes the freestream properties for the transition altitudes
    mix_properties.compute_freestream(atm_data, alt_cont, free.velocity, lref, free_cont, assembly, options)
    mix_properties.compute_freestream(atm_data, alt_free, free.velocity, lref, free_free, assembly, options)
    
    #HFcont = aerothermodynamics_module_continuum(nodes_normal,nodes_radius, free,p, wall_temperature, flow_direction, hf_model)
    #HFfree = aerothermodynamics_module_freemolecular(nodes_normal,free,p, flow_direction, wall_temperature)


    #Interpolates the data according to experimental values and local radius to obtain a more accurate bridging factor

    Rmodels = np.array([0.0875,   #Mars Micro
                        0.664,    #Pathfinder
                        3.0,      #Average Rn
                        5.3])     #Orion CEV

    Thermal_bridge = np.zeros((4))

    Micro_breaks = np.array([0.001, 0.0017, 0.0063, 0.0261, 0.0583, 0.2903, 0.9300, 2.2, 9.3, 79.8])
    Micro_coeffs = np.array([[-3491400.76967448, 1301.36167961469, 41.2299369761223, 0],
                             [-495178.115347958, 1498.63483389429, 37.9194841961614, 0.0283010726422985],
                             [6337.16885119717, -520.292748574922, 20.4908498052555, 0.185825845944000],
                             [-578.660943327858, -56.7472736230651, 7.34097688561876, 0.436749240415114],
                             [5.85844037619246, -5.49517312716001, 1.89134968638308, 0.594931001233851],
                             [0.0933853668135608, -0.278250671898889, 0.287410511387558, 0.811122770199370],
                             [0.00419312892561080, -0.0220736733955154, 0.0460577235912695, 0.905561385099685],
                             [1.11170515521551e-05, -0.000780200690684111, 0.0102802145472998, 0.937040923399790],
                             [3.30532319579131e-08, -9.75492854066848e-06, 0.000882596445817803, 0.974679444906316],
                             [3.30532319579131e-08, -9.75492854066848e-06, 0.000882596445817803, 0.974679444906316]])


    f2 = PchipInterpolator(Micro_breaks, Micro_coeffs)
    Thermal_bridge[0] = f2(free.knudsen)[3]

    MarsPath_breaks = np.array([0.00103, 0.00357, 0.014, 0.0271, 0.0547, 0.109, 0.206, 0.404, 1.54, 5.03, 24.1, 100])
    MarsPath_coeffs = np.array([[-434289.992260872, 409.141588399997, 27.0574112949722, 0],
                                [8968.42553338165, -518.860453182897, 20.7302546218335, 0.0642487046632124],
                                [-2524.01267973610, -149.524331276682, 12.8337133928574, 0.234196891191710],
                                [-705.575250716640, -51.0802316051753, 7.61673846549975, 0.370984455958549],
                                [78.6842497491937, -24.9647150334351, 3.18467267193635, 0.527461139896373],
                                [12.9643407989090, -5.39482234943265, 1.16950378993431, 0.639378238341969],
                                [-0.633675217101030, -0.649309891750675, 0.488852701875176, 0.713892426289748],
                                [0.0176076589687376, -0.0838525503839340, 0.157198175108222, 0.780310880829016],
                                [9.04305262329855e-05, -0.00484113006748949, 0.0348530210414839, 0.876489515791929],
                                [2.73454165324125e-06, -0.000178526372138471, 0.00436629172811849, 0.943005181347150],
                                [3.79269163767899e-09, -3.99349982287506e-06, 0.000540666345372661, 0.980310880829016],
                                [3.79269163767899e-09, -3.99349982287506e-06, 0.000540666345372661, 0.980310880829016]])

    MarsPath_bridge = MarsPath_coeffs[:,3]+MarsPath_breaks*MarsPath_coeffs[:,2]+MarsPath_breaks**2*MarsPath_coeffs[:,1]+MarsPath_breaks**3*MarsPath_coeffs[:,0]
    
    f2 = PchipInterpolator(MarsPath_breaks, MarsPath_coeffs)
    Thermal_bridge[1] = f2(free.knudsen)[3]

    MeanR_breaks = np.array([0.001, 0.0033, 0.0073, 0.0161, 0.0456, 0.0788, 0.3857, 0.8532, 2.5, 7, 20, 100])
    MeanR_coeffs = np.array([[-25014.0060084375, -130.324225043622, 14.8316002530129, 0],
                             [-93038.9726637940, 216.378730476698, 13.8409792135817, 0.0329853075035311],
                             [2188.18942066116, -233.172651488835, 11.1266074737498, 0.0857301662534536],
                             [-235.155283061758, -46.8133661621039, 7.52283025568612, 0.167256748445935],
                             [-504.650095030751, -13.0877422038533, 4.14422385489454, 0.342486239744378],
                             [2.06975409022096, -3.04277995511635, 1.60872244924224, 0.447152590756571],
                             [0.181093568436558, -0.376124205150528, 0.325996551248476, 0.714092136992955],
                             [0.00289934055336248, -0.0288490000566234, 0.0930563937550737, 0.802795013423228],
                             [9.76192752829730e-05, -0.00259432494971926, 0.0216269952125893, 0.890752495187731],
                             [2.82868265276213e-06, -0.000189872684323298, 0.00420844163855657, 0.944434449872729],
                             [5.88313029489772e-09, -5.11781282977032e-06, 0.000705893951101214, 0.973270323311445],
                             [5.88313029489772e-09, -5.11781282977032e-06, 0.000705893951101214, 0.973270323311445]])

    MeanR_bridge = MeanR_coeffs[:,3]+MeanR_breaks*MeanR_coeffs[:,2]+MeanR_breaks**2*MeanR_coeffs[:,1]+MeanR_breaks**3*MeanR_coeffs[:,0]

    f2 = PchipInterpolator(MeanR_breaks, MeanR_coeffs)
    Thermal_bridge[2] = f2(free.knudsen)[3]

    Orion_breaks = np.array([0.001, 0.0033, 0.0073, 0.0161, 0.04562, 0.0788, 0.3857, 0.8532 ,2.5,7,20,100])
    Orion_coeffs = np.array([[-16779.2722274906, -82.7046724094568, 8.06917824107994, 0],
                            [-87282.7376790222, 249.069257202799, 7.42627785930088, 0.0178457633428624],
                            [16724.0889896745, -294.011209950247, 5.24741349233841, 0.0458902191980685],
                            [-476.210482460133, 14.5441763778172, 3.96516647443931, 0.0807899864845479],
                            [-886.021627665031, 19.1806629801446, 3.57890958610900, 0.198265096011358],
                            [1.04505577211155, -2.91121839021878, 1.92514512255928, 0.305768843456316],
                            [0.186937795116975, -0.471887181374021, 0.433699200216903, 0.652585280865702],
                            [0.00444761436152593, -0.0383048048069601, 0.115051519182560, 0.771306810652743],
                            [0.000111056135577246, -0.00302891238480069, 0.0250748905331701, 0.876756956076944],
                            [3.68304211401379e-06, -0.000216439606659083, 0.00456133930628160, 0.938378478038472],
                            [4.83482194239522e-09, -5.58775291402762e-06, 0.000801211884950431, 0.969189239019236],
                            [4.83482194239522e-09, -5.58775291402762e-06, 0.000801211884950431, 0.969189239019236]])

    Orion_bridge = Orion_coeffs[:,3]+Orion_breaks*Orion_coeffs[:,2]+Orion_breaks**2*Orion_coeffs[:,1]+Orion_breaks**3*Orion_coeffs[:,0]

    f2 = PchipInterpolator(Orion_breaks, Orion_coeffs)
    Thermal_bridge[3] = f2(free.knudsen)[3]

    Thermal_bridge[Thermal_bridge<0] = 0
    Thermal_bridge[Thermal_bridge>1] = 1 

    rN_bridge = np.copy(facet_radius)

    rN_bridge[rN_bridge > 5.3] = 5.3; # The maximum calibrated radius is 5.3m.
    rN_bridge[rN_bridge < 0.0875] = 0.0875; # The minimum calibrated radius is 0.0875m. (Mars Micro Probe)

    fBridge2 = PchipInterpolator(Rmodels, Thermal_bridge)
    BridgeReq = fBridge2(rN_bridge)
    
    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    mix_properties.compute_stagnation(free_cont, options.freestream)
    mix_properties.compute_stagnation(free_free, options.freestream)

    #Compute the Stanton number for both regimes, in the transition altitudes
    Stc = aerothermodynamics_module_continuum(assembly, p, options)
    Stfm = aerothermodynamics_module_freemolecular(assembly, p)

    St = Stc + (Stfm - Stc) * BridgeReq[p]

    St.shape = (-1)
    return St

def bridging_altitudes(model, Kn_cont,Kn_free, lref):
    """Documentation for the function.
:param model: Value for model.
:type model: Any
:param Kn_cont: Value for kn cont.
:type Kn_cont: Any
:param Kn_free: Value for kn free.
:type Kn_free: Any
:param lref: Value for lref.
:type lref: Any
:return: Return value.
:rtype: Any"""

    h_interval = np.linspace(1000,300000,25000)
    altitude_knudsen = mix_properties.interpolate_atmosphere_knudsen(model, lref, h_interval)

    alt_cont = altitude_knudsen(Kn_cont)
    alt_free = altitude_knudsen(Kn_free)

    return alt_cont, alt_free

### Standoff Distance:
def compute_delta(flow, method_delta):
    """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param method_delta: Value for method delta.
:type method_delta: Any
:return: Return value.
:rtype: Any"""
    if method_delta.lower() == 'billig':
        return 0.143*np.exp(3.24/flow.Mfree**2)
    
    if method_delta.lower() == 'lobb':
        return 0.82*flow.rhofree/flow.rho_post

    if method_delta.lower() == 'serbin':
        M = flow.Mfree
        g = flow.gammafree
        return 2.0 / (3.0 * ((((g+1.0)**2*M**2)/(4*g*M**2-2*(g-1.0)))**(1/(g-1))*((g+1)*M**2)/(2+(g-1)*M**2)-1)) 
    
    if method_delta.lower() == 'probstein':
        ratio = flow.rhofree/flow.rho_post
        return ratio/(1-ratio+np.sqrt(8.0/3.0*ratio))
    
    if method_delta.lower() == "freeman":
        return flow.rhofree/flow.rho_post


### Velocity Gradient:
def velocity_gradient(method, radius, flow, method_delta = 'billig'):
    """Documentation for the function.
:param method: Value for method.
:type method: Any
:param radius: Numeric value for radius.
:type radius: float
:param flow: Value for flow.
:type flow: Any
:param method_delta: Value for method delta.
:type method_delta: Any
:return: Return value.
:rtype: Any"""
    if method.lower() == "fr":
        return 1/radius*(np.sqrt(2*(flow.Pe - flow.Pfree)/flow.rhoe))

    if method.lower() == "linnell":
        k = flow.rhofree/flow.rho_post
        return flow.ufree/radius*np.sqrt(flow.rho_post/flow.rhoe*k*(2-k))

    if method.lower() == "newton":
        return flow.ufree/radius

    if method.lower() == "stokes":
        delta = compute_delta(flow, method_delta)
        return 3.0/2.0*flow.u_post/radius*(((1+delta)**3)/((1+delta)**3-1))

    if method.lower() == "olivier":
        delta = compute_delta(flow, method_delta)
        return flow.ufree/radius*(1+delta)/delta*(flow.Pe-flow.P_post)/(flow.rhofree*flow.ufree**2)*(flow.rho_post/flow.rho0_post)# or low.rhoe?? Do I need to equilibrate right after shock?

### Heatflux_equations:
def general_eq(flow, vel_grad, method = "FR", cat_rate = 0):
    """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:param method: Value for method.
:type method: Any
:param cat_rate: Value for cat rate.
:type cat_rate: Any
:return: Return value.
:rtype: Any"""
    q = flow.muw/flow.Pr* \
        detady(flow, vel_grad, method) * \
        dhdeta(flow, method) * (flow.He - flow.Hw) *\
        LAF(flow, method, cat_rate, vel_grad)
    
    return q

#Distance used in heat equations:
def detady(flow, vel_grad, method):
    """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:param method: Value for method.
:type method: Any
:return: Return value.
:rtype: Any"""
    return np.sqrt(2)*flow.rhow*np.sqrt(vel_grad)/(flow.rhoe*flow.mue)**0.5

#Approximation dh/dη
def dhdeta(flow, method):
    """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param method: Value for method.
:type method: Any
:return: Return value.
:rtype: Any"""
    if method.lower() == 'fr' or method.lower() == 'fr_noncat' or method.lower() == 'fr_parcat':
        return 0.54*(flow.rhoe*flow.mue/flow.rhow/flow.muw)**0.9*flow.Pr**0.4
    if method.lower() == 'vd':
        return 0.54*(flow.rhoe*flow.mue/flow.rhow/flow.muw)**1.0*flow.Pr**0.4
    if method.lower() == 'sg':
        return 0.58*(flow.MW_free/flow.MWe)**(1.0/8.0)*(flow.rhoe*flow.mue/flow.rhow/flow.muw)*flow.Pr**0.4*np.sqrt(flow.mu_orig_e/flow.mue)


def coeff_goulard(flow, vel_grad, rate):
    """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:param rate: Value for rate.
:type rate: Any
:return: Return value.
:rtype: Any"""
    #TODO. Not sure what would be the Sc number here, leaving to be approximatly one
    Sc = 1.0
    coeff =  1.0 / (1 + (0.47 * Sc **(-2/3.0) * (2*vel_grad*flow.mue * flow.rhoe) ** 0.5) / (flow.rhow * rate / (2*np.pi * 28.96 / (8.314)/ flow.Twall)) )

    return coeff

#Lewis augmentation factor:
def LAF(flow, method, cat_rate = 0, vel_grad = 0):
    """Documentation for the function.
:param flow: Value for flow.
:type flow: Any
:param method: Value for method.
:type method: Any
:param cat_rate: Value for cat rate.
:type cat_rate: Any
:param vel_grad: Value for vel grad.
:type vel_grad: Any
:return: Return value.
:rtype: Any"""
    if method == 'fr_noncat': return (1 - flow.Hd/flow.He)
    if method == 'fr_parcat': return (1+(flow.Le*coeff_goulard(flow, vel_grad, cat_rate) -1)*flow.Hd/flow.He)
    return 1

def compute_per_facet_flow_dir(assembly,flow_direction, do_pfm=False):
    """Recalculates flow direction on a per-facet basis, returns array of flow directions and array of local Machs
:param assembly: Assembly object to process.
:type assembly: object
:param flow_direction: Value for flow direction.
:type flow_direction: str
:param do_pfm: Value for do pfm.
:type do_pfm: Any
:return: Return value.
:rtype: Any"""

    free = assembly.freestream
    velocity_resultant = free.mach*free.sound*np.tile(flow_direction,[len(assembly.mesh.facet_area),1])
    if do_pfm:
        angular_velocity_vector = np.array([assembly.roll_vel,assembly.pitch_vel,assembly.yaw_vel])
        centroid_radii =  assembly.mesh.facet_COG - assembly.mesh.COG
        tangential_velocity = np.cross(angular_velocity_vector,centroid_radii)
        velocity_resultant -= tangential_velocity
        
    mach_resultant = np.linalg.norm(velocity_resultant,axis=1)/free.sound
    velocity_norm = np.linalg.norm(velocity_resultant, axis=1, keepdims=True)
    velocity_resultant = velocity_resultant/velocity_norm
    return velocity_resultant, mach_resultant


def SoI_assembly_groups(assembly_list : list, sphere_radius : float):
    """Takes a list of assemblies and returns list of assembly groups by sphere of influence as well as mean flow direction and mapping dict
:param assembly_list: List of assembly.
:type assembly_list: list
:param sphere_radius: Value for sphere radius.
:type sphere_radius: Any
:return: Return value.
:rtype: Any"""
    # Firstly convert assembly positions into kd-tree and build base groupings
    positions = np.array([_assembly.position for _assembly in assembly_list])
    tree = KDTree(positions)
    groupings =[[i] for i in range(len(assembly_list))]

    # Then query pairs to find "constellations" of assemblies in space
    assembly_pairs = tree.query_pairs(sphere_radius)
    

    # Which can be assigned back to the "home" assembly
    for pair in assembly_pairs:
        groupings[pair[0]].append(pair[1])


    # Then iterate through groups in descending order (largest groups first)
    group_lengths = [len(group) for group in groupings]
    assembly_in_group = [False for _ in assembly_list]

    group_ids = []
    group_mapping = {}

    for i_group in np.argsort(group_lengths)[::-1]:
        if np.all(assembly_in_group): break # If we've finished assigning assemblies to groups
        # Otherwise assign the assemblies from the biggest group and remove them from other groups
        if len(groupings[i_group])>0:
            group_ids.append(groupings[i_group])
            for assem_id in groupings[i_group]:
                group_mapping[assem_id] = len(group_ids)-1
                assembly_in_group[assem_id] = True
                for group in groupings:
                    if len(group)>0:
                        if (not group==groupings[i_group]) and (assem_id in group):
                            group.remove(assem_id)

    # Finally collect the actual assembly reference and compute a mean flow vector for the group
    assembly_groups = []
    mean_flow_dirs = []

    for assembly_ids in group_ids: 
        assembly_groups.append([])
        [assembly_groups[-1].append(assembly_list[i_assem]) for i_assem in assembly_ids]
        flow_dirs = []
        for _assembly in assembly_groups[-1]:
            flow_dirs.append(-Rot.from_quat(_assembly.quaternion).inv().apply(_assembly.velocity))
            flow_dirs[-1] /= np.linalg.norm(flow_dirs[-1])
            assert np.isclose(np.linalg.norm(flow_dirs[-1]), 1.0)
        mean_flow_dirs.append(np.mean(flow_dirs,axis=0))

    return mean_flow_dirs, assembly_groups, group_mapping

def write_rays_to_vtk(filename : str, origins : np.ndarray, ends : np.ndarray):
    """Write ray segments to a legacy VTK PolyData file.
:param filename: Path to the relevant file.
:type filename: str
:param origins: Value for origins.
:type origins: Any
:param ends: Value for ends.
:type ends: Any"""
    origins = np.asarray(origins)
    ends = np.asarray(ends)

    assert origins.shape == ends.shape
    N = origins.shape[0]

    # Create point list (each ray = 2 points)
    points = np.vstack([origins, ends])
    
    # Connectivity: each ray is a 2-point polyline
    # VTK polyline format:  "2 i j"
    lines = []
    for i in range(N):
        start = i
        end = i + N
        lines.append(f"2 {start} {end}")

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Rays\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Write points
        f.write(f"POINTS {2*N} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # Write lines
        f.write(f"LINES {N} {3*N}\n")
        for line in lines:
            f.write(line + "\n")

    print(f"[write_rays_to_vtk] Wrote {N} rays to {filename}")
