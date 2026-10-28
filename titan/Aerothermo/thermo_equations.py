#
# Copyright (c) 2026 TITAN Contributors (cf. AUTHORS.md).
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
"""Equations used in continuum aerothermodynamics calculations"""

import numpy as np
from functools import partial
from scipy.optimize import brentq

from ..Aerothermo import jump_equations as jump
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

def energy_loop_brent(mix : mpp.Mixture, T_eq : float, P_eq : float, h_ref : float, T_min = 10.0, T_max = 1e5, verbose=False) -> mpp.Mixture:
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
    if verbose: print('Equilibriating energy for h_ref {}'.format(h_ref))
    if verbose: print('{}<T_eq={}<{}  P_eq={}'.format(T_min,T_eq,T_max,P_eq))
    T_eq = brentq(f, T_min, T_eq)
    if verbose: print('Got T_eq as {}'.format(T_eq))
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
        
        self.T0_free = jump.stagnation_T(self.Tfree, self.gammafree, self.Mfree)
        self.P0_free = jump.stagnation_P(self.Pfree, self.gammafree, self.Mfree)
        #Total enthalpy at freestream
        self.H0_free = self.mix.mixtureHMass() + (self.Mfree*self.mix.frozenSoundSpeed())**2/2.0

        #Post-shock conditions:
        self.T_post = jump.normal_shock_T(self.Tfree, self.gammafree, self.Mfree)
        self.P_post = jump.normal_shock_P(self.Pfree, self.gammafree, self.Mfree)
        self.rho_post = jump.normal_shock_rho(self.rhofree, self.gammafree, self.Mfree)
        self.M_post = jump.normal_shock_M(self.gammafree, self.Mfree)
        self.u_post = self.M_post*np.sqrt((self.gammafree*self.P_post)/self.rho_post)

        self.T0_post = jump.stagnation_T(self.T_post, self.gammafree, self.M_post)
        self.P0_post = jump.stagnation_P(self.P_post, self.gammafree, self.M_post)
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
        
        self.T0_free = jump.stagnation_T(self.Tfree, self.gammafree, self.Mfree)
        self.P0_free = jump.stagnation_P(self.Pfree, self.gammafree, self.Mfree)
        #Total enthalpy at freestream
        self.H0_free = self.mix.mixtureHMass() + (self.Mfree*self.mix.frozenSoundSpeed())**2/2.0

        #Post-shock conditions:
        self.T_post = jump.normal_shock_T(self.Tfree, self.gammafree, self.Mfree)
        self.P_post = jump.normal_shock_P(self.Pfree, self.gammafree, self.Mfree)
        self.rho_post = jump.normal_shock_rho(self.rhofree, self.gammafree, self.Mfree)
        self.M_post = jump.normal_shock_M(self.gammafree, self.Mfree)
        self.u_post = self.M_post*np.sqrt((self.gammafree*self.P_post)/self.rho_post)

        self.T0_post = jump.stagnation_T(self.T_post, self.gammafree, self.M_post)
        self.P0_post = jump.stagnation_P(self.P_post, self.gammafree, self.M_post)
        self.rho0_post = self.rho_post*(1+(self.gammafree - 1) / 2.0 * self.M_post**2)**(1/(self.gammafree - 1))

        #Boundary layer edge conditions
        #Assuming mixture at equilibrium
        self.Te = 0.25*(self.Tfree+self.T0_post) # Heuristic guess at output T
        self.Pe = self.P0_post

        #Energy loop (Need to match the Total enthalpy)
        self.mix = energy_loop_brent(self.mix, self.Te, self.Pe, self.H0_free, self.Tfree, self.T0_post, True)

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
### Standoff Distance:
def compute_delta(flow, method_delta):
    """Computes shock standoff distance.

    :param flow: Stagnation conditions.
    :type flow: stagnation_line_per_facet or stagnation_line
    :param method_delta: Shock standoff method.
    :type method_delta: str
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
    """Computes boundary layer velocity gradient.

    :param method: Method for calculating du/dy.
    :type method: str
    :param radius: Numeric value for radius.
    :type radius: float
    :param flow: Stagnation conditions.
    :type flow: stagnation_line_per_facet or stagnation_line
    :param method_delta: Shock standoff method
    :type method_delta: str
    :return: du/dy.
    :rtype: float
    """
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
    """Computes stanton number from input stagnation conditions and velocity gradient.

    :param flow: Stagnation conditions
    :type flow: stagnation_line_per_facet or stagnation_line
    :param vel_grad: Velocity gradient in the boundary layer.
    :type vel_grad: np.ndarray
    :param method: Aerothermodynamics method.
    :type method: Any
    :param cat_rate: Recombinative efficiency (catalycity)
    :type cat_rate: float
    :return: Array of stanton numbers.
    :rtype: np.ndarray"""

    q = flow.muw/flow.Pr* \
        detady(flow, vel_grad, method) * \
        dhdeta(flow, method) * (flow.He - flow.Hw) *\
        LAF(flow, method, cat_rate, vel_grad)
    
    return q

#Distance used in heat equations:
def detady(flow, vel_grad, method):
    """Computes change of self-similar boundary layer coordinate with respect to BL height
    
    :param flow: Stagnation conditions
    :type flow: stagnation_line_per_facet or stagnation_line
    :param vel_grad: Velocity gradient
    :type vel_grad: np.ndarray
    :param method: Value for method.
    :type method: Method for computing dη/dy
    :return: Array of dη/dy
    :rtype: np.ndarray"""
    return np.sqrt(2)*flow.rhow*np.sqrt(vel_grad)/(flow.rhoe*flow.mue)**0.5

#Approximation dh/dη
def dhdeta(flow, method):
    """Approximates rate of change of enthalpy with respect to boundary layer coordinate.

    :param flow: Stagnation conditions.
    :type flow: stagnation_line_per_facet or stagnation_line
    :param method: Method for calculating dh/dη
    :type method: str
    :return: Array of dh/dη.
    :rtype: np.ndarray"""
    if method.lower() == 'fr' or method.lower() == 'fr_noncat' or method.lower() == 'fr_parcat':
        return 0.54*(flow.rhoe*flow.mue/flow.rhow/flow.muw)**0.9*flow.Pr**0.4
    if method.lower() == 'vd':
        return 0.54*(flow.rhoe*flow.mue/flow.rhow/flow.muw)**1.0*flow.Pr**0.4
    if method.lower() == 'sg':
        return 0.58*(flow.MW_free/flow.MWe)**(1.0/8.0)*(flow.rhoe*flow.mue/flow.rhow/flow.muw)*flow.Pr**0.4*np.sqrt(flow.mu_orig_e/flow.mue)


def coeff_goulard(flow, vel_grad, rate):
    """Computes the Goulard coefficient

    :param flow: Stagnation conditions
    :type flow: stagnation_line_per_facet or stagnation_line
    :param vel_grad: Velocity gradient in boundary layer
    :type vel_grad: np.ndarray
    :param rate: Catalycity rate/recombinative efficiency.
    :type rate: float or np.ndarray
    :return: Array of Goulard coefficients.
    :rtype: np.ndarray"""
    #TODO. Not sure what would be the Sc number here, leaving to be approximatly one
    Sc = 1.0
    coeff =  1.0 / (1 + (0.47 * Sc **(-2/3.0) * (2*vel_grad*flow.mue * flow.rhoe) ** 0.5) / (flow.rhow * rate / (2*np.pi * 28.96 / (8.314)/ flow.Twall)) )

    return coeff

#Lewis augmentation factor:
def LAF(flow, method, cat_rate = 0., vel_grad = 0.):
    """Computes the Lewis augmentation factor (LAF)

    :param flow: Stagnation conditions
    :type flow: stagnation_line_per_facet or stagnation_line
    :param method: Method for Lewis augmentation factor
    :type method: str
    :param cat_rate: Catalycity rate/recombinative efficiency, defaults to 0.
    :type cat_rate: float, optional
    :param vel_grad: Velocity gratient, defaults to 0.
    :type vel_grad: float or np.ndarray, optional
    :return: Array of Lewis Augmentation Factors
    :rtype: np.ndarray
    """
    if method == 'fr_noncat': return (1 - flow.Hd/flow.He)
    if method == 'fr_parcat': return (1+(flow.Le*coeff_goulard(flow, vel_grad, cat_rate) -1)*flow.Hd/flow.He)
    return 1




