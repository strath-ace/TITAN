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
"""Rankine-Hugioniot jump equations used in continuum aerothermodynamics calculations"""
import numpy as np
from scipy.optimize import fsolve
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
    :rtype: float
    """
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