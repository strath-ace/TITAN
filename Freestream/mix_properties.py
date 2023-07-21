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
from Freestream.atmosphere import load_atmosphere, retrieve_atmosphere_data
from Freestream.mix_mpp import mixture_mpp
from Freestream import gram
import numpy as np
from scipy.interpolate import interp1d

#Hard sphere model
def compute_percent_mole(species_index, percent_mass):

    percent_mole = np.copy(percent_mass)

    mN2 = 28.01340;               #molar mass of nitrogen molecule, grams/mole
    mO2 = 31.99880;               #molar mass of oxigen molecule, grams/mole
    mO = mO2/2;                   #molar mass of oxigen atom, grams/mole
    mN = mN2/2;                   #molar mass of Nitrogen atom, grams/mole
    mAr = 39.9480;                #molar mass of Argon molecule, grams/mole
    mHe = 4.0026020;              #molar mass of helium molecule, grams/mole
    mH= 1.007940;                 #molar mass of Hydrogen molecule, grams/mole

    for index,specie in enumerate(species_index):
        if specie == "N2":    percent_mole[:,index] /= mN2
        if specie == "O2":    percent_mole[:,index] /= mO2
        if specie == "O" :    percent_mole[:,index] /= mO 
        if specie == "N" :    percent_mole[:,index] /= mN 
        if specie == "Ar":    percent_mole[:,index] /= mAr
        if specie == "He":    percent_mole[:,index] /= mHe
        if specie == "H" :    percent_mole[:,index] /= mH    

    percent_mole = percent_mole/np.sum(percent_mole, axis = -1)[:,None]

    return percent_mole

def compute_gas_contant_mean(species_index, percent_gas):
    
    molar_mass_mean = 0

    mN2 = 28.01340;               #molar mass of nitrogen molecule, grams/mole
    mO2 = 31.99880;               #molar mass of oxigen molecule, grams/mole
    mO = mO2/2;                   #molar mass of oxigen atom, grams/mole
    mN = mN2/2;                   #molar mass of Nitrogen atom, grams/mole
    mAr = 39.9480;                #molar mass of Argon molecule, grams/mole
    mHe = 4.0026020;              #molar mass of helium molecule, grams/mole
    mH= 1.007940;                 #molar mass of Hydrogen molecule, grams/mole

    for index,specie in enumerate(species_index):
        if specie == "N2":    molar_mass_mean += mN2*percent_gas[:,index]/1E3
        if specie == "O2":    molar_mass_mean += mO2*percent_gas[:,index]/1E3
        if specie == "O" :    molar_mass_mean += mO *percent_gas[:,index]/1E3
        if specie == "N" :    molar_mass_mean += mN *percent_gas[:,index]/1E3
        if specie == "Ar":    molar_mass_mean += mAr*percent_gas[:,index]/1E3
        if specie == "He":    molar_mass_mean += mHe*percent_gas[:,index]/1E3
        if specie == "H" :    molar_mass_mean += mH *percent_gas[:,index]/1E3   

    R_mean = 8.314472/molar_mass_mean

    return R_mean

def compute_gamma_mean(species_index, percent_gas):
    gamma_mean = 0

    for index,specie in enumerate(species_index):
        if specie == "N2":    gamma_mean += 7.0/5.0*percent_gas[:,index]
        if specie == "O2":    gamma_mean += 7.0/5.0*percent_gas[:,index]
        if specie == "O" :    gamma_mean += 5.0/3.0*percent_gas[:,index]
        if specie == "N" :    gamma_mean += 5.0/3.0*percent_gas[:,index]
        if specie == "Ar":    gamma_mean += 5.0/3.0*percent_gas[:,index]
        if specie == "He":    gamma_mean += 5.0/3.0*percent_gas[:,index]
        if specie == "H" :    gamma_mean += 5.0/3.0*percent_gas[:,index]   

    return gamma_mean

def compute_cp_mean(species_index, percent_gas, temperature):
#N,O,N2,O2,H,He,Ar

    cp_mean = 0
    T = temperature
    R = 8.314472

    mN2 = 28.01340/1E3;               #molar mass of nitrogen molecule, kg/mole
    mO2 = 31.99880/1E3;               #molar mass of oxigen molecule,   kg/mole
    mO = mO2/2.0;                   #molar mass of oxigen atom,       kg/mole
    mN = mN2/2.0;                   #molar mass of Nitrogen atom,     kg/mole
    mAr = 39.9480/1E3;                #molar mass of Argon molecule,    kg/mole
    mHe = 4.0026020/1E3;              #molar mass of helium molecule,   kg/mole
    mH= 1.007940/1E3;                 #molar mass of Hydrogen molecule, kg/mole

    poli_N2 = np.array([3.53100528,-1.23660987E-04, -5.02999437E-07, 2.43530612E-09, -1.40881235E-12])
    poli_O2 = np.array([3.78246636,-2.99673416E-03, 9.84730200E-06, -9.68129508E-09, 3.24372836E-12  ])
    poli_O  = np.array([3.16826710,-3.27931884E-03, 6.64306396E-06, -6.12806624E-09, 2.11268971E-12]) 
    poli_N  = np.array([2.5,        0,               0,               0,               0              ])
    poli_Ar = np.array([2.59316097,-1.32892944E-03, 5.26503944E-06, -5.97956691E-09, 2.18967862E-12  ])
    poli_He = np.array([2.5,        0,               0,               0,               0              ])
    poli_H  = np.array([2.5,        0,               0,               0,               0              ])

    for index,specie in enumerate(species_index):
        if specie == "N2":    
            poli = poli_N2
            cp_mean += R/mN2*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]
        if specie == "O2":    
            poli = poli_O2
            cp_mean += R/mO2*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]
        if specie == "O" :    
            poli = poli_O
            cp_mean += R/mO*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]
        if specie == "N" :    
            poli = poli_N
            cp_mean += R/mN*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]
        if specie == "Ar":    
            poli = poli_Ar
            cp_mean += R/mAr*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]
        if specie == "He":    
            poli = poli_He
            cp_mean += R/mHe*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]
        if specie == "H" :    
            poli = poli_H
            cp_mean += R/mH*(1*poli[0]+T*poli[1]+T**2*poli[2]+T**3*poli[3]+T**4*poli[4])*percent_gas[:,index]

    return cp_mean


def compute_mass_mean(species_index, percent_gas):

    mN2 = 28.01340;               #molar mass of nitrogen molecule, grams/mole
    mO2 = 31.99880;               #molar mass of oxigen molecule, grams/mole
    mO = mO2/2;                   #molar mass of oxigen atom, grams/mole
    mN = mN2/2;                   #molar mass of Nitrogen atom, grams/mole
    mAr = 39.9480;                #molar mass of Argon molecule, grams/mole
    mHe = 4.0026020;              #molar mass of helium molecule, grams/mole
    mH= 1.007940;                 #molar mass of Hydrogen molecule, grams/mole

    mass_mean = 0

    for index,specie in enumerate(species_index):
        if specie == "N2":    mass_mean += mN2*percent_gas[:,index]/1E3
        if specie == "O2":    mass_mean += mO2*percent_gas[:,index]/1E3
        if specie == "O" :    mass_mean += mO* percent_gas[:,index]/1E3
        if specie == "N" :    mass_mean += mN* percent_gas[:,index]/1E3
        if specie == "Ar":    mass_mean += mAr*percent_gas[:,index]/1E3
        if specie == "He":    mass_mean += mHe*percent_gas[:,index]/1E3
        if specie == "H" :    mass_mean += mH* percent_gas[:,index]/1E3   

    return mass_mean

def compute_omega_mean(species_index, percent_gas):

    omega_mean = 0
    
    for index,specie in enumerate(species_index):
        if specie == "N2":  omega_mean += 0.74*percent_gas[:,index]
        if specie == "O2":  omega_mean += 0.77*percent_gas[:,index]
        if specie == "O" :  omega_mean += 0.8* percent_gas[:,index]
        if specie == "N" :  omega_mean += 0.8* percent_gas[:,index]
        if specie == "Ar":  omega_mean += 0.81*percent_gas[:,index]
        if specie == "He":  omega_mean += 0.66*percent_gas[:,index]
        if specie == "H" :  omega_mean += 0.8* percent_gas[:,index]

    return omega_mean

def compute_diameter_mean(species_index, percent_gas):

    diameter_mean = 0
    
    for index,specie in enumerate(species_index):
        if specie == "N2":  diameter_mean += 3.784E-10*percent_gas[:,index]
        if specie == "O2":  diameter_mean += 3.636E-10*percent_gas[:,index]
        if specie == "O" :  diameter_mean += 3.000E-10*percent_gas[:,index]
        if specie == "N" :  diameter_mean += 3.000E-10*percent_gas[:,index]
        if specie == "Ar":  diameter_mean += 3.659E-10*percent_gas[:,index]
        if specie == "He":  diameter_mean += 2.330E-10*percent_gas[:,index]
        if specie == "H" :  diameter_mean += 3.000E-10*percent_gas[:,index]

    return diameter_mean

def compute_sutherland(species_index, percent_gas, temperature):

    S1 = np.zeros((len(species_index),1))
    S2 = np.zeros((len(species_index),1))

    for index,specie in enumerate(species_index):
        if specie == "N2":  S1[index] = 111;  S2[index] = 1.4067E-6
        if specie == "O2":  S1[index] = 127;  S2[index] = 1.6934E-6
        if specie == "O":   S1[index] = 127;  S2[index] = 1.6934E-6
        if specie == "N":   S1[index] = 111;  S2[index] = 1.4067E-6
        if specie == "Ar":  S1[index] = 144;  S2[index] = 21.250E-6
        if specie == "He":  S1[index] = 79.4; S2[index] = 1.48438E-6
        if specie == "H":   S1[index] = 72;   S2[index] = 0.63624E-6

    S1mix = np.dot(percent_gas,S1)  
    S2mix = np.dot(percent_gas,S2)

    muSu = (S2mix * temperature**(3.0/2.0))/(temperature + S1mix)
    return muSu


def compute_freestream( model, altitude, velocity, lref, freestream, assembly, options):
    """
    Compute the freestream properties

    The user needs to specify the method for the freestream computation (Standard, Mutationpp)

    Parameters
    ----------
    model: str
        Name of the atmospheric model
    altitude: float
        Altitude value in meters
    velocity: float
        Velocity value in meters
    lref: float
        Refence length in meters
    freesteam: Freestream
        Object of class assembly.freestream
    options: Options
        Object of class Options
    """

    data, species_index = retrieve_atmosphere_data(model, altitude, assembly, options)

    temperature = data[1]
    density = data[2:]

    freestream.species_index = species_index
    freestream.percent_mass = density/np.sum(density)
    freestream.percent_mass.shape = (1,-1)

    freestream.temperature = temperature
    freestream.density = np.sum(density)
    freestream.velocity = velocity

    #Avogadro number
    avo = 6.0221408E+23 

    if options.freestream.method.lower() == "mutationpp":

        mix = mixture_mpp(species = species_index, temperature = temperature, density = density)
        
        freestream.percent_mole = mix.X()
        freestream.percent_mole.shape = (1,-1)
        freestream.pressure = mix.P()
        freestream.R = mix.P()/(mix.density()*mix.T())
        freestream.gamma = mix.mixtureFrozenGamma()
        freestream.cp = mix.mixtureFrozenCpMass()
        freestream.cv = mix.mixtureFrozenCvMass()
        freestream.mu = mix.viscosity()
        freestream.sound = mix.frozenSoundSpeed()
        freestream.mach = freestream.velocity/freestream.sound

        m_mean = mix.mixtureMw()/avo

        k = 2.64638e-3*freestream.temperature**1.5/(freestream.temperature+245*10**(-12/freestream.temperature))
        freestream.prandtl = freestream.mu*freestream.cp/k

    elif options.freestream.method.lower() == "standard":
        if options.planet.name != "earth": raise Exception("The Standard method only works for Earth. Needs further data for other chemical species")
        
        freestream.percent_mole = compute_percent_mole(species_index = freestream.species_index, percent_mass = freestream.percent_mass)
        freestream.R = compute_gas_contant_mean(species_index = freestream.species_index, percent_gas = freestream.percent_mole)[0]
        freestream.pressure =  freestream.density * freestream.R * freestream.temperature
        freestream.gamma = compute_gamma_mean(species_index = freestream.species_index, percent_gas = freestream.percent_mole)[0]
        freestream.cp = compute_cp_mean(species_index = freestream.species_index, percent_gas = freestream.percent_mole, temperature = freestream.temperature)[0]
        freestream.mu = compute_sutherland(species_index = freestream.species_index, percent_gas = freestream.percent_mole, temperature = freestream.temperature)[0][0]
        freestream.sound = np.sqrt((freestream.gamma*freestream.pressure)/freestream.density)
        m_mean = (compute_mass_mean(species_index = freestream.species_index, percent_gas = freestream.percent_mole)/avo)[0]

        freestream.mach = freestream.velocity/freestream.sound

        k = 2.64638e-3*freestream.temperature**1.5/(freestream.temperature+245*10**(-12/freestream.temperature))
        freestream.prandtl = freestream.mu*freestream.cp/k

    elif options.freestream.method.upper() == "GRAM":
        if options.freestream.model.upper() != "GRAM": raise Exception("The freestream properties can only be retrieved through the use of the GRAM model")
        if options.planet.name == "earth": raise Exception ("The aerothermodynamic models used for Earth need to be computed using the Standard or Mutationpp method")

        data = gram.read_gram(assembly, options)

        freestream.temperature = data['Temperature_K'].to_numpy()[0]
        freestream.density = data['Density_kgm3'].to_numpy()[0]
        freestream.pressure = data['Pressure_Pa'].to_numpy()[0]
        freestream.sound = data['SpeedOfSound_ms'].to_numpy()[0]
        freestream.mach = assembly.freestream.velocity/assembly.freestream.sound
        freestream.gamma = data['SpecificHeatRatio'].to_numpy()[0]
        freestream.R = data['SpecificGasConstant_JkgK'].to_numpy()[0]

        k = 2.64638e-3*freestream.temperature**1.5/(freestream.temperature+245*10**(-12/freestream.temperature))
        freestream.prandtl = freestream.mu*freestream.cp/k

    else:
        raise Exception("Freestream method not found")

    #The species data at the moment is only available for Earth
    #Missing CH4 -> afterwards it can be computed for Neptune and Uranus
    if options.planet.name == "earth":
        
        d_mean = compute_diameter_mean(species_index, freestream.percent_mole)[0]
        freestream.diameter = d_mean
        omega_mean = compute_omega_mean(species_index, freestream.percent_mole)[0]
        freestream.omega = omega_mean
    
        C0 = (5.0*m_mean/16)*np.sqrt(np.pi*freestream.R)/(np.pi*d_mean**2)
        C1 = 2*C0/(15*np.sqrt(2*np.pi*freestream.R))*(5-2*omega_mean)*(7-2*omega_mean)
    
        freestream.mfp = C1/freestream.density
        freestream.knudsen = freestream.mfp/lref
    
def compute_stagnation(free, options):
    """
    Compute the post-shock stagnation values

    Parameters
    ----------
    free: Freestream
        Object of class assembly.freestream
    options: Options
        Object of class Options
    """

    if free.mach >= 1.0:
        free.P1_s = free.pressure *(0.5*(free.gamma+1.0)*free.mach**2.0)**(free.gamma/(free.gamma-1.0)) * ((free.gamma+1.0)/(2.0*free.gamma*free.mach**2.0 - (free.gamma-1.0)))**(1.0/(free.gamma-1.0))
    else:
        free.P1_s = 0.5*free.density*free.velocity**2 + free.pressure #free.pressure *(0.5*(free.gamma+1.0)*free.mach**2.0)**(free.gamma/(free.gamma-1.0))

    free.T1_s  = free.temperature * (1 + 0.5 * (free.gamma-1) * free.mach**2)    

    free.h1_s  = free.cp * free.T1_s
    free.rho_s = free.P1_s/free.R/free.T1_s

    if options.method == "Mutationpp":
        mix = mixture_mpp(species = free.species_index, temperature = free.temperature, density = free.density*free.percent_mass.reshape((-1)))
        free.mu_s = mix.viscosity()

    elif options.method == "Standard":
        free.mu_s = compute_sutherland(species_index = free.species_index, percent_gas = free.percent_mole, temperature = free.T1_s)

def interpolate_atmosphere_knudsen(name, lref, altitude):

    avo = 6.0221408E+23 

    #This is only possible for the NRLSMISE00 at the moment
    f_values, species_index = load_atmosphere(name = name)

    data = f_values(altitude)
    
    temperature = data[:,1]
    density = data[:,2:]
    
    percent_mass = density/np.sum(density, axis = 1)[:,None]
    percent_mole = compute_percent_mole(species_index = species_index, percent_mass = percent_mass)
    R = compute_gas_contant_mean(species_index = species_index, percent_gas = percent_mole)
    m_mean = compute_mass_mean(species_index = species_index, percent_gas = percent_mole)/avo
    d_mean = compute_diameter_mean(species_index = species_index, percent_gas = percent_mole)
    omega_mean = compute_omega_mean(species_index = species_index, percent_gas = percent_mole)

    C0 = (5.0*m_mean/16)*np.sqrt(np.pi*R)/(np.pi*d_mean**2)
    C1 = 2*C0/(15*np.sqrt(2*np.pi*R))*(5-2*omega_mean)*(7-2*omega_mean)

    mfp = C1/np.sum(density, axis = 1)
    knudsen = mfp/lref

    f=interp1d(knudsen, altitude, kind = 'cubic')

    return f