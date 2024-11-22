from Uncertainty.dynamics_tools import add_airspeed_velocity_NED
import os, pathlib
import pandas as pd
import datetime as dt
from scipy.interpolate import PchipInterpolator
import pickle
import numpy as np
from scipy import stats
from Freestream.mix_mpp import mixture_mpp
import copy
def add_wind(assembly,options):
    if options.gram.wind:
        wind = get_wind_vector(assembly.trajectory.altitude,options)
        assembly=add_airspeed_velocity_NED(assembly, wind)

def construct_reference_traj(f,options):
    if not os.path.exists(options.output_folder + '/GRAM/gramTraj.csv'):
        print('Perturbing atmosphere...')
        try:
            data = pd.read_csv(options.output_folder + "/Data/data.csv")
        except Exception as e: raise Exception('Could not load reference trajectory! Error:{}'.format(e))
        clipped_data = data[['Time', 'Altitude', 'Latitude', 'Longitude']].copy()
        clipped_data['Altitude'] = clipped_data['Altitude'].multiply(0.001)
        clipped_data.to_csv(options.output_folder + '/GRAM/gramTraj.csv', index=False, header=False)

        if os.path.exists(options.output_folder+'/GRAM/gramSpecies.pkl'):
            pathlib.Path(options.output_folder+'/GRAM/gramSpecies.pkl').unlink()

        if os.path.exists(options.output_folder+'/GRAM/gramWind.pkl'):
            pathlib.Path(options.output_folder+'/GRAM/gramWind.pkl').unlink()
            
    f.write("  UseTrajectoryFile     = 1 \n")
    f.write("  TrajectoryFileName    ='" + options.output_folder + "/GRAM/gramTraj.csv'\n")

    return f

def perturbGRAM(f,options):
    options.gram.Seed = str(os.getpid()+dt.datetime.now().microsecond) if options.gram.Seed.lower() == 'auto' else options.gram.Seed
    f.write("  InitialRandomSeed               = " + options.gram.Seed+ "\n")
    f.write("  RandomPerturbationScale         = 1\n")
    f.write("  HorizontalWindPerturbationScale = 1\n")
    f.write("  VerticalWindPerturbationScale   = 1\n")
    f.write("  NumberOfMonteCarloRuns          = 1\n")
    return f

def get_wind_vector(altitude, options):
    if options.gram.reference and os.path.exists(options.output_folder+'/GRAM/gramWind.pkl'):
        with open(options.output_folder + '/GRAM/gramWind.pkl', 'rb') as file: interp_data = pickle.load(file)
    else:
        try: data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")
        except: return [0.0,0.0,0.0]

        n_str = 'PerturbedNSWind_ms' if options.gram.Uncertain else 'NSWind_ms'
        e_str = 'PerturbedEWWind_ms' if options.gram.Uncertain else 'EWWind_ms'
        d_str = 'PerturbedVerticalWind_ms' if options.gram.Uncertain else 'VerticalWind_ms'

        heights = data['Height_km'].to_numpy() * 1000 if options.gram.reference else altitude
        n_points = len(heights) if isinstance(heights,np.ndarray) else 1
        vectors = np.zeros((4, n_points))

        vectors[0, :] = heights
        vectors[1, :] = data[n_str].to_numpy()
        vectors[2, :] = data[e_str].to_numpy()
        vectors[3, :] = -1*data[d_str].to_numpy()

        if not options.gram.reference: return vectors[1:, 0]

        _, unique_alts = np.unique(vectors[0, :], return_index=True)

        interp_data=vectors[:, unique_alts]

        with open(options.output_folder+'/GRAM/gramWind.pkl','wb') as file: pickle.dump(interp_data, file)

    #interp_data=np.transpose(interp_data)
    # f = PchipInterpolator(interp_data[:, 0], interp_data, axis=0, extrapolate=False)
    # vector = f(altitude)
    vector = [np.interp(altitude,interp_data[0,:],interp_data[i_dataseries,:]) for i_dataseries in range(np.shape(interp_data)[0])]
    if np.isnan(vector).any(): vector = interp_data[-1,:]
    return vector[1:]

def pull_freestream_stats(options,velocity):
    data = pd.read_csv(options.output_folder+"/GRAM/OUTPUT.csv")

    temp_mu = 'Temperature_K'
    dens_mu = 'Density_kgm3'
    v_mu = 'WindSpeed_ms'
    names = ['temperature','density','velocity']
    std_strs = ['TemperatureStandardDeviation_pct','DensityStandardDeviation_pct','WindSpeedStandardDeviation_pct']
    means = np.array([data[temp_mu].to_numpy(),data[dens_mu].to_numpy(),data[v_mu].to_numpy()]).flatten()
    means[2] = velocity

    stds = [(mu * (0.01) * data[std_strs[i_mu]].to_numpy()[0]) for i_mu, mu in enumerate(means)]
    stds[2] = (data['WindSpeed_ms'].to_numpy()[0]/3)
    freestream_stats = {}
    for i_name, name in enumerate(names):freestream_stats[name]= [means[i_name],stds[i_name]]
    return freestream_stats


def mpp_solve_freestream(temperature,density,velocity,freestream):
        freestream.percent_mass.shape = (1,-1)

        dens_array = freestream.percent_mass*density
        freestream.temperature = temperature
        freestream.density = density
        freestream.velocity = velocity


        mix = mixture_mpp(species = freestream.species_index, temperature = temperature, density = dens_array.flatten().tolist())
        freestream.percent_mole = mix.X()
        freestream.percent_mass = mix.Y()
        freestream.percent_mole.shape = (1,-1)
        freestream.pressure = mix.P()
        freestream.R = mix.P()/(mix.density()*mix.T())
        freestream.gamma = mix.mixtureFrozenGamma()
        freestream.cp = mix.mixtureFrozenCpMass()
        freestream.cv = mix.mixtureFrozenCvMass()
        freestream.mu = mix.viscosity()
        freestream.sound = mix.frozenSoundSpeed()
        freestream.mach = freestream.velocity/freestream.sound


        avo = 6.0221408E+23 
        m_mean = mix.mixtureMw()/avo
        k = 2.64638e-3*freestream.temperature**1.5/(freestream.temperature+245*10**(-12/freestream.temperature))
        freestream.prandtl = freestream.mu*freestream.cp/k
        freestream.percent_mass.shape = (1,-1)
        return freestream