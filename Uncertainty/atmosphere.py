from Freestream.gram import get_wind_vector
from Uncertainty.dynamics_tools import add_airspeed_velocity_NED
import os, pathlib
import pandas as pd
import datetime as dt

def add_wind(assembly,options):
    if options.gram.wind:
        wind = get_wind_vector(assembly.trajectory.altitude,options)
        assembly=add_airspeed_velocity_NED(assembly, wind)

def setupGRAM(assembly,options):
    if not os.path.exists(options.output_folder + "/Data/data.csv"):
        raise Exception('Uncertainty in GRAM requires a reference trajectory, please simulate a certain trajectory first')

    options.gram.isPerturbed = 1
    print('Perturbing Atmosphere...')

    if os.path.exists(options.output_folder+'/GRAM/gramSpecies.pkl'):
        pathlib.Path(options.output_folder+'/GRAM/gramSpecies.pkl').unlink()

    if os.path.exists(options.output_folder+'/GRAM/gramWind.pkl'):
        pathlib.Path(options.output_folder+'/GRAM/gramWind.pkl').unlink()

    if not os.path.exists(options.output_folder + '/GRAM/gramTraj.csv'):
        data = pd.read_csv(options.output_folder + "/Data/data.csv")
        clipped_data = data[['Time', 'Altitude', 'Latitude', 'Longitude']].copy()
        clipped_data['Altitude'] = clipped_data['Altitude'].multiply(0.001)
        clipped_data.to_csv(options.output_folder + '/GRAM/gramTraj.csv', index=False, header=False)

def perturbGRAM(f,options):
    seed = str(os.getpid()+dt.datetime.now().microsecond) if options.gram.Seed.lower() == 'auto' else options.gram.Seed

    f.write("  InitialRandomSeed               = " + seed + "\n")
    f.write("  RandomPerturbationScale         = 1\n")
    f.write("  HorizontalWindPerturbationScale = 1\n")
    f.write("  VerticalWindPerturbationScale   = 1\n")
    f.write("  NumberOfMonteCarloRuns          = 1\n")
    f.write("  UseTrajectoryFile     = 1 \n")
    f.write("  TrajectoryFileName    ='" + options.output_folder + "/GRAM/gramTraj.csv'\n")

    return f
