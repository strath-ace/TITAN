import os
import pandas as pd
from Freestream import gram
from Configuration.options import Options
from Model.planet import ModelPlanet

# -------------------------------------------------------
# INITIALISE OPTIONS FOR A SINGLE MARS ATMOSPHERE QUERY
# -------------------------------------------------------

# Load TITAN config template
options = Options("Config_template.cfg")

# Force planet = Mars
options.planet = ModelPlanet("mars")

# Ensure GRAM output folder exists
os.makedirs("GRAM_SAMPLES", exist_ok=True)


def sample_mars_atmosphere(alt_km, lat_deg=0.0, lon_deg=0.0):
    """
    Runs a single MarsGRAM atmosphere query at a given altitude and location.
    Returns one row of atmospheric data.
    """

    # Create run folder
    run_folder = f"GRAM_SAMPLES/run_{alt_km}km"
    os.makedirs(run_folder + "/GRAM", exist_ok=True)

    # Update TITAN options
    options.output_folder = run_folder

    # --------------------------
    # Create dummy TITAN assembly
    # --------------------------

    class DummyAssembly:
        id = 0

        class trajectory:
            altitude = alt_km * 1000.0  # meters
            latitude = lat_deg * 3.14159 / 180.0
            longitude = lon_deg * 3.14159 / 180.0

    assembly = DummyAssembly()

    # --------------------------
    # Run MarsGRAM once
    # --------------------------

    gram.run_single_gram(assembly, options)

    # --------------------------
    # Load the produced OUTPUT.csv
    # --------------------------

    df = pd.read_csv(f"{run_folder}/GRAM/OUTPUT.csv")

    return df.iloc[0]  # return first (and only) row


# -------------------------------------------------------
# EXAMPLE: get atmosphere at 0, 10, 20, 30 km
# -------------------------------------------------------

if __name__ == "__main__":
    for alt in [0, 10, 20, 30]:
        row = sample_mars_atmosphere(alt)
        print(f"\nMars atmosphere at {alt} km:")
        print(f"  Temperature: {row['Temperature_K']} K")
        print(f"  Pressure:    {row['Pressure_Pa']} Pa")
        print(f"  Density:     {row['Density_kgm3']} kg/m³")
