from Configuration.options import Options
from Configuration.assembly import Assembly
import numpy as np
import pandas as pd
import os

# -----------------------------------
# Output folder
# -----------------------------------
folder = "MarsProfile"
os.makedirs(f"{folder}/GRAM", exist_ok=True)

# -----------------------------------
# Build TITAN Options
# -----------------------------------
options = Options()
options.planet.name = "mars"
options.atmosphere.model = "gram"
options.output_folder = folder

# -----------------------------------
# Sweep altitudes
# -----------------------------------
alts_km = np.arange(0, 60, 5)

results = []

for alt_km in alts_km:
    print(f"Running at altitude {alt_km} km")

    assembly = Assembly()
    assembly.trajectory.altitude = alt_km * 1000
    assembly.trajectory.latitude = 0
    assembly.trajectory.longitude = 0
    assembly.trajectory.velocity = 0
    
    from TITAN import main
    main(assembly=assembly, options=options)

    df = pd.read_csv(f"{folder}/GRAM/OUTPUT.csv")
    row = df.iloc[0]

    results.append([
        alt_km,
        row["Temperature_K"],
        row["Pressure_Pa"],
        row["Density_kgm3"]
    ])

df_out = pd.DataFrame(results, columns=["alt_km","T","P","rho"])
df_out.to_csv("mars_profile_results.csv", index=False)

print("DONE.")
