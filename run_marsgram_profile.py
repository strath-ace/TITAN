import os
import pandas as pd
import numpy as np
import subprocess

alts = list(range(0, 41, 2))   # 0–40 km in 2 km steps

folder = "MarsProfile"
os.makedirs(folder, exist_ok=True)

results = []

for h in alts:
    print(f"Running MarsGRAM at {h} km")

    # Build TITAN config file
    config = f"""
[Options]
Output_folder = {folder}

[Planet]
Name = mars

[Trajectory]
Altitude = {h * 1000}
Velocity = 5000
Flight_path_angle = -15
Latitude = 0
Longitude = 0

[Freestream]
Method = Standard
Model = GRAM
"""

    with open("mars_config.txt", "w") as f:
        f.write(config)

    subprocess.run(["python", "TITAN.py", "-c", "mars_config.txt"])

    # Read MarsGRAM output
    df = pd.read_csv(f"{folder}/GRAM/OUTPUT.csv")
    row = df.iloc[0]

    results.append({
        "alt_km": h,
        "temp_K": row["Temperature_K"],
        "rho": row["Density_kgm3"],
        "p": row["Pressure_Pa"]
    })

# Save profile
pd.DataFrame(results).to_csv("mars_profile.csv", index=False)

print("DONE. Run plot script now.")
