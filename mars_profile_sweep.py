import json
import subprocess
import os
import pandas as pd
import csv

# Altitude range (km)
altitudes = list(range(0, 51, 1))   # 0–50 km in steps of 1 km

# Base configuration
base_config = {
    "planet": "Mars",
    "atmosphere_model": "GRAM",
    "run_mode": "single_point",
    "latitude": 0.0,
    "longitude": 0.0,
    "velocity": 0.0
}

results = []

for alt in altitudes:
    print(f"Running GRAM at altitude {alt} km")

    # Create config for this run
    config = base_config.copy()
    config["altitude"] = alt * 1000.0  # meters

    # Write config.json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run TITAN
    subprocess.run(["python", "TITAN.py", "-f", "config.json"])

    # Read GRAM output
    gram_path = "./Sphere/GRAM/OUTPUT.csv"
    if not os.path.exists(gram_path):
        print("No OUTPUT.csv found — skipping.")
        continue

    df = pd.read_csv(gram_path)
    row = df.iloc[0]

    results.append({
        "alt_km": alt,
        "T_K": row["Temperature_K"],
        "P_Pa": row["Pressure_Pa"],
        "rho": row["Density_kgm3"]
    })

# Save the assembled vertical profile
profile_df = pd.DataFrame(results)
profile_df.to_csv("mars_profile.csv", index=False)

print("Done! Saved vertical profile to mars_profile.csv")
