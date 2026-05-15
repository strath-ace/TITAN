import pandas as pd
import subprocess
import os
import shutil

# Altitudes (km)
altitudes_km = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# Validation results file
results_file = "mars_profile_results.csv"

# Remove old results
if os.path.exists(results_file):
    os.remove(results_file)

# Create results CSV
with open(results_file, "w") as f:
    f.write("altitude_km,temperature_K,pressure_Pa,density_kgm3\n")

template = "MarsProfile.txt"   # Your TITAN config template

for alt in altitudes_km:

    print(f"\nRunning TITAN at {alt} km")

    run_file = "MarsProfile_run.txt"

    # Copy template → run file
    shutil.copyfile(template, run_file)

    # Replace altitude line
    lines = []
    with open(run_file, "r") as f:
        for line in f:
            if line.strip().startswith("Altitude"):
                lines.append(f"Altitude = {alt*1000}\n")  
            else:
                lines.append(line)

    with open(run_file, "w") as f:
        f.writelines(lines)

    # Run TITAN
    subprocess.run(["python", "TITAN.py", "-c", run_file])

    # Load GRAM output
    output_path = "MarsProfile/GRAM/OUTPUT.csv"

    if not os.path.exists(output_path):
        print("⚠️  ERROR: No OUTPUT.csv — skipping altitude")
        continue

    df = pd.read_csv(output_path)
    row = df.iloc[0]

    # Extract key fields
    T = row["Temperature_K"]
    P = row["Pressure_Pa"]
    rho = row["Density_kgm3"]

    # Append to results
    with open(results_file, "a") as f:
        f.write(f"{alt},{T},{P},{rho}\n")

print("\n🌟 Completed altitude sweep!")
print("Results saved to mars_profile_results.csv")
