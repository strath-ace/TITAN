import os
import shutil
import subprocess
import configparser
import csv

# -------------------------------
# Validation sweep parameters
# -------------------------------
altitudes_km = [0, 5, 10, 15, 20, 25, 30, 35, 40]   # 0–40 km
latitudes_deg = [-60, -30, 0, 30, 60]              # coverage similar to paper

template_file = "Config_template.cfg"
results_file = "marsgram_validation_results.csv"

# Clean previous results file
if os.path.exists(results_file):
    os.remove(results_file)

# Write CSV header
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["altitude_km", "latitude_deg", "density", "pressure", "temperature"])

# Load template (ALLOW duplicate keys)
config = configparser.ConfigParser(strict=False)
config.optionxform = str  # preserve case
config.read(template_file)

# ---------------------------------------------
# Loop over altitude/latitude combinations
# ---------------------------------------------
for alt in altitudes_km:
    for lat in latitudes_deg:

        print(f"\nRunning: altitude={alt} km, latitude={lat} deg")

        # Make a working config file
        cfg_name = f"cfg_alt{alt}_lat{lat}.cfg"
        shutil.copy(template_file, cfg_name)

        # Reload config (ALLOW duplicates)
        run_cfg = configparser.ConfigParser(strict=False)
        run_cfg.optionxform = str
        run_cfg.read(cfg_name)

        # ------------------------
        # MODIFY TRAJECTORY SETUP
        # ------------------------
        run_cfg["Trajectory"]["Altitude"] = str(alt * 1000.0)   # meters
        run_cfg["Trajectory"]["Latitude"] = str(lat)            # degrees

        # ------------------------
        # SWITCH TO MARS + GRAM
        # ------------------------
        run_cfg["Model"]["Planet"] = "mars"
        run_cfg["Freestream"]["method"] = "Standard"
        run_cfg["Freestream"]["model"] = "GRAM"

        # ---------------------------------------------
        # DISABLE GEOMETRY (CRITICAL FOR VALIDATION)
        # ---------------------------------------------
        # Provide a harmless path
        run_cfg["Assembly"]["Path"] = "./"
        # Tell TITAN explicitly: "no connectivity"
        run_cfg["Assembly"]["Connectivity"] = "none"

        # Remove all objects (prevents STL loading)
        if run_cfg.has_section("Objects"):
            run_cfg.remove_section("Objects")
        run_cfg.add_section("Objects")

        # Save modified config
        with open(cfg_name, "w") as f:
            run_cfg.write(f)

        # --------------------------------
        # RUN TITAN WITH THIS CONFIG
        # --------------------------------
        subprocess.run(["python", "TITAN.py", "-c", cfg_name])

        # --------------------------------
        # READ GRAM OUTPUT
        # --------------------------------
        outfolder = run_cfg["Options"]["Output_folder"] + "/GRAM/"
        gram_file = os.path.join(outfolder, "OUTPUT.csv")

        if not os.path.isfile(gram_file):
            print("WARNING: OUTPUT.csv not found, skipping.")
            continue

        # Read a single-line GRAM output
        with open(gram_file, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

            dens = row.get("Density_kgm3", "")
            pres = row.get("Pressure_Pa", "")
            temp = row.get("Temperature_K", "")

        # Append to summary CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([alt, lat, dens, pres, temp])

print("\nAll validation runs completed.")
print(f"Results saved to: {results_file}")
