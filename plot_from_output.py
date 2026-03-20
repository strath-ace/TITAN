import pandas as pd
import matplotlib.pyplot as plt

# Path to your OUTPUT.csv — update if needed
csv_path = "Sphere/GRAM/OUTPUT.csv"

# Load data
df = pd.read_csv(csv_path)

# Extract columns
alt = df["Height_km"]
dens = df["Density_kgm3"]
pres = df["Pressure_Pa"]
temp = df["Temperature_K"]

# ---- Plot Density ----
plt.figure()
plt.plot(dens, alt)
plt.xlabel("Density (kg/m³)")
plt.ylabel("Altitude (km)")
plt.title("Mars Density vs Altitude")
plt.grid(True)
plt.savefig("mars_density_profile.png")

# ---- Plot Pressure ----
plt.figure()
plt.plot(pres, alt)
plt.xlabel("Pressure (Pa)")
plt.ylabel("Altitude (km)")
plt.title("Mars Pressure vs Altitude")
plt.grid(True)
plt.savefig("mars_pressure_profile.png")

# ---- Plot Temperature ----
plt.figure()
plt.plot(temp, alt)
plt.xlabel("Temperature (K)")
plt.ylabel("Altitude (km)")
plt.title("Mars Temperature vs Altitude")
plt.grid(True)
plt.savefig("mars_temperature_profile.png")

print("Plots saved as:")
print("  mars_density_profile.png")
print("  mars_pressure_profile.png")
print("  mars_temperature_profile.png")
