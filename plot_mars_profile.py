import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mars_profile_results.csv")

alt = df["altitude_km"]
temp = df["temperature_K"]
pres = df["pressure_Pa"]
rho  = df["density_kgm3"]

# ---------------------------
# Density vs Altitude
# ---------------------------
plt.figure()
plt.plot(rho, alt)
plt.xlabel("Density (kg/m³)")
plt.ylabel("Altitude (km)")
plt.title("Mars Atmosphere: Density vs Altitude")
plt.gca().invert_yaxis()
plt.grid()
plt.savefig("mars_density_profile.png", dpi=300)

# ---------------------------
# Pressure vs Altitude
# ---------------------------
plt.figure()
plt.plot(pres, alt)
plt.xlabel("Pressure (Pa)")
plt.ylabel("Altitude (km)")
plt.title("Mars Atmosphere: Pressure vs Altitude")
plt.gca().invert_yaxis()
plt.grid()
plt.savefig("mars_pressure_profile.png", dpi=300)

# ---------------------------
# Temperature vs Altitude
# ---------------------------
plt.figure()
plt.plot(temp, alt)
plt.xlabel("Temperature (K)")
plt.ylabel("Altitude (km)")
plt.title("Mars Atmosphere: Temperature vs Altitude")
plt.gca().invert_yaxis()
plt.grid()
plt.savefig("mars_temperature_profile.png", dpi=300)

print("Plots generated!")
