import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load TITAN MarsGRAM atmosphere output
# -------------------------------------------------
df = pd.read_csv("Sphere/GRAM/OUTPUT.csv")

alt = df["Height_km"]
rho = df["Density_kgm3"]
T   = df["Temperature_K"]
P   = df["Pressure_Pa"]


# -------------------------------------------------
# Density vs Altitude
# -------------------------------------------------
plt.figure()
plt.plot(rho, alt)
plt.xlabel("Density (kg/m³)")
plt.ylabel("Altitude (km)")
plt.title("Mars Atmosphere: Density vs Altitude")
plt.grid(True)
plt.savefig("mars_density_vs_altitude.png", dpi=300)


# -------------------------------------------------
# Temperature vs Altitude
# -------------------------------------------------
plt.figure()
plt.plot(T, alt)
plt.xlabel("Temperature (K)")
plt.ylabel("Altitude (km)")
plt.title("Mars Atmosphere: Temperature vs Altitude")
plt.grid(True)
plt.savefig("mars_temperature_vs_altitude.png", dpi=300)


# -------------------------------------------------
# Pressure vs Altitude (log scale is standard)
# -------------------------------------------------
plt.figure()
plt.semilogx(P, alt)
plt.xlabel("Pressure (Pa)")
plt.ylabel("Altitude (km)")
plt.title("Mars Atmosphere: Pressure vs Altitude (log scale)")
plt.grid(True)
plt.savefig("mars_pressure_vs_altitude.png", dpi=300)

print("Plots generated!")
