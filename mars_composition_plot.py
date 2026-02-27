import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/mnt/c/Users/Hussain Ahmad/Documents/MAE_Y5/ESA_Project/Validation/excel_data/data_tommy.csv")

plt.figure(figsize=(8,6))

plt.stackplot(
    data["Altitude_km"],
    data["CO2_mass_pct"]/100,
    data["N2_mass_pct"]/100,
    data["Ar_mass_pct"]/100,
    data["O2_mass_pct"]/100,
    labels=["CO2", "N2", "Ar", "O2"]
)

plt.xlabel("Altitude (km)")
plt.ylabel("Mass Fraction")
plt.title("Mars Atmospheric Composition (TITAN Output)")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.savefig("mars_composition_stacked.png", dpi=300)
plt.show()
