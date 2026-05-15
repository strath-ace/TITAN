"""
Build GRAM altitude profile (Mars/Venus/Titan) by running GRAM at each altitude.
Output CSV is used by load_atmosphere("GRAM") for bridging.
Run from TITAN root: python Freestream/build_gram_profile.py Freestream/gram_profile_config.cfg
"""
import os
import sys
import configparser
import numpy as np
import pandas as pd

PLANET_SPECIES = {"mars": ["CO2", "N2", "Ar", "O2", "CO"], "venus": ["CO2", "N2"], "titan": ["N2", "CH4"]}

# Number density output in 1/cm³
AVO = 6.0221408e23
MOLAR_MASS_KG = {"CO2": 44.0095e-3, "N2": 28.0134e-3, "Ar": 39.948e-3, "O2": 31.9988e-3, "CO": 28.010e-3, "CH4": 16.043e-3}


def get_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    s = cfg["GRAM_Profile"]
    return {
        "planet": s.get("planet", "mars").strip().lower(),
        "altitude_min_km": float(s.get("altitude_min_km", 0)),
        "altitude_max_km": float(s.get("altitude_max_km", 300)),
        "altitude_step_km": float(s.get("altitude_step_km", 5)),
        "gram_root": s.get("gram_root", "").strip(),
        "gram_exe_dir": s.get("gram_exe_dir", "").strip(),
        "spice_path": s.get("spice_path", "").strip(),
        "year": int(s.get("year", 2021)),
        "month": int(s.get("month", 2)),
        "day": int(s.get("day", 18)),
        "hour": int(s.get("hour", 12)),
        "minute": int(s.get("minute", 0)),
        "seconds": float(s.get("seconds", 0)),
        "latitude_deg": float(s.get("latitude_deg", 0)),
        "longitude_deg": float(s.get("longitude_deg", 0)),
        "output_csv": s.get("output_csv", "Freestream/Models/Mars_GRAM_profile.csv").strip(),
        "work_dir": s.get("work_dir", "Freestream/temp_gram_profile").strip(),
    }


def write_namelist(cfg, alt_km, work_dir):
    p = cfg["planet"]
    data_path = os.path.join(cfg["gram_root"], p.capitalize(), "data")
    lines = [
        " $INPUT",
        "  SpicePath      = '{}'".format(cfg["spice_path"]),
        "  DataPath       = '{}'".format(data_path),
        "  ListFileName   = '{}'".format(os.path.join(work_dir, "LIST")),
        "  ColumnFileName = '{}'".format(os.path.join(work_dir, "OUTPUT")),
        "  Month = '{}'".format(cfg["month"]),
        "  Day = '{}'".format(cfg["day"]),
        "  Year = '{}'".format(cfg["year"]),
        "  Hour = '{}'".format(cfg["hour"]),
        "  Minute = '{}'".format(cfg["minute"]),
        "  Seconds = '{}'".format(cfg["seconds"]),
        "NumberOfPositions     = 1",
        "EastLongitudePositive = 1",
        "InitialHeight         = {}".format(alt_km),
        "InitialLatitude       = {}".format(cfg["latitude_deg"]),
        "InitialLongitude      = {}".format(cfg["longitude_deg"]),
    ]
    if p in ("neptune", "uranus"):
        lines.append("MinMaxFactor = 0.0")
        lines.append("ComputeMinMaxFactor = 0")
    lines.append(" $END")
    path = os.path.join(work_dir, "gram_config_1")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run_gram(cfg, config_path):
    exe = os.path.join(cfg["gram_exe_dir"], cfg["planet"].capitalize() + "GRAM")
    if not os.path.isfile(exe):
        raise FileNotFoundError("GRAM executable not found: {}".format(exe))
    ret = os.system("echo {} | {}".format(config_path, exe))
    if ret != 0:
        raise RuntimeError("GRAM failed (code {}) at {}".format(ret, config_path))


def row_from_output(csv_path, alt_m, species_index):
    df = pd.read_csv(csv_path)
    T = df["Temperature_K"].iloc[0]
    rho = df["Density_kgm3"].iloc[0]
    fracs = np.array([df[sp + "mass_pct"].iloc[0] / 100.0 for sp in species_index])
    fracs /= fracs.sum()
    # Species mass density (kg/m³) -> number density (1/cm³), same units as NRLMSISE00.csv
    rho_species = rho * fracs
    n_per_cm3 = [rho_species[j] * AVO / MOLAR_MASS_KG[sp] / 1e6 for j, sp in enumerate(species_index)]
    return [alt_m, T] + n_per_cm3


def main():
    if len(sys.argv) < 2:
        print("Usage: python Freestream/build_gram_profile.py Freestream/gram_profile_config.cfg")
        sys.exit(1)
    cfg = get_config(sys.argv[1])
    planet = cfg["planet"]
    if planet not in PLANET_SPECIES:
        print("Unsupported planet: {}. Use one of {}.".format(planet, list(PLANET_SPECIES.keys())))
        sys.exit(1)

    species_index = PLANET_SPECIES[planet]
    work_dir = cfg["work_dir"]
    os.makedirs(work_dir, exist_ok=True)

    alts_km = np.arange(cfg["altitude_min_km"], cfg["altitude_max_km"] + 0.5 * cfg["altitude_step_km"], cfg["altitude_step_km"])
    n = len(alts_km)
    print("Building {} profile: {}–{} km, step {} km ({} points).".format(
          planet, alts_km[0], alts_km[-1], cfg["altitude_step_km"], n))

    rows = []
    for i, alt_km in enumerate(alts_km):
        write_namelist(cfg, alt_km, work_dir)
        run_gram(cfg, os.path.join(work_dir, "gram_config_1"))
        csv_path = os.path.join(work_dir, "OUTPUT.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("GRAM did not produce OUTPUT.csv at {} km".format(alt_km))
        rows.append(row_from_output(csv_path, alt_km * 1000.0, species_index))
        if (i + 1) % 20 == 0 or i == 0:
            r = rows[-1]
            print("  {} km -> T={:.1f} K, n_tot={:.2e} 1/cm³".format(alt_km, r[1], sum(r[2:])))

    cols = ["Altitude_m", "Temperature_K"] + species_index
    df = pd.DataFrame(rows, columns=cols)
    out_path = cfg["output_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Wrote {} ({} rows). Use in load_atmosphere('GRAM') for {}.".format(out_path, len(df), planet))


if __name__ == "__main__":
    main()
