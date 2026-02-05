import numpy as np
import pandas as pd
from Control.jets import JetSystem

class ControlSystem:
    def __init__(self, command_file: str, mode: str = "time"):
        self.command_file = command_file
        self.mode = mode  # "time" or "iter"

        df = pd.read_csv(command_file)

        # normalize column names
        df.columns = [c.strip() for c in df.columns]

        # force numeric time/iter
        if "time_s" in df.columns:
            df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        if "iter" in df.columns:
            df["iter"] = pd.to_numeric(df["iter"], errors="coerce")

        # normalize strings
        for c in ("type", "name", "field", "units"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # drop rows with bad independent var
        key = "time_s" if mode == "time" else "iter"
        if key in df.columns:
            df = df.dropna(subset=[key]).sort_values(key).reset_index(drop=True)

        self._df = df

        # optional: cache to avoid redundant sets every step
        self._last_applied = {}  # (type,name,field) -> value_in_radians

    def step(self, titan, options):
        x = titan.time if self.mode == "time" else titan.iter
        key = "time_s" if self.mode == "time" else "iter"

        if key not in self._df.columns:
            return

        df_up_to_now = self._df[self._df[key] <= x]
        if df_up_to_now.empty:
            return

        latest = (
            df_up_to_now
            .sort_values(key)
            .drop_duplicates(subset=["type", "name", "field"], keep="last")
        )

        any_changed = False

        for _, r in latest.iterrows():
            typ = str(r.get("type", "")).strip().lower()
            field = str(r.get("field", "")).strip().lower()
            target = str(r.get("name", "")).strip()
            val = float(r.get("value", 0.0))
            units = str(r.get("units", "")).strip().lower()

            # -----------------------------
            # Control surfaces (existing)
            # -----------------------------
            if typ in ("control_surface", "controlsurface"):
                if field != "deflection":
                    continue

                cmd = np.radians(val) if units in ("deg", "degree", "degrees") else val

                for ass in titan.assembly:
                    for obj in ass.objects:
                        if not hasattr(obj, "set_deflection"):
                            continue
                        if self._name_matches(target, obj.name):
                            k = (typ, target.lower(), field)
                            if k in self._last_applied and np.isclose(self._last_applied[k], cmd):
                                continue
                            obj.set_deflection(cmd)
                            self._last_applied[k] = cmd
                            any_changed = True

                continue  # done with this row

            # -----------------------------
            # Jets (NEW)
            # -----------------------------
            if typ in ("jet", "thruster"):
                if field != "thrust":
                    continue

                for ass in titan.assembly:
                    if not hasattr(ass, "jet_system") or ass.jet_system is None:
                        continue

                    # if pct, you need the jet object, so resolve first
                    if units in ("pct", "percent", "%"):
                        j = ass.jet_system.jets.get(target.strip().lower())
                        if j is None:
                            print(f"[CONTROL][JET] WARNING: jet '{target}' not found on ass={ass.id}")
                            continue
                        thrust_N = (val / 100.0) * float(j.thrust_max_N)
                    else:
                        thrust_N = val

                    k = (typ, target.lower(), field, ass.id)
                    if k in self._last_applied and np.isclose(self._last_applied[k], thrust_N):
                        continue

                    ok = ass.jet_system.set_thrust(target, thrust_N)
                    if not ok:
                        print(f"[CONTROL][JET] WARNING: jet '{target}' not found on ass={ass.id}")
                        continue

                    self._last_applied[k] = thrust_N
                    any_changed = True


                    print(f"[CONTROL][JET] t={titan.time:.3f} ass={ass.id} target='{target}' thrust_N={thrust_N} units='{units}'")


                continue

        if any_changed:
            print(f"[CONTROL] t={titan.time:.6f}s applied latest commands")


    @staticmethod
    def _name_matches(target_name: str, obj_name: str) -> bool:
        # target like "Flap_T"; obj_name like ".../flap_t.stl"
        t = target_name.strip().lower()
        base = obj_name.split("/")[-1].split("\\")[-1]
        stem = base.split(".")[0].lower()
        return (t == stem) or (t == base.lower())