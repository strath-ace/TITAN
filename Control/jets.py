import numpy as np

def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    return v * 0.0 if n < eps else v / n

def _clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

class Jet:
    def __init__(self, name, position_B, direction_B, thrust_max_N=0.0, isp=None, group=None):
        self.name = str(name)
        self.position_B = np.asarray(position_B, dtype=float).reshape(3)
        self.direction_B = _unit(direction_B)
        self.thrust_max_N = float(thrust_max_N)   # max thrust [N]
        # self.thrust_min_N = 0.0
        self.thrust_N = 0.0                       # applied thrust [N], ALWAYS used in sums
        self.isp = isp
        self.group = group

    def set_thrust(self, thrust_N: float):
        # clamp
        t = float(thrust_N)
        if t < 0.0:
            t = 0.0
        if t > self.thrust_max_N:
            t = self.thrust_max_N

        # write both fields so force_B() works even if you never call step()
        self.thrust_cmd_N = t
        self.thrust_N = t


    def force_B(self):
        return self.thrust_N * self.direction_B

    def moment_B(self, cog_B):
        r = self.position_B - np.asarray(cog_B, dtype=float).reshape(3)
        return np.cross(r, self.force_B())

class JetSystem:
    def __init__(self, jets, tank=None):
        # store by lowercase key
        self.jets = {j.name.strip().lower(): j for j in jets}
        self.tank = tank
        self.groups = {}
        self.prop_used_kg_total = 0.0

    def add_group(self, group, jet_names):
        g = str(group).strip().lower()
        self.groups[g] = [str(n).strip().lower() for n in jet_names]

    def set_thrust(self, name, thrust_N):
        key = str(name).strip().lower()
        j = self.jets.get(key)
        if j is None:
            return False
        j.set_thrust(thrust_N)
        return True

    def set_group_thrust(self, group, thrust_N):
        g = str(group).strip().lower()
        any_set = False
        for n in self.groups.get(g, []):
            any_set = self.set_thrust(n, thrust_N) or any_set
        return any_set

    def step(self, dt):
        # No slew, no prop bookkeeping (yet). Keep for compatibility.
        return

    def net_force_moment_B(self, cog_B):
        F = np.zeros(3, dtype=float)
        M = np.zeros(3, dtype=float)
        for j in self.jets.values():
            F += j.force_B()
            M += j.moment_B(cog_B)
        return F, M
