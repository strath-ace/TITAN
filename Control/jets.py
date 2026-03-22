from math import gamma
import numpy as np

def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    return v * 0.0 if n < eps else v / n

class Jet:
    def __init__(self, name, position_B, direction_B, thrust_max_N = 0.0, group = None, isp = None):
        self.name = str(name)
        self.position_B = np.asarray(position_B, dtype=float).reshape(3)
        self.direction_B = _unit(direction_B)
        self.thrust_max_N = float(thrust_max_N)   # max thrust [N]
        self.thrust_N = 0.0                       # applied thrust [N]
        self.isp = isp
        self.group = group
        self.g0 = 9.80665 # m/s^2

    def set_thrust(self, thrust_N: float):
        # clamp
        t = float(thrust_N)
        if t < 0.0:
            t = 0.0
        if t > self.thrust_max_N:
            t = self.thrust_max_N

        self.thrust_cmd_N = t
        self.thrust_N = t

    def force_B(self):
        return self.thrust_N * self.direction_B

    def moment_B(self, cog_B):
        r = self.position_B - np.asarray(cog_B, dtype=float).reshape(3)
        return np.cross(r, self.force_B())

class JetSystem:
    def __init__(self, jets, tank=None):
        self.jets = {j.name.strip().lower(): j for j in jets}
        self.tank = tank
        self.groups = {}          # group -> [jet_name_lower]
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
        if self.tank is None or dt <= 0.0:
            return

        total_mdot = 0.0
        for j in self.jets.values():
            if j.isp is None or j.isp <= 0.0 or j.thrust_N <= 0.0:
                continue
            total_mdot += j.thrust_N / (float(j.isp) * 9.80665)

        prop_needed = total_mdot * float(dt)
        used = self.tank.consume(prop_needed)

        self.prop_used_kg_total += used

        # If we couldn't supply full demand -> tank at residual -> cut jets
        if used + 1e-12 < prop_needed:
            for j in self.jets.values():
                j.set_thrust(0.0)

    def net_force_moment_B(self, cog_B):
        F = np.zeros(3, dtype=float)
        M = np.zeros(3, dtype=float)
        for j in self.jets.values():
            F += j.force_B()
            M += j.moment_B(cog_B)
        return F, M

    def add_group(self, group, jet_names):
        g = str(group).strip().lower()
        self.groups[g] = [str(n).strip().lower() for n in jet_names]

    def has_group(self, group: str) -> bool:
        return str(group).strip().lower() in self.groups

    def set_group_thrust(self, group, thrust_N, *, per_jet: bool = True):
        """
        If per_jet=True: apply thrust_N to each jet in the group.
        If per_jet=False: split total thrust_N equally across jets in the group.
        """
        g = str(group).strip().lower()
        members = self.groups.get(g, [])
        if not members:
            return False

        if per_jet:
            for n in members:
                self.set_thrust(n, thrust_N)
        else:
            share = float(thrust_N) / float(len(members))
            for n in members:
                self.set_thrust(n, share)
        return True