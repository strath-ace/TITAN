import numpy as np

class PropellantTank:
    def __init__(
        self,
        propellant_type: str,
        capacity: float,
        initial_amount: float = None,
        residual: float = 0.0,
        position_B=(0.0, 0.0, 0.0),
        radius: float = 0.0,
        dry_mass: float = 0.0,
    ):
        self.propellant_type = propellant_type
        self.capacity = float(capacity)
        self.residual = float(residual)

        self.position_B = np.asarray(position_B, dtype=float).reshape(3)  # <- enforce 3
        self.radius = float(radius)
        self.dry_mass = float(dry_mass)

        self.prop_mass = 0.0
        if initial_amount is None:
            self.fill(self.capacity)
        else:
            self.prop_mass = min(float(initial_amount), self.capacity)

        self.total_consumed = 0.0

    def fill(self, amount: float):
        amount = float(amount)
        if amount <= 0:
            return 0.0

        added = min(amount, self.capacity - self.prop_mass)
        self.prop_mass += added
        return added

    def consume(self, amount: float) -> float:
        """
        Consume propellant safely.
        Returns the actual mass consumed (important if tank runs dry).
        """
        amount = float(amount)
        if amount <= 0:
            return 0.0

        available = max(0.0, self.prop_mass - self.residual)
        used = min(amount, available)

        self.prop_mass -= used
        self.total_consumed += used
        return used

    def consume_mdot(self, mdot: float, dt: float) -> float:
        """
        Convenience method for propulsion models.
        mdot : mass flow rate [kg/s]
        dt   : timestep [s]
        """
        return self.consume(mdot * dt)

    def get_remaining_propellant(self) -> float:
        return self.prop_mass

    def is_empty(self) -> bool:
        return self.prop_mass <= self.residual

    def fill_fraction(self) -> float:
        return self.prop_mass / self.capacity if self.capacity > 0 else 0.0
    
    @property
    def total_mass(self) -> float:
        """Dry mass + remaining propellant."""
        return self.dry_mass + self.prop_mass