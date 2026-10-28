"""filtering module."""
import numpy as np

class FilterPoints():
    """Apply prediction and measurement updates to a point-based state model."""
    def __init__(self, stateModel, measurementModel, parallel=True):
        """Connect a state transformer and observation model.

:param stateModel: Model that propagates and transforms state points.
:type stateModel: Any
:param measurementModel: Model that predicts observations from an independent variable.
:type measurementModel: Any
:param parallel: Compatibility argument; point processing is currently serial.
:type parallel: bool
"""
        self.transformer = stateModel
        self.sensor = measurementModel
        self.parallel = False

        self.points = self.transformer.generate_points()
    
    def predict(self, new_points):
        """Propagate state points and regenerate the transformed point set.

    :param new_points: Input points to transform in place.
    :type new_points: numpy.ndarray
    """
        self.transformer.transform(new_points, overwrite=True)
        self.points = self.transformer.generate_points()

    def update(self, measurements, independent_variable):
        """Update the state-point distribution using available observations.

    :param measurements: Measurement values used to condition the state model.
:type measurements: Any
    :param independent_variable: Independent-variable value passed to the sensor model.
    :type independent_variable: Any
    """
        observations = self.sensor.observe(independent_variable)
        if observations is None: return
        mus, covs = self.transformer.transform(measurements, overwrite=False)
        
        
