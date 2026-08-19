"""filtering module."""
import numpy as np

class FilterPoints():
    """FilterPoints."""
    def __init__(self, stateModel, measurementModel, parallel=True):
        """Documentation for the function.
:param stateModel: Value for statemodel.
:type stateModel: Any
:param measurementModel: Value for measurementmodel.
:type measurementModel: Any
:param parallel: Value for parallel.
:type parallel: Any"""
        self.transformer = stateModel
        self.sensor = measurementModel
        self.parallel = False

        self.points = self.transformer.generate_points()
    
    def predict(self, new_points):
        """Documentation for the function.
:param new_points: Value for new points.
:type new_points: Any"""
        self.transformer.transform(new_points, overwrite=True)
        self.points = self.transformer.generate_points()

    def update(self, measurements, independent_variable):
        """Documentation for the function.
:param measurements: Value for measurements.
:type measurements: Any
:param independent_variable: Value for independent variable.
:type independent_variable: Any"""
        observations = self.sensor.observe(independent_variable)
        if observations is None: return
        mus, covs = self.transformer.transform(measurements, overwrite=False)
        
        
