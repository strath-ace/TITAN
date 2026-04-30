import numpy as np

class FilterPoints():
    def __init__(self, stateModel, measurementModel, parallel=True):
        self.transformer = stateModel
        self.sensor = measurementModel
        self.parallel = False

        self.points = self.transformer.generate_points()
    
    def predict(self, new_points):
        self.transformer.transform(new_points, overwrite=True)
        self.points = self.transformer.generate_points()

    def update(self, measurements, independent_variable):
        observations = self.sensor.observe(independent_variable)
        if observations is None: return
        mus, covs = self.transformer.transform(measurements, overwrite=False)
        
        