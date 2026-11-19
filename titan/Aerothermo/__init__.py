"""
Aerothermo - aerothermodynamic analysis and ray tracing

This module includes functionality to calculate the loads induced upon an assembly object by the aerothermodynamic environment.

Impinging hypersonic flow pushes (the \"aero\" part) and heats an assembly (the \"thermo\" part). 

In order for us to compute the pressure, shear and heat flux on an object we need to deploy an aerothermodynamic model. 
In TITAN these models are split into high and low fidelity methods. 

For high fidelity aerothermo modelling refer to the CFD SU2 and DSMC SPARTA sections. These modules interface with high fidelity tools.

Low fidelity aerothermo modelling is handled in the aerothermo module.
"""
