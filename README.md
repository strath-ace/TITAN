# TITAN

|   |   |
| :---: | :--- |
|<img src="https://github.com/strath-ace/TITAN/blob/main/TITAN-logo.png" width="90" height="100"> | TransatmospherIc flighT simulAtioN <br /> A python code for multi-fidelity and multi-physics simulations of access-to-space and re-entry|

# Usage

## Installation

To install TITAN, it is required to use an Anaconda environment. The required libraries are listed in the requirements.txt file.
In order to install the required packages, the Anaconda environment can be created using

```console
$ conda create --name myenv --file requirements.txt
```

If the packages are not found, the user can append a conda channel to retrieve the packages, by running

```console
$ conda config --append channels conda-forge
```
To activate the Conda environment:

```console
$ conda activate myenv
```

After activation, the user needs to install other packages that were not possible (as GMSH) using the conda requirements. To do so,
the user can use the pip manager to install the packages listed in the pip_requirements.txt

```console
(.venv) $ pip install -r pip_requirements.txt
```

To install pymap3D, you can clone the following github page <https://github.com/geospace-code/pymap3d/> into the Executables foldar and install using

```console
(.venv) $ pip install -e pymap3d
```

### Optional

#### Mutation++
The mutation++ package is an optional method to compute the freestream conditions. It can be installed by following the instructions in <https://github.com/mutationpp/Mutationpp>
Once the mutation++ has been compiled, you can install by:

1. In the github link, go to the thirdpary folder and clone the Pybind repository into your thirdparty folder in mutationpp

2. In the Mutationpp root folder, run
```console
(.venv) $ python setup.py build
(.venv) $ python setup.py install
```

#### AMGio
AMGio is a library that is required to perform mesh adaptation when running high-fidelity simulations. To install the AMGio library, one must clone the following github page to the TITAN/Executables folder: <https://github.com/bmunguia/amgio>. THe user can the proceed to the installation using

```console
(.venv) $ pip install -e amgio/su2gmf/
```

### GRAM model
TITAN has the capability to use the NASA-GRAM <https://software.nasa.gov/software/MFS-33888-1> to retrieve the atmospheric properties of Earth, Neptune and Uranus. The user needs to request NASA to use the atmospheric model.

Once the GRAM tool is compiled, the user needs to link the binaries, and place them in the Executables folder

### Troubleshooting

If mpirun is not working, the user may require to reinstall openmpi and/or mpi4py using pip, by following the steps:

```console
(.venv) $ mamba uninstall mpi4py
(.venv) $ mamba uninstall openmpi
(.venv) $ pip install openmpi
(.venv) $ pip install mpi4py
```

## Setting up the Configuration file 

An explanation of the Configuration file can be found in the Config_temmplate.cfg file, in the root folder.

TITAN will read the configuration file using the config parser package. The file is divided into several subsections:

### Options
* **Num_iters** - Maximum number of iterations
* **Load_State** - Load the last simulation state
* **Fidelity** - Select the level of the aerothermodynamics in the simulation (Low/High/Multi)
* **Output_folder** - Folder where the simulation solution is stored
* **Load_mesh** - Flag to indicate if the mesh should be loaded (if already pre-processed in previous simulation)
* **Load_state** - Flag to resume the simulation (overrules the flag Load_mesh)

### Trajectory
* **Altitude** - Initial altitude [meters]
* **Velocity** - Initial Velocity [meters/second]
* **Flight_path_angle** - Initial FLight Path Angle [degree]
* **Heading_angle** - Initial Heading Angle [degree]
* **Latitude** - Initial Latitude [degree]
* **Longitude** - Initial Longitude [degree]

### Model
* **Planet** - Name of the planel (Earth, Neptune, Uranus)
* **Vehicle** - Flag for use of custom vehicle parameters (Mass, Nose radius, Area of reference)
* **Drag** - Flag for use of drag model (if Vehicle = True)

### Vehicle
* **Mass** - Mass of the vehicle [kg]
* **Nose_radius** - Nose radius of the vehicle [meters]
* **Area_reference** - Area of reference for coefficient computation  [meters^2]
* **Drag_file** - Name of the Drag model containing the Mach vs drag coefficient information in TITAN/Model/Drag

### Freestream
* **method** - Method used for the computation of the freestream (Standard, Mutationpp, GRAM)
* **model** - Atmospheric model (Earth - NRLMSISE00,GRAM ; Neptune - GRAM; Uranus - GRAM)

### GRAM
* **MinMaxFactor** - Value of the MinMaxFactor for the NeptuneGRAM
* **ComputeMinMaxFactor** - Automatic computation of the MinMaxFactor for the NeptuneGRAM (see NeptuneGRAM manual. 0 = False, 1 = True)
* **SPICE_Path** - Path for the SPICE database
* **GRAM_Path** - Path for GRAM software (required for Earth GRAM)

### Time
* **Time_step** - Value of the time step [second]

### SU2
* **Solver** - Solver to be used in CFD simulation (EULER/NAVIER_STOKES or NEMO_EULER/NEMO_NAVIER_STOKES)
* **Num_iters** - Number of CFD iterations
* **Conv_method** - Convective scheme (Default = AUSM)
* **Adapt_iter** - Number of mesh adaptations
* **Num_cores** - Number of cores to run CFD simulation
* **Muscl** - Flag for MUSCL reconstruction (Yes/No)
* **Cfl** - CFL number

### Bloom
* **Flag** - Flag to activate Bloom (True/False)
* **Layers** - Number of layers in the boundary layer
* **Spacing** - Spacing of the initial layer
* **Growth_Rate** - Growth rate between layers

### AMG
* **Flag** - Flag to activate AMG
* **P** - Norm of the error estimate for the Hessian computation
* **C** - Correction for metric complexity
* **Sensor** - Name of the computational field used to compute the metric tensor for mesh adaptation

### Assembly
* **Path** - Path for the geometry files
* **Connectivity** - Linkage information for the specified components in the Objects section
* **Angle_of_attack** - Angle of attack of the assembly [degree]
* **Sideslip** - Angle of sideslip of the assembly [degree]

### Objects
* **Primitive used in the Assembly** - name_Marker = (NAME, TYPE, MATERIAL)
* **Joints used in the Assembly** - name_Marker = (NAME, TYPE, MATERIAL, TRIGGER_TYPE, TRIGGER_VALUE)
	* NAME -> Name of the geometry file in stl format
	* TYPE -> Type of the object (Primitive/Joint)
	* MATERIAL -> Material of the object, needs to one specified in the material database
	* TRIGGER_TYPE  -> The criteria for the joint fragmentation (Altitude, time, iteration, Temperature)
	* TRIGGER_VALUE -> The value to trigger the fragmentation

## Running a simulation

TITAN is called in the conda environment using 

```console
(.venv) $ python TITAN.py -c config.cfg
```

The solution is stored in the specifed output folder. The structure in the output folder is as **SPECIFY HERE**

After obtaining the solution of the simulation, the data can be postprocessed by introducing a new flag to the instruction, refering to the Postprocess method that can be **WIND** or **ECEF**. The following command does not run a new simulation, but it postprocess the already obtained solutions in the **Output_folder** specified field.

```console
(.venv) $ python TITAN.py -c config.cfg -pp WIND
```

## Geometry modelling

The frame convention in the geometry modelling are such that the X axis is the longitudinal axis pointing ahead, Z axis is the vertical axis pointing downwards, and the Y axis is the lateral one, pointing in such a way that the frame is right-handed. 

In case of multiple components, if the components are in contact with each other, the respective meshes need to be identical in the interface (i.e. same node positioning and same facets).


# Patch Notes

\[2023-04-17\]

* Change name of variable facets_normal to facet_normal in mesh class
* Compute facet_normal to be proportional to the facet area
* Object of class aerothermo creates array based of number of facets instead of number of nodes
* Aerodynamic computation now takes facets normals as input
* Output files now have cell_data in addition to point_data
* Computation of facet radius in addition to nodes radius
* Aerothermodynamics computation now takes facet normals and facet radius as input
* Ablation is now working with facets
* SU2 solution is interpolated to the facets using a linear approach. Reverse function still needs to be written
* Test-cases have been adapted to accomodate the vertex->facet changes

\[2023-04-18\]

* Introduced a new interpolation, the reverse interpolation needs to be double checked
* Created function to map the surface facets to the correspondent tetra for thermal computation
* Created a thermal function to use the tetras mass to compute surface temperature
* Thermal function loops to check which tetras are ablated

\[2023-04-23\]

* Improved the speed and robustness of the tetra ablation
* Started commenting the functions
* Introduction of a new config term "Ablation_mode: (0D, Tetra)"