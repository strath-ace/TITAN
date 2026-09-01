# TITAN

|   |   |
| :---: | :--- |
|<img src="https://github.com/strath-ace/TITAN/blob/main/TITAN-logo.png" width="90" height="100"> | TransatmospherIc flighT simulAtioN <br /> A python code for multi-fidelity and multi-physics simulations of access-to-space and re-entry|

# Usage

## Installation

Start by cloning this repository

```console
git clone https://github.com/strath-ACE/TITAN -b rc_v02_atlas
cd TITAN
```

To install TITAN, it is required to use an Anaconda environment. The required libraries are listed in the titan_env.yml file.
In order to install the required packages, the Anaconda environment can be created using

```console
conda env create --name titan --file titan_env.yml
```
If using an ARM architecture (i.e. Apple silicon) you will likely need to specify an alternate platform, e.g.

```console
conda env create --name titan --platform=osx-64 --file titan_env.yml
```
***NOTE:*** TITAN is developed for **linux** first and foremost, you should **not expect perfect behaviour** on other operating systems. 

TITAN can then be installed as a package in your environment.

```console
conda activate titan
pip install .
```

### Optional
Submodule installation is required if you wish to use the extended functionality of TITAN. First clone the git submodules

```console
git submodule update --init --recursive
cd ./Executables
```
#### Mutation++
The mutation++ package is an optional method to compute the freestream conditions. The library can be compiled and installed as follows

```console
cd ./mutationpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install
cd ..
pip install .
```
#### AMGio
AMGio is a library that is required to perform mesh I/O for adaptation when running high-fidelity simulations. The installation can be particular depending on your gcc version. The following has been verified for gcc v14 and v15.

```console
CFLAGS="-Wno-return-mismatch -Wno-implicit-function-declaration" pip install --config-settings editable_mode=compat -e amgio/su2gmf/
ln -s ./amgio/su2gmf/su2_to_gmf.py ./su2_to_gmf.py
ln -s ./amgio/su2gmf/gmf_to_su2.py ./gmf_to_su2.py
```

### GRAM model
TITAN has the capability to use the NASA-GRAM <https://software.nasa.gov/software/MFS-33888-1> to retrieve the atmospheric properties of Earth, Neptune and Uranus. The user needs to request NASA to use the atmospheric model.

Once the GRAM tool is compiled, the user needs to link the binaries, and place them in the Executables folder

### PATO
TITAN can use PATO for thermal response calculations. In order to do so one must create another conda environment named "pato" (case sensitive), note if the name is different TITAN will not be able to run PATO.

```console
conda deactivate
conda config --add channels conda-forge
conda config --add channels pato.devel
conda config --set channel_priority strict
conda create -y --name pato -c conda-forge -c pato.devel pato
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
(.venv) $ python -m titan -c config.cfg
```

The solution is stored in the specifed output folder.

After obtaining the solution of the simulation, the data can be postprocessed by introducing a new flag to the instruction, refering to the Postprocess method that can be **WIND** or **ECEF**. The following command does not run a new simulation, but it postprocess the already obtained solutions in the **Output_folder** specified field.

```console
(.venv) $ python -m titan -c config.cfg -pp WIND
```

## Geometry modelling

The frame convention in the geometry modelling are such that the X axis is the longitudinal axis pointing ahead, Z axis is the vertical axis pointing downwards, and the Y axis is the lateral one, pointing in such a way that the frame is right-handed. 

In case of multiple components, if the components are in contact with each other, the respective meshes need to be identical in the interface (i.e. same node positioning and same facets).
