## TITAN Uncertainty Propagation: Parameter Handbook
*24.11.2025* 
  
This handbook details all exposed variables that can be controlled using a *.yaml* file (see associated example in this folder).  Parametres are sorted by options section and given in the following format (note names are case sensitive)...  
*{**NAME** [data type] : Brief description}*

## Trajectory
 - **altitude** [metres]        : Initial height above body
 - **gamma** [radians]          : Initial flight path angle value.
 - **chi** [radians]            : Initial heading angle value.
 - **velocity** [metres/second] : Initial velocity value.
 - **latitude** [radians]       : Initial latitude value. 
 - **longitude** [radians]      : Initial longitude value.
 **OR**
 - **ECEF_x** [metres/second] : Initial x position in ECEF frame
 - **ECEF_y** [metres/second] : Initial y position in ECEF frame
 - **ECEF_z** [metres/second] : Initial z position in ECEF frame
 - **ECEF_u** [metres/second] : Initial u velocity in ECEF frame
 - **ECEF_v** [metres/second] : Initial v velocity in ECEF frame
 - **ECEF_w** [metres/second] : Initial w velocity in ECEF frame
 **OR**
 - **ECI_x** [metres/second]    : Initial x position in ECI frame
 - **ECI_y** [metres/second]    : Initial y position in ECI frame
 - **ECI_z** [metres/second]    : Initial z position in ECI frame
 - **ECI_u** [metres/second]    : Initial u velocity in ECI frame
 - **ECI_v** [metres/second]    : Initial v velocity in ECI frame
 - **ECI_w** [metres/second]    : Initial w velocity in ECI frame
 - **ECI_epoch_UNIX** [seconds] : ECI initial time (seconds since Jan 1 1970 00:00:00)

## Attitude
 - **omega_roll** [radians/second]  : Initial rotation rate about body roll (x) axis.
 - **omega_pitch** [radians/second] : Initial rotation rate about body pitch (y) axis.
 - **omega_yaw** [radians/second]   : Initial rotation rate about body yaw (z) axis.

 - **quat_x** [0-1] : Initial body-frame quaterion x component
 - **quat_y** [0-1] : Initial body-frame quaterion y component
 - **quat_z** [0-1] : Initial body-frame quaterion z component
 - **quat_w** [0-1] : Initial body-frame quaterion w component
  **OR**
 - **roll** [radians] : Initial rotation about wind roll (x) axis.
 - **aoa** [radians]  : Initial rotation about wind pitch (y) axis.
 - **slip** [radians] : Initial rotation about wind yaw (z) axis.

## Assembly
**Note if assembly fragments inertia tensor will be recalculated (these parameters will be discarded)**
- **Ixx** [kilogram/metre^2] : Moment of inertia about roll (x) axis
- **Iyy** [kilogram/metre^2] : Moment of inertia about pitch (y) axis
- **Izz** [kilogram/metre^2] : Moment of inertia about yaw (z) axis
- **Ixy** [kilogram/metre^2] : x-y product of inertia
- **Iyz** [kilogram/metre^2] : y-z product of inertia
- **Ixz** [kilogram/metre^2] : x-z product of inertia

## Aerothermo
- **catalycity** [0-1] : Surface catalytic efficiency/recombination, note "material" catalycity must be disabled

## Objects
Uncertain parameters assigned to objects should have the object name after the double underscore (e.g. 'trigger__Cube_A.stl')
- **trigger__** [float]      : The fragmentation trigger of the object, units determined by how the trigger is set
- **temperature__** [Kelvin] : The initial temperature of the object
**(Variables exposed by the in-development explosion model)**
- **energy__** [J] : Available energy of explosion

## Atmospheric
- **density_mult** [float] : Multiplier applied to freestream density retrieved from database