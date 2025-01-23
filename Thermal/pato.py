#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN 
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import os
from vtk import *
import glob
import os, pathlib
import re
from scipy.spatial import KDTree
from Material import material as Material

conda_preamble = ['conda', 'run', '-n', 'pato'] # Better ideally to separate pato env from TITAN?

def compute_thermal(obj, time, iteration, options, hf, Tinf):

    """
    Compute the aerothermodynamic properties using the CFD software

    Parameters
    ----------
    assembly_list: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    """
    time_to_postprocess = setup_PATO_simulation(obj, time, iteration, options, hf, Tinf)

    run_PATO(options, obj.global_ID)

    postprocess_PATO_solution(options, obj, time_to_postprocess)

def setup_PATO_simulation(obj, time, iteration, options, hf, Tinf):
    """
    Sets up the PATO simulation - creates PATO simulation folders and required input files

    Parameters
    ----------
	?????????????????????????
    """

    write_PATO_BC(options, obj, time, hf, Tinf)
    time_to_postprocess = write_All_run(options, obj, time - options.dynamics.time_step, iteration)
    write_system_folder(options, obj.global_ID, time - options.dynamics.time_step)

    return time_to_postprocess

def write_material_properties(options, obj):

    #emissivity_coeffs = obj.material.material_emissivity_polynomial()
    emissivity_coeffs = Material.polynomial_fit(obj.material, obj.material_name, 'emissivity', 1)
    cp_coeffs = Material.polynomial_fit(obj.material, obj.material_name, 'specificHeatCapacity', 4)
    k_coeffs = Material.polynomial_fit(obj.material, obj.material_name, 'heatConductivity', 4)
    density = obj.material.density

    cp = obj.material.specificHeatCapacity(300+(obj.material.meltingTemperature-300)/2)
    em = obj.material.emissivity(300+(obj.material.meltingTemperature-300)/2)
    tc = obj.material.heatConductivity(300+(obj.material.meltingTemperature-300)/2)

    object_id = obj.global_ID

    with open(options.output_folder + '/PATO_'+str(object_id)+'/data/constantProperties', 'w') as f:

        f.write('/*---------------------------------------------------------------------------*\\n')
        f.write('Material properties for the substructure materials\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('\n')
        f.write('FoamFile {\n')
        f.write('  version     2.0;\n')
        f.write('  format      ascii;\n')
        f.write('  class       dictionary;\n')
        f.write('  location    "constant/subMati/FourierProperties";\n')
        f.write('  object      constantProperties;\n')
        f.write('}\n')
        f.write('// * * * * * *  Units * * * * * [kg m s K mol A cd] * * * * * * * * * * * * * //\n')
        f.write('// e.g. W: kg m^2 s^{-3}    [1 2 -3 0 0 0 0]\n')
        f.write('\n')
        f.write('/***        Temperature dependent material properties   ***/\n')
        f.write('/***        5 coefs - n0 + n1 T + n2 T² + n3 T³ + n4 T⁴ ***/\n')
        f.write('// specific heat capacity - cp - [0 2 -2 -1 0 0 0]\n')
        f.write('cp_sub_n[0] '+str(cp)+';\n')
        f.write('cp_sub_n[1] 0;\n')
        f.write('cp_sub_n[2] 0;\n')
        f.write('cp_sub_n[3] 0;\n')
        f.write('cp_sub_n[4] 0;\n')
        f.write('\n')
        f.write('// isotropic conductivity  - k - [1 1 -3 -1 0 0 0]\n')
        f.write('k_sub_n[0]  '+str(tc)+';\n')
        f.write('k_sub_n[1]  0;\n')
        f.write('k_sub_n[2]  0;\n')
        f.write('k_sub_n[3]  0;\n')
        f.write('k_sub_n[4]  0;\n')
        f.write('\n')
        f.write('// density - rho - [1 -3 0 0 0 0 0]\n')
        f.write('rho_sub_n[0]    '+str(density)+';\n')
        f.write('rho_sub_n[1]    0;\n')
        f.write('rho_sub_n[2]    0;\n')
        f.write('rho_sub_n[3]    0;\n')
        f.write('rho_sub_n[4]    0;\n')
        f.write('\n')
        f.write('// emissivity - e - [0 0 0 0 0 0 0]\n')
        f.write('e_sub_n[0]  '+str(em)+';\n')
        f.write('e_sub_n[1]  0;\n')
        f.write('e_sub_n[2]  0;\n')
        f.write('e_sub_n[3]  0;\n')
        f.write('e_sub_n[4]  0;\n')
        f.write('\n')
        f.write('Tmelt ' + str(obj.material.meltingTemperature) + ';\n')
        f.write('Tboil ' + str(obj.material.vaporizationTemperature) + ';\n')
        f.write('Hfusion ' + str(obj.material.meltingHeat) + ';\n')
        f.write('Hboil ' + str(obj.material.vaporizationHeat) + ';\n')
        f.write('fstrip ' + str(options.pato.fstrip) + ';\n')
        f.write('mass ' + str(obj.mass) + ';\n')
        f.write('density ' + str(obj.material.density) + ';\n')

    f.close()

def write_All_run_init(options, object_id):
    """
    Write the Allrun PATO file

    Generates an executable file to run a PATO simulation according to the state of the object and the user-defined parameters.

    Parameters
    ----------
    options: Options
        Object of class Options
    """

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(options.output_folder + '/PATO_'+str(object_id)+'/Allrun_init', 'w') as f:

        f.write('#!/bin/bash \n')
        f.write('cd ' + options.output_folder + '/PATO_'+str(object_id)+' \n')
        f.write('cp -r origin.0 0 \n')
        f.write('cd verification/unstructured_gmsh/ \n')
        f.write('ln -s ' + os.path.abspath(options.output_folder) + '/PATO_'+str(object_id)+'/mesh/mesh.msh \n')
        f.write('cd ../.. \n')
        f.write('gmshToFoam verification/unstructured_gmsh/mesh.msh \n')
        f.write('mv constant/polyMesh constant/subMat1 \n')
        f.write('count=`ls -1 processor* 2>/dev/null | wc -l`\n')
        f.write('if [ $count != 0 ];\n')
        f.write('then\n')
        f.write('    rm -rf processor*\n')
        f.write('fi\n')                                                                                                                                                                                                                             
        f.write('decomposePar -region subMat1\n')

    f.close()

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.system("chmod +x " + options.output_folder +'/PATO_'+str(object_id)+'/Allrun_init' )

    pass

def write_All_run(options, obj, time, iteration):
    """
    Write the Allrun PATO file

    Generates an executable file to run a PATO simulation according to the state of the object and the user-defined parameters.

    Parameters
    ----------
    options: Options
        Object of class Options
    """

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    end_time = time + options.dynamics.time_step
    start_time = time

    time_step_to_delete = np.round(time - options.dynamics.time_step, len(str(options.dynamics.time_step).lstrip('0.'))) # Round to time step sig figs
    #iteration_to_delete = int((iteration-1)*options.dynamics.time_step/options.pato.time_step)

    #print('copying BC:', end_time, ' - ', start_time)

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/Allrun', 'w') as f:

        if ((end_time).is_integer()): end_time = int(end_time)
        else: end_time = np.round(end_time, 5)
        if ((start_time).is_integer()): start_time = int(start_time)
        if ((time_step_to_delete).is_integer()): time_step_to_delete = int(time_step_to_delete)
        f.write('#!/bin/bash \n')
        f.write('cd ${0%/*} || exit 1 \n')
        f.write('. $PATO_DIR/src/applications/utilities/runFunctions/RunFunctions \n')
        f.write('pato_init \n')
        f.write('if [ "$(uname)" = "Darwin" ]; then\n')
        f.write('    source $FOAM_ETC/bashrc\n')
        f.write('    source $PATO_DIR/bashrc\n')
        f.write('fi\n')
        f.write('\n')
        f.write('if [ -z $1 ];\n')
        f.write('then\n')
        f.write('    echo "error: correct usage = ./Allrun_parallel <number_processors>"\n')
        f.write('    exit 1\n')
        f.write('fi\n')
        f.write('re="^[0-9]+$"\n')
        f.write('if ! [[ $1 =~ $re ]] ; then\n')
        f.write('   echo "error: First argument is not a number" >&2\n')
        f.write('   exit 1\n')
        f.write('fi\n')
        f.write('\n')
        f.write('NPROCESSOR=$1\n')
        f.write('\n')
        f.write('if [ "$(uname)" = "Darwin" ]; then\n')
        f.write('    sed_cmd=gsed\n')
        f.write('else\n')
        f.write('    sed_cmd=sed\n')
        f.write('fi\n')
        f.write('$sed_cmd -i "s/numberOfSubdomains \+[0-9]*;/numberOfSubdomains ""$NPROCESSOR"";/g" system/subMat1/decomposeParDict\n')
        f.write('cp qconv/BC_'+str(end_time) + ' qconv/BC_' + str(start_time) + '\n')
        f.write('mpiexec -np $NPROCESSOR PATOx -parallel \n')
        f.write('TIME_STEP='+str(end_time)+' \n')
        f.write('MAT_NAME=subMat1 \n')
        for n in range(options.pato.n_cores):
            f.write('cd processor' + str(n) + '/\n')
            f.write('cp -r "$TIME_STEP/$MAT_NAME"/* "$TIME_STEP" \n')
            f.write('cp -r constant/"$MAT_NAME"/polyMesh/  "$TIME_STEP"/ \n')
            f.write('cd .. \n')
        f.write('cp system/"$MAT_NAME"/fvSchemes  system/ \n')
        f.write('cp system/"$MAT_NAME"/fvSolution system/ \n')
        f.write('cp system/"$MAT_NAME"/decomposeParDict system/ \n')
        f.write('foamJob -p -s foamToVTK -time '+str(end_time)+' -useTimeName\n')
        f.write('cp qconv/BC* qconv-bkp/ \n')
        f.write('rm qconv/BC* \n')
        f.write('rm mesh/*su2 \n')
        #f.write('rm mesh/*meshb \n')
        #print('time_step_to_delete:', time_step_to_delete)
        #print('end_time:', end_time)
        #print('start_time:', start_time)
        for n in range(options.pato.n_cores):
            if not options.pato.solution_type=='volume':
                f.write('rm -rf processor'+str(n)+'/VTK/proc* \n')
            f.write('rm processor'+str(n)+'/VTK/top/top_'+str(time_step_to_delete)+'.vtk \n')
            #f.write('rm -rf processor'+str(n)+'/restart/* \n')
            if options.current_iter%options.save_freq == 0:
                f.write('rm -rf processor'+str(n)+'/restart/* \n')
                f.write('cp -r  processor'+str(n)+'/'+str(start_time)+'/ processor'+str(n)+'/restart/ \n')
            f.write('rm -rf processor'+str(n)+'/'+str(time_step_to_delete)+' \n')
        #f.write('rm -rf processor'+str(n)+'/'+str(time_step_to_delete)+' \n')
        #if time_step_to_delete/options.dynamics.time_step != options.save_freq:
            #f.write('rm -rf processor'+str(n)+'/'+str(time_step_to_delete)+' \n')
            #print('delete time_step_to_delete:', time_step_to_delete)
        #if options.current_iter%options.save_freq != 0:
        #    f.write('rm -rf processor'+str(n)+'/'+str(end_time)+' \n')
        

    f.close()

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.system("chmod +x " + options.output_folder +'/PATO_'+str(obj.global_ID)+'/Allrun' )

    return end_time

def write_constant_folder(options, object_id):
    """
    Write the constant/ PATO folder

    Generates input files defining the 'constant' folder in PATO

    Parameters
    ----------
    options: Options
        Object of class Options
    """

    with open(options.output_folder + '/PATO_'+str(object_id)+'/constant/regionProperties', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  2.1.x                                 |\n')
        f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version     2.0;\n')
        f.write('  format      ascii;\n')
        f.write('  class       dictionary;\n')
        f.write('  location    "constant";\n')
        f.write('  object      regionProperties;\n')
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('regions\n')
        f.write('(\n')
        f.write('    solid       (subMat1) // regions to be defined in blockMeshDict\n')
        f.write(');\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')

    f.close()

    if options.pato.Ta_bc == 'qconv':

        with open(options.output_folder + '/PATO_'+str(object_id)+'/constant/subMat1/subMat1Properties', 'w') as f:
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  5.0                                   |\n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     4.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       dictionary;\n')
            f.write('  location    "constant/subMat1";\n')
            f.write('  object      subMat1Properties;\n')
            f.write('}\n')
            f.write('\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('/****************************** GENERAL ************************************/\n')
            f.write('//debug yes;\n')
            f.write('movingMesh      no;\n')
            f.write('/****************************** end GENERAL ********************************/\n')
            f.write('\n')
            f.write('/****************************** IO *****************************************/\n')
            f.write('IO {\n')
            f.write('  writeFields(); // write fields in the time folders\n')
            f.write('}\n')
            f.write('/****************************** END IO ************************************/\n')
            f.write('\n')
            f.write('/****************************** MASS **************************************/\n')
            f.write('Mass {\n')
            f.write('  createFields ((p volScalarField)); // read pressure [Pa]\n')
            f.write('}\n')
            f.write('/****************************** END MASS **********************************/\n')
            f.write('\n')
            f.write('/****************************** ENERGY ************************************/\n')
            f.write('Energy {\n')
            f.write('  EnergyType PureConduction; // Solve the temperature equation\n')
            f.write('}\n')
            f.write('/****************************** END ENERGY ********************************/\n')
            f.write('\n')
            f.write('/****************************** MATERIAL PROPERTIES  ************************/\n')
            f.write('MaterialProperties {\n')
            f.write('  MaterialPropertiesType Fourier; \n')
            f.write('  MaterialPropertiesDirectory "$FOAM_CASE/data"; \n')
            f.write('}\n')
            f.write('/****************************** END MATERIAL PROPERTIES  ********************/\n')
    
        f.close()


    if options.pato.Ta_bc == 'ablation':

        with open(options.output_folder + '/PATO_'+str(object_id)+'/constant/subMat1/subMat1Properties', 'w') as f:
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  5.0                                   |\n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     4.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       dictionary;\n')
            f.write('  location    "constant/subMat1";\n')
            f.write('  object      subMat1Properties;\n')
            f.write('}\n')
            f.write('\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('/****************************** GENERAL ************************************/\n')
            f.write('//debug yes;\n')
            f.write('movingMesh      yes;\n')
            f.write('/****************************** end GENERAL ********************************/\n')
            f.write('\n')
            f.write('/****************************** IO *****************************************/\n')
            f.write('IO {\n')
            f.write('  writeFields(); // write fields in the time folders\n')
            f.write('}\n')
            f.write('/****************************** END IO ************************************/\n')
            f.write('\n')
            f.write('/****************************** MASS, ENERGY, PYROLYSIS **************************************/\n')
            f.write('MaterialProperties {\n')
            f.write('  MaterialPropertiesType Fourier; \n')
            f.write('  MaterialPropertiesDirectory "$FOAM_CASE/data"; \n')
            f.write('}\n')
            f.write('Mass {\n')
            f.write('  MassType no; // Solve the semi implicit pressure equation\n')
            f.write('  createFields ((p volScalarField) (mDotG volVectorField) (mDotGw volScalarField) (mDotVapor volScalarField) (mDotMelt volScalarField));\n')
            f.write('}\n')
            f.write('Energy {\n')
            f.write('  EnergyType PureConduction; // Solve the temperature equation\n')
            f.write('}\n')
            f.write('/****************************** MASS, ENERGY, PYROLYSIS **********************************/\n')
            f.write('\n')
            f.write('/****************************** GAS PROPERTIES  ************************************/\n')
            f.write('GasProperties {\n')
            f.write('  GasPropertiesType no; // tabulated gas properties\n')
            f.write('  createFields ((h_g volScalarField));\n')
            f.write('}\n')
            f.write('/****************************** END GAS PROPERTIES **************************/\n')
            f.write('\n')
            f.write('/****************************** TIME CONTROL  **********************************/\n')
            f.write('TimeControl {\n')
            f.write('  TimeControlType no; // change the integration time step in function of the gradient of the pressure and the species mass fractions\n')
            f.write('  chemTransEulerStepLimiter no;\n')
            f.write('}\n')
            f.write('/****************************** END TIME CONTROL  ******************************/\n')
    
        f.close()

        with open(options.output_folder + '/PATO_'+str(object_id)+'/constant/subMat1/dynamicMeshDict', 'w') as f:
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('| =========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  5.0                                   | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*---------------------------------------------------------------------------*/ \n')
            f.write('FoamFile { \n')
            f.write('  version         4.0; \n')
            f.write('  format          ascii; \n')
            f.write('  class           dictionary; \n')
            f.write('  object          dynamicMeshDict; \n')
            f.write('} \n')
            f.write(' \n')
            f.write('/* * *          User-defined mesh motion parameters      * * */ \n')
            f.write('// For code initialization - Do NOT modify \n')
            f.write('dynamicFvMesh           dynamicMotionSolverFvMesh;              // mesh motion class \n')
            f.write('solver                  velocityLaplacian;                      // mesh motion solver \n')
            f.write('velocityLaplacianCoeffs { \n')
            f.write('  diffusivity          uniform;                                // try quadratic if topology is lost due to large dispacements \n')
            f.write('} \n')
            f.write('v0                      v0 [ 0 1 -1 0 0 0 0 ]   (0 0 0);        // initialization of the recession velocity (t=0) \n')
    
        f.close()

        with open(options.output_folder + '/PATO_'+str(object_id)+'/constant/subMat1/BoundaryConditions', 'w') as f:
    
            f.write('/*---------------------------------------------------------------------------*\\n')
            f.write('BoundaryConditions\n')
            f.write('\n')
            f.write('Application\n')
            f.write('    Provides boundary-condition information at the surface, tabulated as a function of time.\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('/*\n')
            f.write('t(s)    p_total_w(Pa)   rhoUeCH(kg/m²/s)    h_r(J/kg)   chemistryOn\n')
            f.write('*/\n')
            f.write('0       101325          0.3e-2                  0               1\n')
            f.write('0.1     101325          0.3                     2.5e7           1\n')
            f.write('60      101325          0.3                     2.5e7           1\n')
            f.write('60.1    101325          0.3e-2                  0               0\n')
            f.write('120     101325          0.3e-2                  0               0\n')
    
        f.close()

    pass

def write_origin_folder(options, obj):
    """
    Write the origin.0/ PATO folder

    Generates input files defining the 'origin.0' folder in PATO

    Parameters
    ----------
    options: Options
        Object of class Options
    """

    Ta_bc = options.pato.Ta_bc

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/p', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  5.0                                   |\n')
        f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version     2.0;\n')
        f.write('  format      ascii;\n')
        f.write('  class       volScalarField;\n')
        f.write('  location    "0";\n')
        f.write('  object      p;\n')
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 0 1 0 0 0 0];\n')
        f.write('\n')
        f.write('internalField   uniform 101325;\n')
        f.write('\n')
        f.write('boundaryField {\n')
        f.write('  top\n')
        f.write('  {\n')
        f.write('    type            fixedValue;\n')
        f.write('    value           uniform 101325;\n')
        f.write('  }\n')
        f.write('}\n')
        f.write('\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')        


    f.close()

    #This is actually not used inside PATO, as we are not using the Bprime mutation++ surfaceMassBalance
    mix_file = 'tacot26'

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/Ta', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  5.0                                   |\n')
        f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version     2.0;\n')
        f.write('  format      ascii;\n')
        f.write('  class       volScalarField;\n')
        f.write('  location    "0";\n')
        f.write('  object      Ta;\n')
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 0 0 1 0 0 0];\n')
        f.write('\n')
        f.write('internalField   uniform '+str(obj.pato.initial_temperature)+';\n')
        f.write('\n')
        f.write('boundaryField {\n')
        f.write('  top\n')
        f.write('  {\n')

        if Ta_bc == "fixed":
            f.write('    type             uniformFixedValue;\n')
            f.write('    uniformValue table\n')
            f.write('    (\n')
            f.write('        (0   1644)\n')
            f.write('        (0.1   1644)\n')
            f.write('        (0.2   1644)\n')
            f.write('        (120 1644)\n')
            f.write('    );\n')
        elif Ta_bc == "qconv":
            f.write('type            HeatFlux;\n')
            f.write('mappingType     "3D-tecplot";\n')
            f.write('mappingFileName "$FOAM_CASE/qconv/BC";\n')
            f.write('mappingFields   (\n')
            f.write('    (qConvCFD "3")\n')
            f.write('    (emissivity "4")\n')
            f.write('    (Tbackground "5")\n')
            f.write(');\n')
            f.write('p 101325;\n')
            f.write('chemistryOn 1;\n')
            f.write('qRad 0;\n')
            f.write('value           uniform 300;\n')
        elif Ta_bc == "ablation":
            f.write('type             Bprime;\n')
            f.write('mixtureMutationBprime '+(mix_file)+';\n')
            f.write('environmentDirectory "$PATO_DIR/data/Environments/RawData/Earth";\n')
            f.write('movingMesh yes;\n')
            f.write('mappingType "3D-tecplot";\n')
            f.write('mappingFileName "$FOAM_CASE/qconv/BC";\n')
            f.write('mappingFields\n')
            f.write('(\n')
            f.write('    (qConv "3")\n')
            f.write('    (emissivity "4")\n')
            f.write('    (Tbackground "5")\n')
            f.write('    (molten "6")\n')
            f.write(');\n')
            f.write('chemistryOn 1;\n')
            f.write('qRad 0;\n')
            f.write('lambda 0.5;\n')
            f.write('Tedge 300;\n')
            f.write('hconv 0;\n')
            f.write('value uniform 300;\n')
            f.write('moleFractionGasInMaterial ( ("O"  0.115) ("N" 0) ("C" 0.206) ("H" 0.679));\n')  
        f.write('  }\n')
        f.write('}\n')
        f.write('\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')

    f.close()

    if Ta_bc == "ablation":

        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/cellMotionU', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  4.x                                   |\n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     5.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       volVectorField;\n')
            f.write('  location    "0/porousMat";\n')
            f.write('  object      cellMotionU;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('dimensions      [0 1 -1 0 0 0 0];\n')
            f.write('\n')
            f.write('internalField   uniform (0 0 0);\n')
            f.write('\n')
            f.write('boundaryField {\n')
            f.write('  top\n')
            f.write('  {\n')
            f.write('    type            fixedValue;\n')
            f.write('    value           uniform (0 0 0);\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')
    
        f.close()
    
        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/h_g', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
            f.write('  =========                 |\n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
            f.write('    \\  /    A nd           | Version:  7\n')
            f.write('     \\/     M anipulation  |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     2.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       volScalarField;\n')
            f.write('  location    "1/porousMat";\n')
            f.write('  object      h_g;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('dimensions      [0 2 -2 0 0 0 0];\n')
            f.write('\n')
            f.write('internalField   uniform 0.;\n')
            f.write(';\n')
            f.write('\n')
            f.write('boundaryField {\n')
            f.write('  top\n')
            f.write('  {\n')
            f.write('    type            calculated;\n')
            f.write('    value           uniform 0.;\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')
            
        f.close()
    
        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/mDotG', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
            f.write('  =========                 |\n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
            f.write('    \\  /    A nd           | Version:  7\n')
            f.write('     \\/     M anipulation  |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     2.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       volVectorField;\n')
            f.write('  location    "1/porousMat";\n')
            f.write('  object      mDotG;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('dimensions      [1 -2 -1 0 0 0 0];\n')
            f.write('\n')
            f.write('internalField   uniform (-0 0 -0);\n')
            f.write('\n')
            f.write('boundaryField {\n')
            f.write('  top\n')
            f.write('  {\n')
            f.write('    type            calculated;\n')
            f.write('    value           uniform (-0 0 -0);\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')
    
        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/mDotGw', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('  =========                 | \n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\  /    A nd           | Version:  7 \n')
            f.write('     \\/     M anipulation  | \n')
            f.write('\*---------------------------------------------------------------------------*/ \n')
            f.write('FoamFile { \n')
            f.write('  version     2.0; \n')
            f.write('  format      ascii; \n')
            f.write('  class       volScalarField; \n')
            f.write('  location    "1/porousMat"; \n')
            f.write('  object      mDotGw; \n')
            f.write('} \n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n')
            f.write(' \n')
            f.write('dimensions      [1 -2 -1 0 0 0 0]; \n')
            f.write(' \n')
            f.write('internalField   uniform 0; \n')
            f.write(' \n')
            f.write('boundaryField { \n')
            f.write('  top \n')
            f.write('  { \n')
            f.write('    type            calculated; \n')
            f.write('    value           uniform 0.; \n')
            f.write('  } \n')
            f.write('} \n')
            f.write(' \n')
            f.write(' \n')
            f.write('// ************************************************************************* // \n')
            
        f.close()
    
        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/pointMotionU', 'w') as f:
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  4.x                                   |\n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     5.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       pointVectorField;\n')
            f.write('  location    "0/porousMat";\n')
            f.write('  object      pointMotionU;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('dimensions      [0 1 -1 0 0 0 0];\n')
            f.write('\n')
            f.write('internalField   uniform (0 0 0);\n')
            f.write('\n')
            f.write('boundaryField {\n')
            f.write('  top\n')
            f.write('  {\n')
            f.write('    type            calculated;\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')
    
        f.close()
    
        density = obj.material.density
    
        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/rho_s', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  5.0                                   |\n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     2.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       volScalarField;\n')
            f.write('  location    "0";\n')
            f.write('  object      rho_s;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('dimensions      [1 -3 0 0 0 0 0];\n')
            f.write('\n')
            f.write('internalField   uniform '+str(density)+';\n')
            f.write('\n')
            f.write('boundaryField {\n')
            f.write('  top\n')
            f.write('  {\n')
            f.write('    type            calculated;\n')
            f.write('    value uniform '+str(density)+';\n')
            f.write('  }\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')
    
        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/vG', 'w') as f:
            
             f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
             f.write('| =========                 |                                                 |\n')
             f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
             f.write('|  \\    /   O peration     | Version:  5.0                                   |\n')
             f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
             f.write('|    \\/     M anipulation  |                                                 |\n')
             f.write('\*---------------------------------------------------------------------------*/\n')
             f.write('FoamFile {\n')
             f.write('  version     2.0;\n')
             f.write('  format      ascii;\n')
             f.write('  class       volScalarField;\n')
             f.write('  location    "0";\n')
             f.write('  object      vG;\n')
             f.write('}\n')
             f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
             f.write('\n')
             f.write('dimensions      [0 1 -1 0 0 0 0];\n')
             f.write('\n')
             f.write('internalField   uniform 0.;\n')
             f.write('\n')
             f.write('boundaryField {\n')
             f.write('  top\n')
             f.write('  {\n')
             f.write('    type            zeroGradient;\n')
             f.write('  }\n')
             f.write('}\n')
             f.write('\n')
             f.write('\n')
             f.write('// ************************************************************************* //\n')
    
        f.close()

        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/mDotMelt', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('  =========                 | \n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\  /    A nd           | Version:  7 \n')
            f.write('     \\/     M anipulation  | \n')
            f.write('\*---------------------------------------------------------------------------*/ \n')
            f.write('FoamFile { \n')
            f.write('  version     2.0; \n')
            f.write('  format      ascii; \n')
            f.write('  class       volScalarField; \n')
            f.write('  location    "1/porousMat"; \n')
            f.write('  object      mDotMelt; \n')
            f.write('} \n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n')
            f.write(' \n')
            f.write('dimensions      [1 -2 -1 0 0 0 0]; \n')
            f.write(' \n')
            f.write('internalField   uniform 0; \n')
            f.write(' \n')
            f.write('boundaryField { \n')
            f.write('  top \n')
            f.write('  { \n')
            f.write('    type            calculated; \n')
            f.write('    value           uniform 0.0; \n')
            f.write('  } \n')
            f.write('} \n')
            f.write(' \n')
            f.write(' \n')
            f.write('// ************************************************************************* // \n')

        f.close()

        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/mDotVapor', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('  =========                 | \n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\  /    A nd           | Version:  7 \n')
            f.write('     \\/     M anipulation  | \n')
            f.write('\*---------------------------------------------------------------------------*/ \n')
            f.write('FoamFile { \n')
            f.write('  version     2.0; \n')
            f.write('  format      ascii; \n')
            f.write('  class       volScalarField; \n')
            f.write('  location    "1/porousMat"; \n')
            f.write('  object      mDotVapor; \n')
            f.write('} \n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n')
            f.write(' \n')
            f.write('dimensions      [1 -2 -1 0 0 0 0]; \n')
            f.write(' \n')
            f.write('internalField   uniform 0; \n')
            f.write(' \n')
            f.write('boundaryField { \n')
            f.write('  top \n')
            f.write('  { \n')
            f.write('    type            calculated; \n')
            f.write('    value           uniform 0.0; \n')
            f.write('  } \n')
            f.write('} \n')
            f.write(' \n')
            f.write(' \n')
            f.write('// ************************************************************************* // \n')

        f.close()

    pass

def write_PATO_BC(options, obj, time, conv_heatflux, freestream_temperature):

    # write tecplot file with facet_COG coordinates and associated facet quantities

    emissivity = obj.material.emissivity(obj.pato.temperature)
    emissivity = np.clip(emissivity, 0, 1)  

    n_data_points = len(obj.mesh.facet_COG)

    x = obj.mesh.facet_COG[:,0]
    y = obj.mesh.facet_COG[:,1]
    z = obj.mesh.facet_COG[:,2]
    Tinf = np.full(n_data_points, freestream_temperature)

    if ((time).is_integer()): time = int(time)  

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/qconv/BC_' + str(time), 'w') as f:

        if options.pato.Ta_bc == "qconv":

            f.write('TITLE     = "vol-for-blayer.fu"\n')
            f.write('VARIABLES = \n')
            f.write('"xw (m)"\n')
            f.write('"yw (m)"\n')
            f.write('"zw (m)"\n')
            f.write('"qConvCFD (W/m^2)"\n')
            f.write('"emissivity (-)"\n')
            f.write('"Tbackground (K)"\n')
            f.write('ZONE T="zone 1"\n')
            f.write(' STRANDID=0, SOLUTIONTIME=0\n')
            f.write(' I=' + str(n_data_points) + ', J=1, K=1, ZONETYPE=Ordered\n')
            f.write(' DATAPACKING=BLOCK\n')
            f.write(' DT=(DOUBLE DOUBLE DOUBLE DOUBLE)   \n')
            f.write(np.array2string(x)[1:-1]+' ')
            f.write(np.array2string(y)[1:-1]+' ')
            f.write(np.array2string(z)[1:-1]+' ')
            f.write(np.array2string(conv_heatflux)[1:-1]+' ')
            f.write(np.array2string(emissivity)[1:-1]+' ')
            f.write(np.array2string(Tinf)[1:-1]+' ')

        if options.pato.Ta_bc == "ablation":

            f.write('TITLE     = "vol-for-blayer.fu"\n')
            f.write('VARIABLES = \n')
            f.write('"xw (m)"\n')
            f.write('"yw (m)"\n')
            f.write('"zw (m)"\n')
            f.write('"pw (Pa)"\n')
            f.write('"qConv (W/m^2)"\n')
            f.write('"emissivity (-)"\n')
            f.write('"Tbackground (K)"\n')
            f.write('"molten (-)"\n')
            f.write('ZONE T="zone 1"\n')
            f.write(' STRANDID=0, SOLUTIONTIME=0\n')
            f.write(' I=' + str(n_data_points) + ', J=1, K=1, ZONETYPE=Ordered\n')
            f.write(' DATAPACKING=BLOCK\n')
            f.write(' DT=(DOUBLE DOUBLE DOUBLE DOUBLE)   \n')
            f.write(np.array2string(x)[1:-1]+' ')
            f.write(np.array2string(y)[1:-1]+' ')
            f.write(np.array2string(z)[1:-1]+' ')
            f.write(np.array2string(conv_heatflux)[1:-1]+' ')
            f.write(np.array2string(emissivity)[1:-1]+' ')
            f.write(np.array2string(Tinf)[1:-1]+' ')
            f.write(np.array2string(obj.pato.molten)[1:-1]+' ')

    f.close()

    pass

def write_system_folder(options, object_id, time):
    """
    Write the system/ PATO folder

    Generates input files defining the 'system' folder in PATO

    Parameters
    ----------
    options: Options
        Object of class Options
    """
    start_time = time
    end_time = time + options.dynamics.time_step
    wrt_interval = end_time - start_time
    pato_time_step = options.pato.time_step

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/controlDict', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  1.5                                   |\n')
        f.write('|   \\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version     2.0;\n')
        f.write('  format      ascii;\n')
        f.write('  class       dictionary;\n')
        f.write('  object      controlDict;\n')
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('application     PATOx;\n')
        f.write('\n')
        f.write('startFrom       startTime;\n')
        f.write('\n')
        f.write('startTime       '+str(start_time)+';\n')
        f.write('\n')
        f.write('stopAt          endTime;\n')
        f.write('\n')
        f.write('endTime         '+str(end_time)+';\n')
        f.write('\n')
        f.write('deltaT          '+str(pato_time_step)+';\n')
        f.write('\n')
        f.write('writeControl    adjustableRunTime;\n')
        f.write('\n')
        f.write('writeInterval   '+str(wrt_interval)+';\n')
        f.write('\n')
        f.write('purgeWrite      0;\n')
        f.write('\n')
        f.write('writeFormat     ascii;\n')
        f.write('\n')
        f.write('writePrecision  6;\n')
        f.write('\n')
        f.write('writeCompression uncompressed;\n')
        f.write('\n')
        f.write('timeFormat      general;\n')
        f.write('\n')
        f.write('timePrecision   6;\n')
        f.write('\n')
        f.write('graphFormat     xmgr;\n')
        f.write('\n')
        f.write('runTimeModifiable yes;\n')
        f.write('\n')
        f.write('adjustTimeStep  yes; // you can turn it off but its going to be very slow\n')
        f.write('\n')
        f.write('maxCo           10;\n')
        f.write('\n')
        f.write('maxDeltaT   0.1; // reduce it if the surface temperature starts oscilliating\n')
        f.write('\n')
        f.write('minDeltaT   1e-6;\n')
        f.write('\n')
        f.write('REVlength   1e3;\n')
        f.write('// ************************************************************************* //\n')

    f.close()

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/fvSchemes', 'w') as f:

        f.write(' /*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  1.5                                   |\n')
        f.write('|   \\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version     2.0;\n')
        f.write('  format      ascii;\n')
        f.write('  class       dictionary;\n')
        f.write('  object      fvSchemes;\n')
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('ddtSchemes {\n')
        f.write('default            Euler; // backward;\n')
        f.write('}\n')
        f.write('\n')
        f.write('gradSchemes {\n')
        f.write('default           Gauss linear corrected;\n')
        f.write('}\n')
        f.write('\n')
        f.write('divSchemes {\n')
        f.write('default             Gauss  linear corrected;\n')
        f.write('}\n')
        f.write('\n')
        f.write('laplacianSchemes {\n')
        f.write('default             Gauss linear corrected;\n')
        f.write('}\n')
        f.write('\n')
        f.write('interpolationSchemes {\n')
        f.write('default         linear;\n')
        f.write('}\n')
        f.write('\n')
        f.write('snGradSchemes {\n')
        f.write('default         corrected;\n')
        f.write('}\n')
        f.write('\n')
        f.write('fluxRequired { // used for the ALE correction\n')
        f.write('default         no;\n')
        f.write('  Ta;\n')
        f.write('}\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')


    f.close()

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/fvSolution', 'w') as f:

        if (options.pato.Ta_bc != 'ablation'):

            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  1.5                                   |\n')
            f.write('|   \\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     2.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       dictionary;\n')
            f.write('  object      fvSolution;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('solvers {\n')
            f.write('  Ta\n')
            f.write('  {\n')
            f.write('    solver           GAMG;\n')
            f.write('    tolerance        1e-06;\n')
            f.write('    relTol           0.01;\n')
            f.write('    smoother         GaussSeidel;\n')
            f.write('    cacheAgglomeration true;\n')
            f.write('    nCellsInCoarsestLevel 2;\n')
            f.write('    agglomerator     faceAreaPair;\n')
            f.write('    mergeLevels      1;\n')
            f.write('  };\n')
            f.write('}\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')


        else:

            f.write('/*--------------------------------*- C++ -*----------------------------------*\\n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\    /   O peration     | Version:  1.5                                   |\n')
            f.write('|   \\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
            f.write('|    \\/     M anipulation  |                                                 |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile {\n')
            f.write('  version     2.0;\n')
            f.write('  format      ascii;\n')
            f.write('  class       dictionary;\n')
            f.write('  object      fvSolution;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('solvers {\n')
            f.write('  Ta\n')
            f.write('  {\n')
            f.write('    solver       PBiCGStab; // asymmetric matrix solver (for mesh motion)\n')
            f.write('    preconditioner   DIC;\n')
            f.write('    tolerance        1e-06;\n')
            f.write('    relTol           0;\n')
            f.write('  };\n')
            f.write('\n')
            f.write('  p\n')
            f.write('  {\n')
            f.write('    solver       PBiCGStab; // asymmetric matrix solver (for mesh motion)\n')
            f.write('    preconditioner   DILU;\n')
            f.write('    tolerance        1e-07;\n')
            f.write('    relTol           0;\n')
            f.write('  };\n')
            f.write('\n')
            f.write('  Xsii\n')
            f.write('  {\n')
            f.write('    solver       PBiCGStab; // asymmetric matrix solver (for mesh motion)\n')
            f.write('    preconditioner   DILU;\n')
            f.write('    tolerance        1e-10;\n')
            f.write('    relTol           1e-06;\n')
            f.write('  };\n')
            f.write('\n')
            f.write('  cellMotionU\n')
            f.write('  {\n')
            f.write('    solver          PCG;\n')
            f.write('    preconditioner  DIC;\n')
            f.write('    tolerance       1e-08;\n')
            f.write('    relTol          0;\n')
            f.write('  };\n')
            f.write('\n')
            f.write('}\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')

    f.close()

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/plotDict', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  dev                                   |\n')
        f.write('|   \\  /    A nd           | Web:      http://www.openfoam.org               |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version         5.0;\n')
        f.write('  format          ascii;\n')
        f.write('  class           dictionary;\n')
        f.write('  location        system/subMat1;\n')
        f.write('  object          plotDict;\n')
        f.write('}\n')
        f.write('\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('type sets;\n')
        f.write('libs ("libsampling.so");\n')
        f.write('\n')
        f.write('interpolationScheme cellPoint;\n')
        f.write('\n')
        f.write('setFormat         raw;\n')
        f.write('surfaceFormat     raw;\n')
        f.write('\n')
        f.write('sets\n')
        f.write('(\n')
        f.write('plot {\n')
        f.write('  type            points;\n')
        f.write('  ordered on;\n')
        f.write('  axis            xyz;\n')
        f.write('  points          (\n')
        f.write('      (0 0.049 0)\n')
        f.write('      (0 0.048 0)\n')
        f.write('      (0 0.046 0)\n')
        f.write('      (0 0.042 0)\n')
        f.write('      (0 0.038 0)\n')
        f.write('      (0 0.034 0)\n')
        f.write('      (0 0.026 0)\n')
        f.write('  );\n')
        f.write('}\n')
        f.write(');\n')
        f.write('\n')
        f.write('fields\n')
        f.write('(\n')
        f.write('    Ta\n')
        f.write(');\n')
        f.write('\n')
        f.write('// *********************************************************************** //\n')


    f.close()

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/surfacePatchDict', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\    /   O peration     | Version:  dev                                   |\n')
        f.write('|   \\  /    A nd           | Web:      http://www.openfoam.org               |\n')
        f.write('|    \\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile {\n')
        f.write('  version         5.0;\n')
        f.write('  format          ascii;\n')
        f.write('  class           dictionary;\n')
        f.write('  location        system/subMat1;\n')
        f.write('  object          surfacePatchDict;\n')
        f.write('}\n')
        f.write('\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('type sets;\n')
        f.write('libs ("libsampling.so");\n')
        f.write('\n')
        f.write('interpolationScheme cellPatchConstrained;\n')
        f.write('\n')
        f.write('setFormat         raw;\n')
        f.write('surfaceFormat     raw;\n')
        f.write('\n')
        f.write('sets\n')
        f.write('(\n')
        f.write('surfacePatch {\n')
        f.write('  type            boundaryPoints;\n')
        f.write('  axis            xyz;\n')
        f.write('  points          (\n')
        f.write('      (0 0.05 0)\n')
        f.write('      (0 0 0)\n')
        f.write('  );\n')
        f.write('  maxDistance     1e-3;\n')
        f.write('  patches         (".*");\n')
        f.write('}\n')
        f.write(');\n')
        f.write('\n')
        f.write('fields\n')
        f.write('(\n')
        f.write('    Ta\n')
        f.write(');\n')
        f.write('\n')
        f.write('// *********************************************************************** //\n')

    f.close()

    n_proc = options.pato.n_cores

    coeff_0 = 1#n_proc/2
    coeff_1 = 2
    coeff_2 = 1

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/decomposeParDict', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 | \n')
        f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
        f.write('|  \\    /   O peration     | Version:  4.x                                   | \n')
        f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      | \n')
        f.write('|    \\/     M anipulation  |                                                 | \n')
        f.write('\*---------------------------------------------------------------------------*/ \n')
        f.write('FoamFile { \n')
        f.write('  version     2.0; \n')
        f.write('  format      ascii; \n')
        f.write('  class       dictionary; \n')
        f.write('  location    "system"; \n')
        f.write('  object      decomposeParDict; \n')
        f.write('} \n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n')
        f.write(' \n')
        f.write('numberOfSubdomains '+str(n_proc)+'; \n')
        f.write(' \n')
        f.write('method          scotch; \n')
        f.write(' \n')
        f.write('simpleCoeffs { \n')
        f.write('  n           ('+str(coeff_0) + ' ' + str(coeff_1) + ' ' + str(coeff_2) + '); \n')
        f.write('  delta       0.001; \n')
        f.write('} \n')
        f.write(' \n')
        f.write('hierarchicalCoeffs { \n')
        f.write('  n           ('+str(coeff_0) + ' ' + str(coeff_1) + ' ' + str(coeff_2) + '); \n')
        f.write('  delta       0.001; \n')
        f.write('  order       xyz; \n')
        f.write('} \n')
        f.write(' \n')
        f.write('scotchCoeffs { \n')
        f.write('} \n')
        f.write(' \n')
        f.write('manualCoeffs { \n')
        f.write('  dataFile    "decompositionData"; \n')
        f.write('} \n')
        f.write(' \n')
        f.write('// ************************************************************************* // \n')


    f.close()

    pass


def initialize(options, obj):
    """
    Calls the PATO executable and run the simulation

    Parameters
    ----------
    ?????????????????????????
    """
    object_id   = obj.global_ID

    write_All_run_init(options,object_id)
    write_constant_folder(options, object_id)
    write_origin_folder(options, obj)
    write_material_properties(options, obj)
    write_system_folder(options, object_id, 0)

    n_proc = options.pato.n_cores

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pato_test = subprocess.run(conda_preamble+['echo', 'PATO environment working!'])
    if pato_test.returncode>0: raise Exception('Error could not find PATO environment! Check you have a conda env named \'pato\'')
    print('Running PATO initialisation...')
    subprocess.run(conda_preamble+[options.output_folder + '/PATO_'+str(object_id)+'/Allrun_init',str(n_proc)],text = True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_PATO(options, object_id):
    """
    Calls the PATO executable and run the simulation

    Parameters
    ----------
	?????????????????????????
    """
    n_proc = options.pato.n_cores

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print('Running PATO simulation...')
    subprocess.run(conda_preamble+[options.output_folder + '/PATO_'+str(object_id)+'/Allrun',str(n_proc)], text = False, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    dir_VTK = pathlib.Path(options.output_folder + '/PATO_'+str(object_id)+'/VTK').glob('**/*.vtk')
    for link in dir_VTK:
        if link.is_symlink():
            link.unlink() # Remove broken symlinks
            if not link.exists(): 
                pass
def postprocess_PATO_solution(options, obj, time_to_read):
    """
    Postprocesses the PATO output

    Parameters
    ----------
	?????????????????????????
    """ 

    if options.pato.Ta_bc == 'ablation': postprocess_mass_inertia(obj, options, time_to_read)

    path = options.output_folder+"/PATO_"+str(obj.global_ID)+"/"

    #iteration_to_read = int(round((iteration+1)*options.dynamics.time_step/options.pato.time_step))

    n_proc = options.pato.n_cores

    solution = options.pato.solution_type

    if solution == 'surface':
        data = retrieve_surface_vtk_data(n_proc, path, time_to_read)
    elif solution == 'volume':
        data = retrieve_volume_vtk_data(n_proc, path, time_to_read)

    # extract distribution
    cell_data = data.GetCellData()
    n_cells=data.GetNumberOfCells()

    #extract temperature distribution
    temperature = cell_data.GetArray('Ta')
    temperature_cell = [temperature.GetValue(i) for i in range(n_cells)]
    temperature_cell = np.array(temperature_cell)

    #extract mDotVapor distribution if BC ablation is used
    if options.pato.Ta_bc == "ablation":
        mDotVapor = cell_data.GetArray('mDotVapor')
        mDotVapor_cell = [mDotVapor.GetValue(i) for i in range(n_cells)]
        mDotVapor_cell = np.array(mDotVapor_cell)
        mDotMelt = cell_data.GetArray('mDotMelt')
        mDotMelt_cell = [mDotMelt.GetValue(i) for i in range(n_cells)]
        mDotMelt_cell = np.array(mDotMelt_cell)

    # mapping: sort vtk and TITAN surface mesh cell numbering by checking facet COG

    # get cell COG from vtk
    vtk_cell_centers=vtk.vtkCellCenters()
    vtk_cell_centers.SetInputData(data)
    vtk_cell_centers.Update()
    vtk_cell_centers_data = vtk_cell_centers.GetOutput()
    vtk_COG = vtk_to_numpy(vtk_cell_centers_data.GetPoints().GetData())
    
    mapping = mapping_facetCOG_TITAN_PATO(obj.mesh.facet_COG, vtk_COG)

    #retrieve solution
    obj.pato.temperature = temperature_cell[mapping]
    obj.temperature = obj.pato.temperature

    #print('obj ID:', obj.global_ID)
    #print('max temp:', max(obj.temperature))

    if options.pato.Ta_bc == "ablation":
        obj.pato.mDotVapor = mDotVapor_cell[mapping]
        obj.pato.mDotMelt = mDotMelt_cell[mapping]
        obj.pato.molten[obj.temperature == obj.material.meltingTemperature] = 1

def postprocess_mass_inertia(obj, options, time_to_read):

    # Define the file path
    file_path = options.output_folder + "/PATO_" + str(obj.global_ID) + "/processor0/" + str(time_to_read) + "/subMat1/uniform/massFile" 
    #file_path = options.output_folder + "PATO_" + str(obj.global_ID) + "/" + str(time_to_read) + "/subMat1/uniform/massFile"    
    # Initialize variables to store mass and density
    new_mass = None
    density_ratio = None
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            if 'new_mass' in line:
                print('line:', line)
                new_mass = float(line.split()[1].strip(';'))
            elif 'density_ratio' in line:
                density_ratio = float(line.split()[1].strip(';'))
            
            # Exit early once both values are found
            if new_mass is not None and density_ratio is not None:
                break

    obj.density_ratio = density_ratio

    print(f"Mass: {new_mass}")
    print(f"Density_ratio: {density_ratio}")

    if obj.density_ratio < 1:

        print('Ablation melting')

        obj.pato.mass_loss = obj.mass - new_mass if new_mass >= 0 else obj.mass
        print('mass loss:', obj.pato.mass_loss)
        obj.material.density *= density_ratio
        obj.mass = new_mass
    
        if (obj.material.density <= 0) or (obj.mass <= 0):
            print("MASS DEMISE OBJ: ", obj.name)
            obj.material.density = 0
            obj.mass = 0
    
        obj.inertia *= density_ratio

    #print('OBJ: ', obj.global_ID, 'DENSITY: ', obj.material.density)
    #print('OBJ: ', obj.global_ID, 'MASS: ', obj.mass)

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/data/constantProperties', 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        lines[48] = 'mass ' + str(obj.mass) + ';\n'
        lines[49] = 'density ' + str(obj.material.density) + ';\n'
        lines[30] = 'rho_sub_n[0]    '+str(obj.material.density)+';\n'
        f.seek(0)
        f.writelines(lines)

def mapping_facetCOG_TITAN_PATO(facet_COG, vtk_COG):

    A = facet_COG
    B = vtk_COG

    tree = KDTree(B)
    
    # Find the nearest point in B for each point in A
    distances, indices = tree.query(A)
        
    # If you need the indices as a list
    mapping = list(indices)

    mapping = np.array(mapping)

    return mapping

def interpolateNearestCOG(facet_COG, input_COG, input_array):

    value = 0;
  
    distance_min = -1;
    indexData = -1;

    xp = facet_COG[0]
    yp = facet_COG[1]
    zp = facet_COG[2]

    for i in range(len(input_COG)):
        x = input_COG[i,0]
        y = input_COG[i,1]
        z = input_COG[i,2]

        dist = np.sqrt(pow(x - xp, 2) + pow(y - yp, 2) + pow(z - zp, 2))
        if (distance_min < 0 or dist < distance_min):
            distance_min = dist;
            indexData = i;
  

    if (indexData >= 0):
        value = input_array[indexData];

    return value    

def retrieve_surface_vtk_data(n_proc, path, time_to_read):

    #n_proc = 1

    filename = [''] * n_proc

    for n in range(n_proc):
        filename[n] = path + "processor" + str(n) + "/VTK/top/" +  "top_" + str(time_to_read) + ".vtk"
        #filename[n] = path + "/VTK/top/" +  "top_" + str(time_to_read) + ".vtk"

    #print('\n PATO solution filenames:', filename)

    #Open the VTK solution files and merge them together into one dataset
    appendFilter = vtkAppendFilter()

    for f in range(n_proc):
        file_data = vtk.vtkPolyDataReader()
        file_data.SetFileName(filename[f])
        file_data.Update()
        file_data = file_data.GetOutput()
        appendFilter.AddInputData(file_data)   
  
    appendFilter.SetMergePoints(True)
    appendFilter.Update()
    vtk_data = appendFilter.GetOutput()        

    writer = vtk.vtkUnstructuredGridWriter()
    pato_output_folder = path + '/Output'
    if not os.path.exists(pato_output_folder): os.mkdir(pato_output_folder)
    time_to_write = str(float(time_to_read)).replace('.','').rjust(5,'0')
    writer.SetFileName(pato_output_folder+'/surface_solution_'+time_to_write+'.vtk')
    writer.SetInputData(vtk_data)
    writer.Write()

    return vtk_data

def retrieve_volume_vtk_data(n_proc, path, time_to_read):

    #n_proc = 1

    filename = [''] * n_proc

    for n in range(n_proc):
        filename[n] = path + "processor" + str(n) + "/VTK/" + "processor" + str(n) + "_" + str(time_to_read) + ".vtk"
        #filename[n] = path + "/VTK/" + "PATO" + "_" + str(time_to_read) + ".vtk"

    #print('\n PATO solution filenames:', filename)

    #Open the VTK solution files and merge them together into one dataset
    appendFilter = vtkAppendFilter()

    for f in range(n_proc):
        file_data = vtk.vtkUnstructuredGridReader()
        file_data.SetFileName(filename[f])
        file_data.Update()
        file_data = file_data.GetOutput()
        appendFilter.AddInputData(file_data)   
  
    appendFilter.SetMergePoints(True)
    appendFilter.Update()
    data = appendFilter.GetOutput()
    
    writer = vtk.vtkUnstructuredGridWriter()
    pato_output_folder = path + '/Output'
    if not os.path.exists(pato_output_folder): os.mkdir(pato_output_folder)
    time_to_write = str(float(time_to_read)).replace('.','').rjust(5,'0')
    writer.SetFileName(pato_output_folder+'/volume_solution_'+time_to_write+'.vtk')
    writer.SetInputData(data)
    writer.Write()

    # extract surface data
    extractSurface=vtk.vtkGeometryFilter()
    extractSurface.SetInputData(data)
    extractSurface.Update()
    vtk_data = extractSurface.GetOutput()



    for file in filename:
        os.remove(file)   

    return vtk_data

def compute_heat_conduction(assembly):

    objects = assembly.objects
    assembly.hf_cond[:] = 0
    for i in range(len(objects)):
        #initialize conductive heat flux of every object
        obj_A = objects[i]
        obj_A.pato.hf_cond[:] = 0
        #loop through each connection of each entry
        for j in range(len(obj_A.connectivity)):
            obj_B = objects[obj_A.connectivity[j]-1]
            compute_heat_conduction_on_surface(obj_A, obj_B)

def identify_object_connections(assembly):

    #create array where each entry correspond to an object I with obj.id
    #each element of the entry will contain the object J obj.id connected to object I
    n_obj = len(assembly.objects)

    #loop through n objects
    obj_id = 1
    for obj in assembly.objects:
        obj.connectivity = np.array([], dtype = int)
        #loop through entries
        for entry in range(len(assembly.connectivity)):
            #if entry contains object
            if obj_id in assembly.connectivity[entry]:
                index = np.where(assembly.connectivity[entry] == obj_id)[0]
                if index == 2: #joint
                    obj.connectivity = np.append(obj.connectivity, assembly.connectivity[entry][0])
                    obj.connectivity = np.append(obj.connectivity, assembly.connectivity[entry][1])
                else: #not joint
                    if assembly.connectivity[entry][2] == 0: #if objects are directly connected
                        #if another object at the left
                        if index == 0:
                            obj.connectivity = np.append(obj.connectivity, assembly.connectivity[entry][1])
                        #if another object at the right !=0
                        if index == 1:
                            obj.connectivity = np.append(obj.connectivity, assembly.connectivity[entry][0])
                    else: #if objects are connected by joint
                        obj.connectivity = np.append(obj.connectivity, assembly.connectivity[entry][2])
        obj_id += 1


def compute_heat_conduction_on_surface(obj_A, obj_B):

    #identify adjacent facets
    #obj_A_adjacent = index of adjacent facets in obj A
    #obj_B_adjacent = index of adjacent facets in obj B
    obj_A_adjacent, obj_B_adjacent = adjacent_facets(obj_A.mesh.facet_COG, obj_B.mesh.facet_COG)

    #pick up k (per facet)
    k_B = obj_B.material.heatConductivity(obj_B.pato.temperature[obj_B_adjacent])
    T_A = obj_A.pato.temperature
    T_B = obj_B.pato.temperature

    L = obj_A.bloom.spacing/2 + obj_B.bloom.spacing/2

    #for the identified facets:
    qcond_A = -k_B*(T_A[obj_A_adjacent]-T_B[obj_B_adjacent])/(L) #qcond_BA

    #append hf_cond cause there will be contribution from different objects
    obj_A.pato.hf_cond[obj_A_adjacent] += qcond_A


def adjacent_facets(facet_COG_A, facet_COG_B):

    COG_A = np.round(facet_COG_A, 5)
    COG_B = np.round(facet_COG_B, 5) 

    # Create dictionaries to store row-index mappings
    dict_A = {tuple(row): index for index, row in enumerate(COG_A)}
    dict_B = {tuple(row): index for index, row in enumerate(COG_B)}
    
    # Find common rows
    common_rows = set(dict_A.keys()) & set(dict_B.keys())
    
    # Create vectors C and D with the indexes
    index_A = [dict_A[row] for row in common_rows]
    index_B = [dict_B[row] for row in common_rows]
    
    return index_A, index_B


