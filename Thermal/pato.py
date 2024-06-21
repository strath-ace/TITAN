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
import os
import re

def compute_thermal(assembly, time, iteration, options):

    """
    Compute the aerothermodynamic properties using the CFD software

    Parameters
    ----------
    assembly_list: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    """
    setup_PATO_simulation(assembly, time, iteration, options, assembly.id)

    run_PATO(options)

    postprocess_PATO_solution(options, assembly, iteration)

def setup_PATO_simulation(assembly, time, iteration, options, id):
    """
    Sets up the PATO simulation - creates PATO simulation folders and required input files

    Parameters
    ----------
	?????????????????????????
    """
    #Ta_bc = "fixed"
    #Ta = 1644
    Ta_bc = "qconv"
    #Ta_bc = "ablation"
    Tfreestream = assembly.freestream.temperature

    # If first TITAN iteration, initialize PATO simulation
    if (iteration == 0):

        if Ta_bc == "qconv" or Ta_bc == "ablation":
            write_PATO_BC(options, assembly, Ta_bc, time)
        write_All_run(options, time - options.dynamics.time_step, iteration, restart = False)
        write_constant_folder(options)
        write_origin_folder(options, Ta_bc, Tfreestream)
        write_system_folder(options, time - options.dynamics.time_step)
        write_material_properties(options)

    # If not first TITAN iteration, restart PATO simulation
    else:
        if Ta_bc == "qconv" or Ta_bc == "ablation":
            write_PATO_BC(options, assembly, Ta_bc, time)
        write_All_run(options, time - options.dynamics.time_step, iteration, restart = True)
        write_system_folder(options, time - options.dynamics.time_step)

def write_material_properties(options):

    with open(options.output_folder + '/PATO/data/constantProperties', 'w') as f:

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
        f.write('cp_sub_n[0] 1004.55;\n')
        f.write('cp_sub_n[1] 0;\n')
        f.write('cp_sub_n[2] 0;\n')
        f.write('cp_sub_n[3] 0;\n')
        f.write('cp_sub_n[4] 0;\n')
        f.write('\n')
        f.write('// isotropic conductivity  - k - [1 1 -3 -1 0 0 0]\n')
        f.write('k_sub_n[0]  146.04;\n')
        f.write('k_sub_n[1]  0;\n')
        f.write('k_sub_n[2]  0;\n')
        f.write('k_sub_n[3]  0;\n')
        f.write('k_sub_n[4]  0;\n')
        f.write('\n')
        f.write('// density - rho - [1 -3 0 0 0 0 0]\n')
        f.write('rho_sub_n[0]    2813;\n')
        f.write('rho_sub_n[1]    0;\n')
        f.write('rho_sub_n[2]    0;\n')
        f.write('rho_sub_n[3]    0;\n')
        f.write('rho_sub_n[4]    0;\n')
        f.write('\n')
        f.write('// emissivity - e - [0 0 0 0 0 0 0]\n')
        f.write('e_sub_n[0]  0.0;\n')
        f.write('e_sub_n[1]  0;\n')
        f.write('e_sub_n[2]  0;\n')
        f.write('e_sub_n[3]  0;\n')
        f.write('e_sub_n[4]  0;        \n')
    f.close()

def write_All_run(options, time, iteration, restart = False):
    """
    Write the Allrun PATO file

    Generates an executable file to run a PATO simulation according to the state of the object and the user-defined parameters.

    Parameters
    ----------
    options: Options
        Object of class Options
    """
    geo_filename = "pato.geo"
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    end_time = time + options.dynamics.time_step
    start_time = time

    time_step_to_delete = time - options.dynamics.time_step
    iteration_to_delete = int((iteration)*options.dynamics.time_step/options.pato.time_step)

    print('copying BC:', end_time, ' - ', start_time)

    with open(options.output_folder + '/PATO/Allrun', 'w') as f:

        if ((end_time).is_integer()): end_time = int(end_time)
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
        if (not restart):
            f.write('if [ ! -d 0 ]; then \n')
            f.write('    scp -r origin.0 0 \n')
            f.write('fi \n')
            f.write('cd verification/unstructured_gmsh/ \n')
            f.write('ln -s ' + path + '/' + options.output_folder + '/PATO/mesh/mesh.msh \n')
            f.write('cd ../.. \n')
            f.write('gmshToFoam verification/unstructured_gmsh/mesh.msh \n')
            f.write('mv constant/polyMesh constant/subMat1 \n')
            f.write('count=`ls -1 processor* 2>/dev/null | wc -l`\n')
            f.write('if [ $count != 0 ];\n')
            f.write('then\n')
            f.write('    rm -rf processor*\n')
            f.write('fi\n')                                                                                                                                                                                                                             
            f.write('decomposePar -region subMat1\n')
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
        f.write('foamJob -p -s foamToVTK -time '+str(end_time)+'\n')
        #f.write('rm qconv/BC* \n')
        f.write('rm mesh/*su2 \n')
        f.write('rm mesh/*meshb \n')
        for n in range(options.pato.n_cores):
            f.write('rm -rf processor'+str(n)+'/VTK/proc* \n')
            f.write('rm -rf processor'+str(n)+'/'+str(time_step_to_delete)+' \n')
            f.write('rm processor'+str(n)+'/VTK/top/top_'+str(iteration_to_delete)+'.vtk \n')

    f.close()

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.system("chmod +x " + options.output_folder +'/PATO/Allrun' )

    pass

def write_constant_folder(options):
    """
    Write the constant/ PATO folder

    Generates input files defining the 'constant' folder in PATO

    Parameters
    ----------
    options: Options
        Object of class Options
    """

    with open(options.output_folder + '/PATO/constant/regionProperties', 'w') as f:

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

    with open(options.output_folder + '/PATO/constant/subMat1/subMat1Properties', 'w') as f:

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
        f.write('  probingFunctions\n')
        f.write('  (\n')
        f.write('      plotDict\n')
        f.write('      surfacePatchDict\n')
        f.write('  ); // name of sampling/probing dictionaries in "system/subMat1"\n')
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

    pass

def write_origin_folder(options, Ta_bc, Tinf):
    """
    Write the origin.0/ PATO folder

    Generates input files defining the 'origin.0' folder in PATO

    Parameters
    ----------
    options: Options
        Object of class Options
    """

    with open(options.output_folder + '/PATO/origin.0/subMat1/p', 'w') as f:

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


    with open(options.output_folder + '/PATO/origin.0/subMat1/Ta', 'w') as f:

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
        f.write('internalField   uniform '+str(options.Twall)+';\n')
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
            f.write(');\n')
            f.write('p 101325;\n')
            f.write('Tbackground ' + str(Tinf)+';\n')
            f.write('chemistryOn 1;\n')
            f.write('qRad 0;\n')
            f.write('value           uniform 300;\n')       
    
        f.write('  }\n')
        f.write('}\n')
        f.write('\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')

    f.close()

    pass

def write_PATO_BC(options, assembly, Ta_bc, time):

    # write tecplot file with facet_COG coordinates and associated facet convective heating
    if Ta_bc == "qconv":

        for obj in assembly.objects:
            obj.temperature = assembly.aerothermo.temperature[obj.facet_index]
            emissivity_obj = obj.material.emissivity(obj.temperature)
            assembly.emissivity[obj.facet_index] = emissivity_obj 
            assembly.emissivity[obj.facet_index] = np.clip(assembly.emissivity[obj.facet_index], 0, 1)     

        x = np.array([])
        y = np.array([])
        z = np.array([])
        q = np.array([])
        e = np.array([])

        n_data_points = len(assembly.mesh.facet_COG)
        for i in range(n_data_points):
            x = np.append(x, assembly.mesh.facet_COG[i,0])
            y = np.append(y, assembly.mesh.facet_COG[i,1])
            z = np.append(z, assembly.mesh.facet_COG[i,2])
            q = np.append(q, assembly.aerothermo.heatflux[i])
            e = np.append(e, assembly.emissivity[i])

        if ((time).is_integer()): time = int(time)  

        print('BC_', time)

        with open(options.output_folder + 'PATO/qconv/BC_' + str(time), 'w') as f:
            f.write('TITLE     = "vol-for-blayer.fu"\n')
            f.write('VARIABLES = \n')
            f.write('"xw (m)"\n')
            f.write('"yw (m)"\n')
            f.write('"zw (m)"\n')
            f.write('"qConvCFD (W/m^2)"\n')
            f.write('"emissivity (-)"\n')
            f.write('ZONE T="zone 1"\n')
            f.write(' STRANDID=0, SOLUTIONTIME=0\n')
            f.write(' I=' + str(n_data_points) + ', J=1, K=1, ZONETYPE=Ordered\n')
            f.write(' DATAPACKING=BLOCK\n')
            f.write(' DT=(DOUBLE DOUBLE DOUBLE DOUBLE)   \n')
            f.write(np.array2string(x)[1:-1]+' ')
            f.write(np.array2string(y)[1:-1]+' ')
            f.write(np.array2string(z)[1:-1]+' ')
            f.write(np.array2string(q)[1:-1]+' ')
            f.write(np.array2string(e)[1:-1]+' ')

        f.close()

    elif Ta_bc == "ablation":
        print("PATO ablation is not implemented yet."); exit(0);    

    pass

def write_system_folder(options, time):
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

    with open(options.output_folder + '/PATO/system/controlDict', 'w') as f:

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

    with open(options.output_folder + '/PATO/system/subMat1/fvSchemes', 'w') as f:

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

    with open(options.output_folder + '/PATO/system/subMat1/fvSolution', 'w') as f:

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


    f.close()

    with open(options.output_folder + '/PATO/system/subMat1/plotDict', 'w') as f:

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

    with open(options.output_folder + '/PATO/system/subMat1/surfacePatchDict', 'w') as f:

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

    coeff_0 = n_proc/2
    coeff_1 = 2
    coeff_2 = 1

    with open(options.output_folder + '/PATO/system/subMat1/decomposeParDict', 'w') as f:

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


def run_PATO(options):
    """
    Calls the PATO executable and run the simulation

    Parameters
    ----------
	?????????????????????????
    """
    n_proc = options.pato.n_cores

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([options.output_folder + '/PATO/Allrun', str(n_proc)], text = True)


def postprocess_PATO_solution(options, assembly, iteration):
    """
    Postprocesses the PATO output

    Parameters
    ----------
	?????????????????????????
    """ 

    path = options.output_folder+"PATO/VTK/"

    iteration_to_read = int((iteration+1)*options.dynamics.time_step/options.pato.time_step)

    n_proc = options.pato.n_cores

    solution = 'surface'

    if solution == 'surface':
        data = retrieve_surface_vtk_data(n_proc, path, iteration_to_read)
    elif solution == 'volume':
        data = retrieve_volume_vtk_data(n_proc, path, iteration_to_read)

    # extract temperature distribution (tetras)
    cell_data = data.GetCellData()
    temperature = cell_data.GetArray('Ta')
    n_cells=data.GetNumberOfCells()
    temperature_cell = [temperature.GetValue(i) for i in range(n_cells)]

    # sort vtk and TITAN surface mesh cell numbering by checking facet COG

    # get cell COG from vtk
    vtk_cell_centers=vtk.vtkCellCenters()
    vtk_cell_centers.SetInputData(data)
    vtk_cell_centers.Update()
    vtk_cell_centers_data = vtk_cell_centers.GetOutput()
    vtk_COG = vtk_to_numpy(vtk_cell_centers_data.GetPoints().GetData())

    round_number = 3

    vtk_COG = (np.round(vtk_COG, round_number)).tolist()
    TITAN_COG = (np.round(assembly.mesh.facet_COG,round_number)).tolist()   
    
    vtk_COG_sorted   = sorted(vtk_COG)
    TITAN_COG_sorted = sorted(TITAN_COG)
    
    vtk_COG_sorted_index   = [vtk_COG.index(x) for x in vtk_COG_sorted[:]]
    TITAN_COG_sorted_index = [TITAN_COG.index(x) for x in TITAN_COG_sorted[:]]
    
    vtk_COG_sorted_index = np.asarray(vtk_COG_sorted_index, dtype=int)
    TITAN_COG_sorted_index = np.asarray(TITAN_COG_sorted_index, dtype=int)
    
    temperature_cell = np.asarray(temperature_cell, dtype=float)
    
    assembly.aerothermo.temperature = temperature_cell[vtk_COG_sorted_index]    

    temperature_corr = np.zeros(n_cells)

    temperature_corr[TITAN_COG_sorted_index[:]] = assembly.aerothermo.temperature[:]
        
    assembly.aerothermo.temperature = temperature_corr

    # may need to change when implementing > 1 objects
    for obj in assembly.objects:
        obj.temperature = assembly.aerothermo.temperature[obj.facet_index]

def retrieve_surface_vtk_data(n_proc, path, iteration):

    filename = [''] * n_proc

    for n in range(n_proc):
        filename[n] = path + "processor" + str(n) + "_top_" + str(iteration) + ".vtk"

    print('\n PATO solution filenames:', filename)

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

    return vtk_data

def retrieve_volume_vtk_data(n_proc, path, iteration):

    filename = [''] * n_proc

    for n in range(n_proc):
        filename[n] = path + "processor" + str(n) + "_processor" + str(n) + '_' + str(iteration) + ".vtk"

    print('\n PATO solution filenames:', filename)

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

    # extract surface data
    extractSurface=vtk.vtkGeometryFilter()
    extractSurface.SetInputData(data)
    extractSurface.Update()
    vtk_data = extractSurface.GetOutput()    

    return vtk_data