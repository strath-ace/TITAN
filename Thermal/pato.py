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
    setup_PATO_simulation(time, iteration, options, assembly.id)

    run_PATO(options)

    postprocess_PATO_solution(options, assembly)

def setup_PATO_simulation(time, iteration, options, id):
    """
    Sets up the PATO simulation - creates PATO simulation folders and required input files

    Parameters
    ----------
	?????????????????????????
    """

    # If first TITAN iteration, initialize PATO simulation
    if (iteration == 0):

        #write outputFolder + PATO/Allrun file
        write_All_run(options, time - options.dynamics.time_step)

        write_constant_folder(options)

        write_origin_folder(options)

        write_system_folder(options, time - options.dynamics.time_step)

    # If not first TITAN iteration, restart PATO simulation
    #else:
    #    print('to be implemented 0')

def write_All_run(options, time):
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

    end_time = 120#time + options.dynamics.time_step

    with open(options.output_folder + '/PATO/Allrun', 'w') as f:

        f.write('#!/bin/bash \n')
        f.write('cd ${0%/*} || exit 1 \n')
        f.write('. $PATO_DIR/src/applications/utilities/runFunctions/RunFunctions \n')
        f.write('pato_init \n')
        f.write('if [ ! -d 0 ]; then \n')
        f.write('    scp -r origin.0 0 \n')
        f.write('fi \n')
        f.write('cd verification/unstructured_gmsh/ \n')
        #f.write('ln -s ' + path + '/' + options.assembly_path + geo_filename + ' mesh.geo \n')
        f.write('ln -s ' + path + '/' + options.output_folder + '/Volume/mesh.msh \n')
        f.write('cd ../.. \n')
        #f.write('gmsh -3 -format msh2 verification/unstructured_gmsh/mesh.geo verification/unstructured_gmsh/mesh.msh \n')
        f.write('gmshToFoam verification/unstructured_gmsh/mesh.msh \n')
        f.write('mv constant/polyMesh constant/subMat1 \n')   
        f.write('PATOx \n')
        f.write('TIME_STEP='+str(end_time)+' \n')
        f.write('MAT_NAME=subMat1 \n')
        f.write('cp -r "$TIME_STEP/$MAT_NAME"/* "$TIME_STEP" \n')
        f.write('cp system/"$MAT_NAME"/fvSchemes  system/ \n')
        f.write('cp system/"$MAT_NAME"/fvSolution system/ \n')
        f.write('cp -r constant/"$MAT_NAME"/polyMesh/  "$TIME_STEP"/ \n')
        f.write('foamToVTK -time '+str(end_time)+'\n')

    f.close()

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #subprocess.run(['chmod +x ' + options.output_folder +'/PATO/Allrun'], text = True)
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
        f.write('  MaterialPropertiesDirectory "$PATO_DIR/data/Materials/Fourier/FourierTemplate"; \n')
        f.write('}\n')
        f.write('/****************************** END MATERIAL PROPERTIES  ********************/\n')
        f.write('\n')
        f.write('/****************************** PYROLYSIS ************************/\n')
        f.write('Pyrolysis {\n')
        f.write('  PyrolysisType virgin;\n')
        f.write('}\n')
        f.write('/****************************** END PYROLYSIS ************************/\n')

    f.close()

    pass

def write_origin_folder(options):
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
        f.write('internalField   uniform 298;\n')
        f.write('\n')
        f.write('boundaryField {\n')
        f.write('  top\n')
        f.write('  {\n')
        f.write('    type             uniformFixedValue;\n')
        f.write('    uniformValue table\n')
        f.write('    (\n')
        f.write('        (0   298)\n')
        f.write('        (0.1   298)\n')
        f.write('        (0.2   1644)\n')
        f.write('        (120 1644)\n')
        f.write('    );\n')
        f.write('  }\n')
        f.write('}\n')
        f.write('\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')

    f.close()

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
    end_time = 120#time + options.dynamics.time_step
    wrt_interval = 120#end_time - start_time

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
        f.write('deltaT          0.1;\n')
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

    pass


def run_PATO(options):
    """
    Calls the PATO executable and run the simulation

    Parameters
    ----------
	?????????????????????????
    """

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([options.output_folder + '/PATO/Allrun'], text = True)
    #subprocess.run([options.output_folder +'/PATO/Allrun'], text = True)


def postprocess_PATO_solution(options, assembly):
    """
    Postprocesses the PATO output

    Parameters
    ----------
	?????????????????????????
    """        

    #find PATO .vtk solution
    for file in os.listdir(options.output_folder+"PATO/VTK/"):
        if file.endswith(".vtk"):
            filename = options.output_folder+"PATO/VTK/" + file
            print('PATO .vtk solution found.')
            break
        else:
            print('PATO .vtk solution not found.'); exit(0);
    

    #Open the VTK solution file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    #Extract the volume temperature distribution (tetras)
    cell_data = data.GetCellData()
    temperature = cell_data.GetArray('Ta')
    n_cells=data.GetNumberOfCells()
    temperature_cell = [temperature.GetValue(i) for i in range(n_cells)]

    #PATO .vtk to TITAN volume temperature distribution (tetras)
    for i in range(n_cells):
        assembly.mesh.vol_T[i] = temperature_cell[i]

    #Map the tetra temperature to surface mesh
    COG = np.round(assembly.mesh.facet_COG,5).astype(str)
    COG = np.char.add(np.char.add(COG[:,0],COG[:,1]),COG[:,2])

    #Limit Tetras temperature so it does not go negative due to small mass
    assembly.mesh.vol_T[assembly.mesh.vol_T<273] = 273

    #from volume to surface temperature distribution
    for index, COG in enumerate(COG):
        assembly.aerothermo.temperature[index] = assembly.mesh.vol_T[assembly.mesh.index_surf_tetra[str(COG)][0]]

