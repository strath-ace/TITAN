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
import shutil
from vtk import *
import glob
import os
import re
import pathlib
import json
import time
from scipy.spatial import KDTree
from Material import material as Material
import pandas as pd
import csv

conda_preamble = ['conda', 'run', '-n', 'pato'] # Better ideally to separate pato env from TITAN?

# Mesh update modes

# Set to true to enable moveDynamicMesh
move_dynamic_mesh = False
# Set to true to 
remesh_volume = True


def _agent_debug_log(location, message, data, hypothesis_id, run_id="pre-fix"):
    # #region agent log
    try:
        lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug-9af45d.log')
        # Ensure JSON-serializable data (pathlib Path -> str)
        safe = {}
        for k, v in data.items():
            if hasattr(v, '__fspath__'):
                safe[k] = str(v)
            elif isinstance(v, dict):
                safe[k] = {kk: (str(vv) if hasattr(vv, '__fspath__') else vv) for kk, vv in v.items()}
            else:
                safe[k] = v
        payload = {
            "sessionId": "9af45d",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": safe,
            "timestamp": int(time.time() * 1000),
        }
        with open(lp, 'a', encoding='utf-8') as _lf:
            _lf.write(json.dumps(payload) + '\n')
    except Exception as _e:
        pass  # Silently skip on any write error
    # #endregion


def _proc_time_state(case_path, proc_idx):
    pdir = pathlib.Path(case_path) / f"processor{proc_idx}"
    if not pdir.exists():
        return {"exists": False, "time_dirs": []}
    tdirs = sorted([d.name for d in pdir.iterdir() if d.is_dir() and d.name.replace(".", "").isdigit()])
    return {
        "exists": True,
        "time_dirs": tdirs,
        "has_0_ta": (pdir / "0" / "subMat1" / "Ta").exists(),
        "has_025_ta": (pdir / "0.25" / "subMat1" / "Ta").exists(),
    }


def _latest_numeric_time_name(base_dir):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return None
    nums = []
    for d in p.iterdir():
        if d.is_dir() and d.name.replace(".", "").isdigit():
            try:
                nums.append((float(d.name), d.name))
            except ValueError:
                continue
    if not nums:
        return None
    nums.sort(key=lambda x: x[0])
    return nums[-1][1]


def _numeric_time_dirs(base_dir):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    out = []
    for d in p.iterdir():
        if d.is_dir() and d.name.replace(".", "").isdigit():
            out.append(d.name)
    try:
        out.sort(key=float)
    except Exception:
        out.sort()
    return out

def _write_mapfields_seed_field(filepath, field_name, field_class, dimensions, uniform_value):
    """Write a minimal OpenFOAM field file with uniform internal field
    and zeroGradient on every boundary.  This is safe for mapFields to parse
    because it uses only standard OF types – no PATO custom BCs."""
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {field_class};\n"
        f"    object      {field_name};\n"
        "}\n\n"
        f"dimensions      {dimensions};\n\n"
        f"internalField   uniform {uniform_value};\n\n"
        "boundaryField\n"
        "{\n"
        '    ".*"\n'
        "    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "}\n"
    )
    filepath.write_text(header)


def _seed_all_fields_from_origin(origin_submat_dir, target_submat_dir):
    """Read every field file in origin.0/subMat1/, extract its class and
    dimensions, and write a clean seed version into the target directory
    using only standard OpenFOAM boundary conditions (zeroGradient via
    wildcard).  This prevents mapFields from choking on PATO-specific BCs
    like HeatFlux or Bprime."""
    import re
    origin = pathlib.Path(origin_submat_dir)
    target = pathlib.Path(target_submat_dir)
    target.mkdir(parents=True, exist_ok=True)
    seeded = []
    for src_file in sorted(origin.iterdir()):
        if not src_file.is_file():
            continue
        text = src_file.read_text(errors="ignore")
        # Extract class
        cls_m = re.search(r"class\s+(vol\w+Field|point\w+Field|surfaceScalarField)", text)
        if not cls_m:
            continue
        field_class = cls_m.group(1)
        # Extract dimensions
        dim_m = re.search(r"dimensions\s+(\[[^\]]+\])", text)
        dims = dim_m.group(1) if dim_m else "[0 0 0 0 0 0 0]"
        # Choose uniform value based on class type
        if "Vector" in field_class:
            uniform_val = "(0 0 0)"
        elif "Tensor" in field_class:
            uniform_val = "(0 0 0 0 0 0 0 0 0)"
        elif "SphericalTensor" in field_class:
            uniform_val = "0"
        else:
            uniform_val = "0"
        _write_mapfields_seed_field(
            target / src_file.name, src_file.name, field_class, dims, uniform_val
        )
        seeded.append(src_file.name)
    return seeded


def _extract_boundary_field(text):
    """Extract the boundaryField { ... } block from an OpenFOAM field file.
    Returns the full block as a string (including 'boundaryField' keyword and
    outer braces), or None if not found."""
    idx = text.find("boundaryField")
    if idx < 0:
        return None
    # Find the opening brace
    brace_start = text.find("{", idx)
    if brace_start < 0:
        return None
    depth = 1
    i = brace_start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[idx:i]


def _graft_boundary_field(mapped_text, original_boundary_block):
    """Replace the boundaryField section in a mapped field file with the
    original PATO boundary conditions, preserving the mapped internalField."""
    idx = mapped_text.find("boundaryField")
    if idx < 0:
        return mapped_text + "\n" + original_boundary_block + "\n"
    brace_start = mapped_text.find("{", idx)
    if brace_start < 0:
        return mapped_text
    depth = 1
    i = brace_start + 1
    while i < len(mapped_text) and depth > 0:
        if mapped_text[i] == "{":
            depth += 1
        elif mapped_text[i] == "}":
            depth -= 1
        i += 1
    return mapped_text[:idx] + original_boundary_block + mapped_text[i:]


def _count_cells_from_polymesh(polymesh_dir):
    """Return the number of volume cells by reading the nCells entry
    from the polyMesh/owner file header note, or by computing max(owner)+1."""
    import re
    owner_file = pathlib.Path(polymesh_dir) / "owner"
    if not owner_file.exists():
        return -1
    text = owner_file.read_text(errors="ignore")

    m = re.search(r"nCells:\s*(\d+)", text)
    if m:
        return int(m.group(1))

    in_list = False
    max_val = -1
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "(":
            in_list = True
            continue
        if stripped == ")":
            break
        if in_list and stripped.isdigit():
            val = int(stripped)
            if val > max_val:
                max_val = val
    return max_val + 1 if max_val >= 0 else -1


def _parse_of_scalar_field(text):
    """Parse an OpenFOAM volScalarField text and return the internalField
    values as a list of floats.  Returns None if uniform or unparseable."""
    in_internal = False
    collecting = False
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if "internalField" in s:
            in_internal = True
            if "nonuniform" in s:
                continue
            return None  # uniform or other
        if in_internal and not collecting:
            if s == "(":
                collecting = True
                continue
            # The line right after internalField nonuniform List<scalar> is the count
            continue
        if collecting:
            if s == ")" or s == ");":
                break
            try:
                vals.append(float(s))
            except ValueError:
                continue
    return vals if vals else None


def compute_thermal(obj, start_time, end_time, iteration, options, hf, Tinf, assembly=None):

    """
    Compute the aerothermodynamic properties using the CFD software

    Parameters
    ----------
    assembly_list: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    """

    obj.pato.q_conv = hf.copy()
    start_time = round(start_time,5)
    end_time = round(end_time, 5)
    time_step = round(end_time - start_time,5)
    if not hasattr(options.pato, 'prev_dt'): options.pato.prev_dt = time_step
    
    print('##### PATO from {} to {} (dt of {})'.format(start_time,
                                                       end_time,
                                                       time_step))
    # #region agent log
    _agent_debug_log(
        "pato.py:compute_thermal",
        "enter compute_thermal",
        {
            "agent_debug_rev": "dbg-2026-02-08-r2",
            "module_file": __file__,
            "has_latest_time_selector": ("_latest_numeric_time_name" in globals()),
            "object_id": obj.global_ID,
            "start_time": start_time,
            "end_time": end_time,
            "time_step": time_step,
        },
        "H1,H2,H3",
    )
    # #endregion

    time_to_postprocess = setup_PATO_simulation(obj, start_time, time_step, iteration, options, hf, Tinf)

    # Quick check: what temperature does processor0/0/subMat1/Ta start at?
    _p0_ta = pathlib.Path(options.output_folder) / f"PATO_{obj.global_ID}" / "processor0" / "0" / "subMat1" / "Ta"
    if _p0_ta.exists():
        _ta_text = _p0_ta.read_text(errors="ignore")
        _vals = _parse_of_scalar_field(_ta_text)
        if _vals is not None and len(_vals) > 0:
            print(f"[PATO start] processor0 Ta: {len(_vals)} values, "
                  f"min={min(_vals):.4f}, max={max(_vals):.4f}, mean={sum(_vals)/len(_vals):.4f}", flush=True)
        else:
            for _line in _ta_text.splitlines():
                if "internalField" in _line.strip():
                    print(f"[PATO start] processor0 Ta: {_line.strip()}", flush=True)
                    break
    else:
        print(f"[PATO start] processor0/0/subMat1/Ta does not exist yet", flush=True)

    run_PATO(options, obj.global_ID)

    # Check end-of-step temperature in processor0 result
    _p0_ta_end = pathlib.Path(options.output_folder) / f"PATO_{obj.global_ID}" / "processor0" / str(time_to_postprocess) / "subMat1" / "Ta"
    if _p0_ta_end.exists():
        _vals_end = _parse_of_scalar_field(_p0_ta_end.read_text(errors="ignore"))
        if _vals_end and len(_vals_end) > 0:
            print(f"[PATO end]   processor0 Ta: {len(_vals_end)} values, "
                  f"min={min(_vals_end):.4f}, max={max(_vals_end):.4f}, mean={sum(_vals_end)/len(_vals_end):.4f}", flush=True)

    postprocess_PATO_solution(options, obj, time_to_postprocess, assembly)
    # Keep root 0/ synchronized with the latest solved state so the following
    # mesh-update decomposePar step seeds processor*/0/ with the current thermal field.
    sync_root_zero_from_latest_parallel(options, obj, time_to_postprocess)
    options.pato.prev_dt = time_step

def setup_PATO_simulation(obj, time, time_step, iteration, options, hf, Tinf):
    """
    Sets up the PATO simulation - creates PATO simulation folders and required input files

    Parameters
    ----------
	?????????????????????????
    """

    write_PATO_BC(options, obj, 0, hf, Tinf)  # PATO-from-0: always BC_0
    # Store convective heat flux for postprocessing
    obj.pato.q_conv = np.asarray(hf, dtype=float)
    time_to_postprocess = write_All_run(options, obj, time, time_step, iteration)
    write_system_folder(options, obj.global_ID, time, time_step)

    return time_to_postprocess

def write_material_properties(options, obj):

    #emissivity_coeffs = obj.material.material_emissivity_polynomial()
    emissivity_coeffs = Material.polynomial_fit(obj.material, obj.material_name, 'emissivity', 1)
    cp_coeffs = Material.polynomial_fit(obj.material, obj.material_name, 'specificHeatCapacity', 4)
    k_coeffs = Material.polynomial_fit(obj.material, obj.material_name, 'heatConductivity', 4)

    T0 = obj.pato.initial_temperature

    eval_T = T0 + (obj.material.meltingTemperature - T0) / 2
    cp = obj.material.specificHeatCapacity(eval_T)
    em = obj.material.emissivity(eval_T)
    tc = obj.material.heatConductivity(eval_T)
    density = obj.material.density

    # #region agent log
    try:
        import json
        lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug-9af45d.log')
        with open(lp, 'a') as _lf:
            _lf.write(json.dumps({"sessionId":"9af45d","location":"pato.py:write_material_properties","message":"thermal conductivity eval","data":{"T0":T0,"Tmelt":obj.material.meltingTemperature,"eval_T":eval_T,"tc":tc,"material":obj.material_name},"hypothesisId":"H1","timestamp":__import__('time').time()*1000}) + '\n')
    except Exception:
        pass
    # #endregion

    object_id = obj.global_ID

    # If initial temperature is larger than melting temperature, we want PATO to skip melting algorithm and instantly
    # go to vaporization algorithm

    if obj.pato.initial_temperature > obj.material.meltingTemperature:
        Tmelt = 1e10
        obj.pato.molten[:] = 1
    else:
        Tmelt = obj.material.meltingTemperature

    with open(options.output_folder + '/PATO_'+str(object_id)+'/data/constantProperties', 'w') as f:

        f.write('/*---------------------------------------------------------------------------*\\n')
        f.write('Material properties for the substructure materials\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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
        f.write('/***        5 coefs - n0 + n1 T + n2 TÂ² + n3 TÂ³ + n4 Tâ´ ***/\n')
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
        f.write('Tmelt ' + str(Tmelt) + ';\n')
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
        f.write('ln -s ' + str(pathlib.Path(options.output_folder).resolve()) + '/PATO_'+str(object_id)+'/mesh/mesh.msh \n')
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

def write_All_run(options, obj, time, time_step, iteration):
    """
    Write the Allrun PATO file.
    PATO-from-0: PATO always runs 0 -> pato_dt.
    At start: copy processor*/pato_dt/ -> processor*/0/ if available.
    At end: delete processor*/0/ (next run repopulates from pato_dt).
    """

    pato_dt = round(time_step, 5)
    if isinstance(pato_dt, float) and pato_dt.is_integer():
        pato_dt = int(pato_dt)

    print(f'[Allrun] PATO-from-0: running 0 -> {pato_dt}')

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/Allrun', 'w') as f:

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
        f.write('$sed_cmd -i "s/numberOfSubdomains \\+[0-9]*;/numberOfSubdomains ""$NPROCESSOR"";/g" system/subMat1/decomposeParDict\n')

        # PATO-from-0: copy previous result (pato_dt/) to 0/ as initial condition
        f.write('# Copy previous result to 0/ (skip on first run / after remesh)\n')
        f.write('if [ -d processor0/' + str(pato_dt) + ' ]; then\n')
        f.write('    for p in processor*; do\n')
        f.write('        cp -r "$p/' + str(pato_dt) + '" "$p/0"\n')
        f.write('        # Reset time metadata to 0 so PATOx solves the full interval\n')
        f.write('        for tf in "$p/0/uniform/time" "$p/0/subMat1/uniform/time"; do\n')
        f.write('            if [ -f "$tf" ]; then\n')
        f.write('                $sed_cmd -i \'s/value\\s\\+[^;]*/value           0/\' "$tf"\n')
        f.write('                $sed_cmd -i \'s/deltaT0\\s\\+[^;]*/deltaT0         0/\' "$tf"\n')
        f.write('            fi\n')
        f.write('        done\n')
        f.write('    done\n')
        f.write('fi\n')

        f.write('cp qconv/BC_0 qconv/BC_' + str(pato_dt) + '\n')
        f.write('mpiexec -np $NPROCESSOR PATOx -parallel \n')
        f.write('TIME_STEP='+str(pato_dt)+' \n')
        f.write('MAT_NAME=subMat1 \n')

        # Flatten subMat1 fields into time folder for foamToVTK
        for n in range(options.pato.n_cores):
            f.write('cd processor' + str(n) + '/\n')
            f.write('cp -r "$TIME_STEP/$MAT_NAME"/* "$TIME_STEP" \n')
            f.write('cp -r constant/"$MAT_NAME"/polyMesh/  "$TIME_STEP"/ \n')
            f.write('cd .. \n')
        f.write('cp system/"$MAT_NAME"/fvSchemes  system/ \n')
        f.write('cp system/"$MAT_NAME"/fvSolution system/ \n')
        f.write('cp system/"$MAT_NAME"/decomposeParDict system/ \n')

        # Patch controlDict so foamToVTK finds the right time
        f.write('$sed_cmd -i "s/startTime *[^;]*;/startTime       $TIME_STEP;/g" system/controlDict\n')
        f.write('foamJob -p -s foamToVTK -time '+str(pato_dt)+' -useTimeName\n')

        # Restore controlDict startTime to 0 for next PATO run
        f.write('$sed_cmd -i "s/startTime *[^;]*;/startTime       0;/g" system/controlDict\n')

        f.write('cp qconv/BC* qconv-bkp/ \n')
        f.write('rm qconv/BC* \n')
        f.write('rm -f mesh/*su2 \n')

        # Cleanup: save restart if needed, then delete processor*/0/
        for n in range(options.pato.n_cores):
            f.write('rm -f processor'+str(n)+'/VTK/top/top_0.vtk \n')
            if options.current_iter%options.save_freq == 0:
                f.write('rm -rf processor'+str(n)+'/restart/* \n')
                f.write('cp -r  processor'+str(n)+'/0/ processor'+str(n)+'/restart/ \n')
                f.write('cp -r  processor'+str(n)+'/'+str(pato_dt)+'/ processor'+str(n)+'/restart/ \n')
                f.write('cp  processor'+str(n)+'/VTK/proce* processor'+str(n)+'/restart/ \n')
            f.write('rm -rf processor'+str(n)+'/0 \n')

    f.close()

    os.system("chmod +x " + options.output_folder +'/PATO_'+str(obj.global_ID)+'/Allrun' )

    return pato_dt

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

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  2.1.x                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  5.0                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  5.0                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            f.write('/****************************** MASS, ENERGY, PYROLYSIS **************************************/\n')
            f.write('MaterialProperties {\n')
            f.write('  MaterialPropertiesType Fourier; \n')
            f.write('  MaterialPropertiesDirectory "$FOAM_CASE/data"; \n')
            f.write('}\n')
            f.write('Mass {\n')
            f.write('  MassType no; // Solve the semi implicit pressure equation\n')
            f.write('  createFields ((p volScalarField) (mDotG volVectorField) (mDotGw volScalarField) (mDotVapor volScalarField) (mDotMelt volScalarField) (molten volScalarField));\n')
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
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 | \n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\\\    /   O peration     | Version:  5.0                                   | \n')
            f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      | \n')
            f.write('|    \\\\/     M anipulation  |                                                 | \n')
            f.write('\\*---------------------------------------------------------------------------*/ \n')
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
    
            f.write('/*---------------------------------------------------------------------------*\\ \n')
            f.write('BoundaryConditions\n')
            f.write('\n')
            f.write('Application\n')
            f.write('    Provides boundary-condition information at the surface, tabulated as a function of time.\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
            f.write('/*\n')
            f.write('t(s)    p_total_w(Pa)   rhoUeCH(kg/mÂ²/s)    h_r(J/kg)   chemistryOn\n')
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

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  5.0                                   |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  5.0                                   |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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
            f.write('value           uniform '+str(obj.pato.initial_temperature)+';\n')
        elif Ta_bc == "ablation":
            f.write('type             Bprime;\n')
            f.write('mixtureMutationBprime '+(mix_file)+';\n')
            f.write('environmentDirectory "$PATO_DIR/data/Environments/RawData/Earth";\n')
            f.write('movingMesh no;\n')
            f.write('mappingType "3D-tecplot";\n')
            f.write('mappingFileName "$FOAM_CASE/qconv/BC";\n')
            f.write('mappingFields\n')
            f.write('(\n')
            f.write('    (qConv "3")\n')
            f.write('    (emissivity "4")\n')
            f.write('    (Tbackground "5")\n')
            f.write('    (molten "6")\n')
            f.write('    (h_r "7")\n')
            f.write('    (rhoeUeCH "8")\n')
            f.write(');\n')
            f.write('chemistryOn 1;\n')
            f.write('p 101325;\n')
            f.write('qRad 0;\n')
            f.write('lambda 0.5;\n')
            f.write('Tedge 300;\n')
            f.write('hconv 0;\n')
            f.write('value uniform '+str(obj.pato.initial_temperature)+';\n')
            f.write('moleFractionGasInMaterial ( ("O"  0.115) ("N" 0) ("C" 0.206) ("H" 0.679));\n')  
        f.write('  }\n')
        f.write('}\n')
        f.write('\n')
        f.write('\n')
        f.write('// ************************************************************************* //\n')

    f.close()

    if Ta_bc == "ablation":

        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/cellMotionU', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  4.x                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 |\n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
            f.write('    \\  /    A nd           | Version:  7\n')
            f.write('     \\/     M anipulation  |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 |\n')
            f.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
            f.write('   \\\\    /   O peration     | Website:  https://openfoam.org\n')
            f.write('    \\\\  /    A nd           | Version:  7\n')
            f.write('     \\\\/     M anipulation  |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 | \n')
            f.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\\\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\\\  /    A nd           | Version:  7 \n')
            f.write('     \\\\/     M anipulation  | \n')
            f.write('\\*---------------------------------------------------------------------------*/ \n')
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
    
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  4.x                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  5.0                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            
             f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
             f.write('| =========                 |                                                 |\n')
             f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
             f.write('|  \\\\    /   O peration     | Version:  5.0                                   |\n')
             f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
             f.write('|    \\\\/     M anipulation  |                                                 |\n')
             f.write('\\*---------------------------------------------------------------------------*/\n')
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
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 | \n')
            f.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\\\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\\\  /    A nd           | Version:  7 \n')
            f.write('     \\\\/     M anipulation  | \n')
            f.write('\\*---------------------------------------------------------------------------*/ \n')
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
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 | \n')
            f.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\\\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\\\  /    A nd           | Version:  7 \n')
            f.write('     \\\\/     M anipulation  | \n')
            f.write('\\*---------------------------------------------------------------------------*/ \n')
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

        with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/origin.0/subMat1/molten', 'w') as f:
            
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 | \n')
            f.write('  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox \n')
            f.write('   \\\\    /   O peration     | Website:  https://openfoam.org \n')
            f.write('    \\\\  /    A nd           | Version:  7 \n')
            f.write('     \\\\/     M anipulation  | \n')
            f.write('\\*---------------------------------------------------------------------------*/ \n')
            f.write('FoamFile { \n')
            f.write('  version     2.0; \n')
            f.write('  format      ascii; \n')
            f.write('  class       volScalarField; \n')
            f.write('  location    "1/porousMat"; \n')
            f.write('  object      molten; \n')
            f.write('} \n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n')
            f.write(' \n')
            f.write('dimensions      [0 0 0 0 0 0 0]; \n')
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

    if isinstance(time, float) and time.is_integer():
        time = int(time)

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
            f.write(' DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)   \n ')
            f.write(np.array2string(x)[1:-1]+' \n')
            f.write(np.array2string(y)[1:-1]+' \n')
            f.write(np.array2string(z)[1:-1]+' \n')
            f.write(np.array2string(conv_heatflux)[1:-1]+' \n')
            f.write(np.array2string(emissivity)[1:-1]+' \n')
            f.write(np.array2string(Tinf)[1:-1]+' \n')

        if options.pato.Ta_bc == "ablation":

            # f.write('TITLE = "vol-for-blayer.fu"\n')
            # f.write('VARIABLES = '
            #         '"xw (m)" "yw (m)" "zw (m)" '
            #         '"qConv (W/m^2)" "emissivity (-)" "Tbackground (K)" '
            #         '"molten (-)" "h_r (J/kg)" "rhoeUeCH (kg/m^2/s)"\n')

            # f.write(f'ZONE T="zone 1", STRANDID=0, SOLUTIONTIME=0, '
            #         f'I={n_data_points}, J=1, K=1, ZONETYPE=Ordered, '
            #         'DATAPACKING=BLOCK, '
            #         'DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)\n')

            # arrays = [
            #     x, y, z, conv_heatflux,
            #     emissivity, Tinf,
            #     obj.pato.molten, obj.pato.h_r, obj.pato.rhoeUeCH
            # ]
            # assert np.all([len(arr)==n_data_points for arr in arrays])
            # for arr in arrays:
            #     f.write(" ".join(str(v) for v in arr) + "\n")

            f.write('TITLE     = "vol-for-blayer.fu"\n')
            f.write('VARIABLES = \n')
            f.write('"xw (m)"\n')
            f.write('"yw (m)"\n')
            f.write('"zw (m)"\n')
            f.write('"qConv (W/m^2)"\n')
            f.write('"emissivity (-)"\n')
            f.write('"Tbackground (K)"\n')
            f.write('"molten (-)"\n')
            f.write('"h_r (J/kg)"\n')
            f.write('"rhoeUeCH (kg/m^2/s)"\n')
            f.write(' ZONE T="zone 1"\n')
            f.write(' STRANDID=0, SOLUTIONTIME=0\n')
            f.write(' I=' + str(n_data_points) + ', J=1, K=1, ZONETYPE=Ordered\n')
            f.write(' DATAPACKING=BLOCK\n')
            f.write(' DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)   \n')
            print(np.nonzero(np.array(np.array(obj.pato.molten, dtype=bool),dtype=int)))
            arrays = [
                x, y, z, conv_heatflux,
                emissivity, Tinf,
                np.array(np.array(obj.pato.molten, dtype=bool),dtype=int), obj.pato.h_r, obj.pato.rhoeUeCH
            ]
            assert np.all([len(arr)==n_data_points for arr in arrays])
            for arr in arrays:
                f.write(" "+"\n ".join(str(v) for v in arr)+'\n')

            # f.write(" "+" ".join(str(v)))
            # f.write(np.array2string(x)[1:-1]+'\n ')
            # f.write(np.array2string(y)[1:-1]+'\n ')
            # f.write(np.array2string(z)[1:-1]+'\n ')
            # f.write(np.array2string(conv_heatflux)[1:-1]+'\n ')
            # f.write(np.array2string(emissivity)[1:-1]+'\n ')
            # f.write(np.array2string(Tinf)[1:-1]+'\n ')
            # f.write(np.array2string(obj.pato.molten)[1:-1]+'\n ')
            # f.write(np.array2string(obj.pato.h_r)[1:-1]+'\n ')
            # f.write(np.array2string(obj.pato.rhoeUeCH)[1:-1]+'\n ')


    f.close()

    pass

def write_system_folder(options, object_id, time, time_step):
    """
    Write the system/ PATO folder.
    PATO-from-0: always startTime=0, endTime=time_step.
    """
    start_time = 0
    end_time = time_step
    wrt_interval = time_step
    pato_time_step = options.pato.time_step#round(time_step*options.pato.time_step,6)
    print('#### PATO STEP {} ####'.format(pato_time_step))

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/controlDict', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  1.5                                   |\n')
        f.write('|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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
        f.write('timePrecision   7;\n')
        f.write('\n')
        f.write('graphFormat     xmgr;\n')
        f.write('\n')
        f.write('runTimeModifiable yes;\n')
        f.write('\n')
        f.write('adjustTimeStep  no; // you can turn it off but its going to be very slow\n')
        f.write('\n')
        f.write('maxCo           10;\n')
        f.write('\n')
        f.write('maxDeltaT   '+str(pato_time_step)+'; // reduce it if the surface temperature starts oscilliating\n')
        f.write('\n')
        f.write('minDeltaT   1e-6;\n')
        f.write('\n')
        f.write('REVlength   1e3;\n')
        f.write('// ************************************************************************* //\n')

    f.close()

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/fvSchemes', 'w') as f:

        f.write(' /*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  1.5                                   |\n')
        f.write('|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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

            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  1.5                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            f.write('PIMPLE\n')
            f.write('{\n')
            f.write('    nOuterCorrectors  1;\n')
            f.write('    nCorrectors       1;\n')
            f.write('}\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')


        else:

            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('| =========                 |                                                 |\n')
            f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            f.write('|  \\\\    /   O peration     | Version:  1.5                                   |\n')
            f.write('|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |\n')
            f.write('|    \\\\/     M anipulation  |                                                 |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
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
            f.write('PIMPLE\n')
            f.write('{\n')
            f.write('    nOuterCorrectors  1;\n')
            f.write('    nCorrectors       1;\n')
            f.write('}\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')

    f.close()

    with open(options.output_folder + '/PATO_'+str(object_id)+'/system/subMat1/plotDict', 'w') as f:

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  dev                                   |\n')
        f.write('|   \\\\  /    A nd           | Web:      http://www.openfoam.org               |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  dev                                   |\n')
        f.write('|   \\\\  /    A nd           | Web:      http://www.openfoam.org               |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
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

        f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
        f.write('| =========                 |                                                 | \n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
        f.write('|  \\\\    /   O peration     | Version:  4.x                                   | \n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      | \n')
        f.write('|    \\\\/     M anipulation  |                                                 | \n')
        f.write('\\*---------------------------------------------------------------------------*/ \n')
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
    write_system_folder(options, object_id, 0, options.dynamics.time_step)

    n_proc = options.pato.n_cores

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pato_test = subprocess.run(conda_preamble+['echo', 'PATO environment working!'])
    if pato_test.returncode>0: raise Exception('Error could not find PATO environment! Check you have a conda env named \'pato\'')
    print('Running PATO initialisation...')

    subprocess.run(conda_preamble+[options.output_folder + '/PATO_'+str(object_id)+'/Allrun_init', str(n_proc)], text = True)
    #subprocess.run([options.output_folder + 'PATO_'+str(object_id)+'/Allrun_init'], text = True)

def run_PATO(options, object_id):
    """
    Calls the PATO executable and run the simulation

    Parameters
    ----------
	?????????????????????????
    """
    n_proc = options.pato.n_cores

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = ' '.join(conda_preamble+[options.output_folder + '/PATO_'+str(object_id)+'/Allrun', str(n_proc)])
    # for arg in conda_preamble+[options.output_folder + '/PATO_'+str(object_id)+'/Allrun', str(n_proc)]: 
    #     cmd+=arg
    #     cmd+=' '
    print(cmd)
    case_path = pathlib.Path(options.output_folder + '/PATO_' + str(object_id)).resolve()
    # #region agent log
    _agent_debug_log(
        "pato.py:run_PATO",
        "pre-Allrun processor state",
        {
            "object_id": object_id,
            "processor0": _proc_time_state(case_path, 0),
            "processor1": _proc_time_state(case_path, 1),
            "controlDict_exists": (case_path / "system" / "controlDict").exists(),
        },
        "H1,H2,H3",
    )
    # #endregion
    _res = subprocess.run(cmd, text=True, shell=True)
    # #region agent log
    _agent_debug_log(
        "pato.py:run_PATO",
        "post-Allrun processor state",
        {
            "object_id": object_id,
            "allrun_returncode": _res.returncode,
            "processor0": _proc_time_state(case_path, 0),
            "processor1": _proc_time_state(case_path, 1),
        },
        "H1,H2,H3",
    )
    # #endregion
    #subprocess.run([options.output_folder + '/PATO_'+str(object_id)+'/Allrun'], text = True)

def sync_root_zero_from_latest_parallel(options, obj, time_to_postprocess):
    """
    Reconstruct latest parallel PATO fields and refresh root 0/subMat1.

    Why:
    meshUpdate runs decomposePar after point motion, and decomposePar seeds
    processor*/0 from root 0/. If root 0/ stays at the initial condition,
    each step effectively restarts from the initial temperature.
    """
    pato_case = pathlib.Path(options.output_folder + f'/PATO_{obj.global_ID}').resolve()
    time_str = str(time_to_postprocess)
    src = pato_case / time_str / "subMat1"
    dst = pato_case / "0" / "subMat1"

    try:
        print(f"[Restart sync] reconstructPar at t={time_str}", flush=True)
        subprocess.run(
            conda_preamble + [
                "reconstructPar", "-region", "subMat1",
                "-time", time_str, "-case", str(pato_case)
            ],
            cwd=str(pato_case), text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[Restart sync] WARNING: reconstructPar failed (rc={e.returncode})", flush=True)
        return
    except Exception as e:
        print(f"[Restart sync] WARNING: reconstructPar error: {e}", flush=True)
        return

    if not src.exists():
        print(f"[Restart sync] WARNING: missing reconstructed source {src}", flush=True)
        return

    try:
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"[Restart sync] Updated root 0/subMat1 from {time_str}/subMat1", flush=True)
    except Exception as e:
        print(f"[Restart sync] WARNING: could not refresh root 0/subMat1: {e}", flush=True)

def write_Ta_from_previous(options, obj, temperature_cell):

    path = options.output_folder + f"/PATO_{obj.global_ID}/origin.0/subMat1/Ta"

    with open(path, "w") as f:

        f.write("FoamFile\n{\n")
        f.write("  version     2.0;\n")
        f.write("  format      ascii;\n")
        f.write("  class       volScalarField;\n")
        f.write("  location    \"0\";\n")
        f.write("  object      Ta;\n")
        f.write("}\n\n")

        f.write("dimensions      [0 0 0 1 0 0 0];\n\n")

        N = len(temperature_cell)

        f.write(f"internalField nonuniform List<scalar>\n{N}\n(\n")

        for T in temperature_cell:
            f.write(f"{float(T)}\n")

        f.write(")\n;\n\n")

        f.write("boundaryField\n{\n")
        f.write("    top { type zeroGradient; }\n")
        f.write("    bottom { type zeroGradient; }\n")
        f.write("}\n")


def perform_PATO_remesh(options, obj, titan, n_cores):
    """
    Full mesh regeneration: reconstruct serial fields from old mesh,
    generate new mesh (GMSH + Bloom + gmshToFoam), map fields from old
    to new mesh via mapFields (nearest-neighbor), then decompose for parallel.

    Uses PATO-from-0: mapped fields are placed in 0/ so the next PATO run
    starts from processor*/0/ as usual.
    """
    from Geometry import gmsh_api as GMSH
    from Aerothermo import bloom
    from .meshUpdate import archive_processor_VTK_to_history

    object_id = obj.global_ID
    pato_case = pathlib.Path(options.output_folder + '/PATO_' + str(object_id)).resolve()
    current_time = titan.time
    time_str = str(current_time)

    print(f"\n{'='*60}", flush=True)
    print(f"[Remesh] Starting full mesh regeneration at t={current_time}", flush=True)
    print(f"{'='*60}", flush=True)
    # #region agent log
    _agent_debug_log(
        "pato.py:perform_PATO_remesh",
        "remesh start case state",
        {
            "agent_debug_rev": "dbg-2026-02-08-r2",
            "module_file": __file__,
            "has_latest_time_selector": ("_latest_numeric_time_name" in globals()),
            "object_id": object_id,
            "time_str": time_str,
            "root_has_time_dir": (pato_case / time_str).exists(),
            "root_has_zero_dir": (pato_case / "0").exists(),
            "processor0": _proc_time_state(pato_case, 0),
            "processor1": _proc_time_state(pato_case, 1),
        },
        "H2,H4,H5",
    )
    # #endregion

    # 1. Archive VTK files before touching processor dirs
    try:
        archive_processor_VTK_to_history(pato_case, time_value=current_time)
    except Exception as e:
        print(f"[Remesh] VTK archiving skipped: {e}", flush=True)

    # 2. Reconstruct parallel fields to serial
    # Use the latest available processor time instead of current simulation time.
    # In PATO-from-0 mode, processors often only store [0.25] even at larger titan.time.
    source_time = _latest_numeric_time_name(pato_case / "processor0")
    if source_time is None:
        source_time = time_str
    # #region agent log
    _agent_debug_log(
        "pato.py:perform_PATO_remesh",
        "selected remesh source time",
        {
            "time_str": time_str,
            "selected_source_time": source_time,
            "processor0": _proc_time_state(pato_case, 0),
        },
        "H2,H4,H5",
    )
    # #endregion
    print("[Remesh] Reconstructing parallel fields to serial...", flush=True)
    try:
        subprocess.run(
            conda_preamble + [
                "reconstructPar", "-region", "subMat1",
                "-time", source_time, "-case", str(pato_case)
            ],
            cwd=str(pato_case), text=True, check=True,
        )
        # #region agent log
        _agent_debug_log(
            "pato.py:perform_PATO_remesh",
            "post-reconstruct source availability",
            {
                "source_time": source_time,
                "root_time_dirs": _numeric_time_dirs(pato_case),
                "root_source_has_submat": (pato_case / source_time / "subMat1").exists(),
                "root_source_has_ta": (pato_case / source_time / "subMat1" / "Ta").exists(),
            },
            "H10,H11,H12",
        )
        # #endregion
    except subprocess.CalledProcessError as e:
        print(f"[Remesh] WARNING: reconstructPar failed (rc={e.returncode}), "
              "attempting remesh anyway", flush=True)

    # 3. Create temporary source case with old mesh + reconstructed fields
    source_case = pato_case / "_remesh_source"
    if source_case.exists():
        shutil.rmtree(source_case)
    source_case.mkdir()
    (source_case / "constant").mkdir()
    (source_case / "system").mkdir()

    old_poly = pato_case / "constant" / "subMat1"
    if old_poly.exists():
        shutil.copytree(str(old_poly), str(source_case / "constant" / "subMat1"))

    serial_time_dir = pato_case / source_time
    if serial_time_dir.exists():
        shutil.copytree(str(serial_time_dir), str(source_case / source_time))
    # #region agent log
    _agent_debug_log(
        "pato.py:perform_PATO_remesh",
        "source case prepared for mapFields",
        {
            "source_time": source_time,
            "source_case_time_dirs": _numeric_time_dirs(source_case),
            "source_case_has_submat": (source_case / source_time / "subMat1").exists(),
            "source_case_has_ta": (source_case / source_time / "subMat1" / "Ta").exists(),
        },
        "H10,H11,H12",
    )
    # #endregion

    sys_src = pato_case / "system"
    if sys_src.exists():
        shutil.copytree(str(sys_src), str(source_case / "system"), dirs_exist_ok=True)

    # 4. Generate new mesh from current (recessed) obj.mesh
    print("[Remesh] Generating new GMSH + Bloom mesh...", flush=True)
    GMSH.generate_PATO_domain(obj, output_folder=options.output_folder)
    bloom.generate_PATO_mesh(options, object_id, bloom=obj.bloom)

    # 5. Run gmshToFoam to create new polyMesh
    mesh_msh = pato_case / "mesh" / "mesh.msh"
    verif_dir = pato_case / "verification" / "unstructured_gmsh"
    verif_dir.mkdir(parents=True, exist_ok=True)
    link_target = verif_dir / "mesh.msh"
    if link_target.exists() or link_target.is_symlink():
        link_target.unlink()
    try:
        link_target.symlink_to(mesh_msh)
    except OSError:
        shutil.copy2(str(mesh_msh), str(link_target))

    subprocess.run(
        conda_preamble + [
            "gmshToFoam", str(link_target), "-case", str(pato_case)
        ],
        cwd=str(pato_case), text=True, check=True,
    )
    new_poly_src = pato_case / "constant" / "polyMesh"
    new_poly_dst = pato_case / "constant" / "subMat1" / "polyMesh"
    if new_poly_src.exists():
        if new_poly_dst.exists():
            shutil.rmtree(new_poly_dst)
        shutil.move(str(new_poly_src), str(new_poly_dst))
    print("[Remesh] gmshToFoam complete, new polyMesh installed", flush=True)

    # 6. Remove old processor dirs and stale time dirs (incompatible cell counts)
    _preserve = {"constant", "system", "mesh", "qconv", "qconv-bkp", "VTK_history",
                 "verification", "origin.0", "data", "Output", "mesh_evolution",
                 "outputs_test", "recession_debug", "_remesh_source"}
    for _d in list(pato_case.iterdir()):
        if _d.is_dir() and _d.name not in _preserve and not _d.name.startswith("_"):
            try:
                _n = _d.name
                if _n.replace(".", "").replace("-", "").isdigit() or _n.startswith("processor"):
                    shutil.rmtree(_d)
                    print(f"[Remesh] Removed stale {_n}/", flush=True)
            except OSError:
                pass

    # 7. Patch controlDict startTime=0 for PATO-from-0 (mapFields writes to 0)
    ctrl_path = pato_case / "system" / "controlDict"
    try:
        ctrl_txt = ctrl_path.read_text(errors="ignore")
        ctrl_txt = re.sub(r"startTime\s+[^;]+;", "startTime       0;", ctrl_txt)
        ctrl_path.write_text(ctrl_txt)
        print("[Remesh] Patched controlDict: startTime = 0", flush=True)
    except Exception as e:
        print(f"[Remesh] WARNING: could not patch controlDict: {e}", flush=True)

    # 8. Write mapFieldsDict (inconsistent mapping: explicit patch map)
    map_fields_dict = pato_case / "system" / "mapFieldsDict"
    with open(map_fields_dict, "w", encoding="utf-8") as _mf:
        _mf.write("FoamFile\n{\n")
        _mf.write("    version     2.0;\n")
        _mf.write("    format      ascii;\n")
        _mf.write("    class       dictionary;\n")
        _mf.write("    object      mapFieldsDict;\n")
        _mf.write("}\n\n")
        _mf.write("patchMap       ( top top );\n")
        _mf.write("cuttingPatches ();\n")

    # 8b. Save original PATO boundary conditions from origin.0/subMat1/ BEFORE
    #     overwriting with clean seeds.  These will be grafted back after mapFields.
    seed_src = pato_case / "origin.0" / "subMat1"
    original_bcs = {}  # field_name -> boundaryField block string
    if seed_src.exists():
        for _f in seed_src.iterdir():
            if _f.is_file():
                _bc = _extract_boundary_field(_f.read_text(errors="ignore"))
                if _bc:
                    original_bcs[_f.name] = _bc
    print(f"[Remesh] Saved original BCs for: {sorted(original_bcs.keys())}", flush=True)

    # Write clean seed fields (uniform internalField + zeroGradient BCs) that
    # mapFields can parse without PATO libraries.
    seed_dst = pato_case / "0" / "subMat1"
    seeded_fields = _seed_all_fields_from_origin(seed_src, seed_dst)
    print(f"[Remesh] Wrote {len(seeded_fields)} clean seed fields in 0/subMat1/: {seeded_fields}", flush=True)

    # Count cells via checkMesh (owner file first integer = nFaces, not nCells)
    src_n_cells = _count_cells_from_polymesh(source_case / "constant" / "subMat1" / "polyMesh")
    tgt_n_cells = _count_cells_from_polymesh(pato_case / "constant" / "subMat1" / "polyMesh")
    print(f"[Remesh] Source mesh (_remesh_source): {src_n_cells} cells", flush=True)
    print(f"[Remesh] Target mesh (new polyMesh):   {tgt_n_cells} cells", flush=True)

    # List source fields
    src_field_dir = source_case / source_time / "subMat1"
    src_fields = []
    if src_field_dir.exists():
        src_fields = sorted([f.name for f in src_field_dir.iterdir()
                             if f.is_file() and f.name != "uniform"])
    print(f"[Remesh] Source fields in {source_case.name}/{source_time}/subMat1/: {src_fields}", flush=True)

    # List target fields before mapping
    tgt_field_dir = pato_case / "0" / "subMat1"
    tgt_fields_before = []
    if tgt_field_dir.exists():
        tgt_fields_before = sorted([f.name for f in tgt_field_dir.iterdir()
                                    if f.is_file() and f.name != "uniform"])
    print(f"[Remesh] Target fields in 0/subMat1/ before mapFields: {tgt_fields_before}", flush=True)

    # 9. Map fields from old mesh to new mesh
    print(f"[Remesh] Mapping fields: {source_case} (t={source_time}) --> {pato_case} (t=0)", flush=True)
    try:
        # #region agent log
        _agent_debug_log(
            "pato.py:perform_PATO_remesh",
            "pre-mapFields target readiness",
            {
                "target_time_dirs_before": _numeric_time_dirs(pato_case),
                "target_has_zero_submat": seed_dst.exists(),
                "target_has_zero_ta": (seed_dst / "Ta").exists() if seed_dst.exists() else False,
                "source_case_has_ta": (source_case / source_time / "subMat1" / "Ta").exists(),
                "src_n_cells": src_n_cells,
                "tgt_n_cells": tgt_n_cells,
                "src_fields": src_fields,
                "tgt_fields_before": tgt_fields_before,
            },
            "H10,H12,H13,H15",
        )
        # #endregion
        _mf = subprocess.run(
            conda_preamble + [
                "mapFields", str(source_case),
                "-sourceTime", source_time,
                "-targetRegion", "subMat1",
                "-sourceRegion", "subMat1",
                "-mapMethod", "mapNearest",
                "-case", str(pato_case),
            ],
            cwd=str(pato_case), text=True, check=True,
            capture_output=True,
        )

        print("[Remesh] mapFields complete (rc=0)", flush=True)

        # mapFields may create source_time dir; normalize to 0.
        mapped_dir = pato_case / source_time
        zero_dir = pato_case / "0"
        if mapped_dir.exists() and mapped_dir.is_dir() and mapped_dir != zero_dir:
            if zero_dir.exists():
                shutil.rmtree(zero_dir)
            shutil.move(str(mapped_dir), str(zero_dir))
            print(f"[Remesh] Renamed {source_time}/ -> 0/", flush=True)

        # Verify mapped Ta
        mapped_field_dir = pato_case / "0" / "subMat1"
        mapped_ta = mapped_field_dir / "Ta"
        ta_n_values = 0
        if mapped_ta.exists():
            ta_content = mapped_ta.read_text(errors="ignore")
            in_internal = False
            for line in ta_content.splitlines():
                stripped = line.strip()
                if "internalField" in stripped:
                    in_internal = True
                    if "nonuniform" in stripped:
                        continue
                    elif "uniform" in stripped:
                        ta_n_values = -1
                        break
                    continue
                if in_internal and stripped.isdigit():
                    ta_n_values = int(stripped)
                    break
        if ta_n_values > 0:
            print(f"[Remesh] Mapped Ta: {ta_n_values} values (target mesh: {tgt_n_cells} cells) "
                  f"{'MATCH' if ta_n_values == tgt_n_cells else 'MISMATCH!'}", flush=True)
        elif ta_n_values == -1:
            print(f"[Remesh] WARNING: Mapped Ta is still 'uniform' — mapping may not have worked", flush=True)
        else:
            print(f"[Remesh] WARNING: Could not parse mapped Ta internalField", flush=True)

        # 9b. Graft original PATO boundary conditions back onto the mapped fields.
        #     mapFields preserved the seed's zeroGradient BCs, but PATO needs
        #     HeatFlux/Bprime on the top patch to apply heat flux.
        grafted = []
        if mapped_field_dir.exists():
            for _fname, _bc_block in original_bcs.items():
                _fpath = mapped_field_dir / _fname
                if _fpath.exists():
                    _mapped_text = _fpath.read_text(errors="ignore")
                    _fpath.write_text(_graft_boundary_field(_mapped_text, _bc_block))
                    grafted.append(_fname)
        print(f"[Remesh] Restored original PATO BCs on: {grafted}", flush=True)

        # Update origin.0/subMat1/ with the mapped fields (now with correct BCs)
        origin_submat = pato_case / "origin.0" / "subMat1"
        if mapped_field_dir.exists():
            if origin_submat.exists():
                shutil.rmtree(origin_submat)
            shutil.copytree(str(mapped_field_dir), str(origin_submat))
            print(f"[Remesh] Updated origin.0/subMat1/ with mapped fields + restored BCs", flush=True)
        else:
            print(f"[Remesh] WARNING: mapped 0/subMat1/ not found, origin.0 NOT updated", flush=True)

    except subprocess.CalledProcessError as e:
        print(f"[Remesh] WARNING: mapFields failed (rc={e.returncode})", flush=True)
        print(f"[Remesh] mapFields stdout:\n{e.stdout[-1500:] if getattr(e, 'stdout', None) else '(empty)'}", flush=True)
        print(f"[Remesh] mapFields stderr:\n{e.stderr[-1500:] if getattr(e, 'stderr', None) else '(empty)'}", flush=True)
        # #region agent log
        _agent_debug_log(
            "pato.py:perform_PATO_remesh",
            "mapFields failed",
            {
                "time_str": time_str,
                "source_time": source_time,
                "returncode": e.returncode,
                "stdout_tail": (e.stdout[-800:] if getattr(e, "stdout", None) else ""),
                "stderr_tail": (e.stderr[-800:] if getattr(e, "stderr", None) else ""),
                "target_time_dirs_after_failure": _numeric_time_dirs(pato_case),
                "zero_has_ta_after_failure": (pato_case / "0" / "subMat1" / "Ta").exists(),
            },
            "H4,H5,H15",
        )
        # #endregion
        raise RuntimeError("Remesh aborted: mapFields failed; skipping decomposePar to avoid corrupt restart state.")

    # 10. Decompose for parallel (creates processor*/0/ with mapped fields)
    print("[Remesh] Running decomposePar...", flush=True)
    for proc_dir in sorted(pato_case.glob("processor*")):
        shutil.rmtree(proc_dir)
    subprocess.run(
        conda_preamble + [
            "decomposePar", "-region", "subMat1", "-case", str(pato_case)
        ],
        cwd=str(pato_case), text=True, check=True,
    )
    print("[Remesh] decomposePar complete", flush=True)
    # #region agent log
    _agent_debug_log(
        "pato.py:perform_PATO_remesh",
        "post-decomposePar processor state",
        {
            "time_str": time_str,
            "processor0": _proc_time_state(pato_case, 0),
            "processor1": _proc_time_state(pato_case, 1),
        },
        "H2,H3,H5",
    )
    # #endregion

    # 11. Cleanup temp source case
    if source_case.exists():
        shutil.rmtree(source_case)

    # Mark that remesh happened (cumulative recession resets)
    if hasattr(options.pato, '_cumulative_recession'):
        options.pato._cumulative_recession = 0.0

    # Signal to caller that remesh happened so write_Ta_from_previous is skipped
    # (origin.0 already has the correctly mapped fields for the new mesh)
    options.pato._remesh_just_happened = True

    print(f"[Remesh] Full mesh regeneration complete at t={current_time}", flush=True)
    print(f"{'='*60}\n", flush=True)


def _build_surface_1ring(nodes, facets):
    """Build 1-ring neighbor lists and per-node mean edge length from a triangle mesh.

    Parameters
    ----------
    nodes : np.ndarray (N, 3)
    facets : np.ndarray (F, 3)  vertex indices per triangle

    Returns
    -------
    neighbors : list[np.ndarray]  neighbors[i] = array of node indices sharing an edge with i
    mean_edge_length : np.ndarray (N,)  mean length of edges incident to each node
    """
    from collections import defaultdict

    N = nodes.shape[0]
    adj = defaultdict(set)
    for i, j, k in facets:
        adj[i].update((j, k))
        adj[j].update((i, k))
        adj[k].update((i, j))

    neighbors = [np.array(sorted(adj[i]), dtype=int) if i in adj else np.array([], dtype=int) for i in range(N)]

    mean_edge_length = np.zeros(N)
    for i in range(N):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            continue
        lengths = np.linalg.norm(nodes[nbrs] - nodes[i], axis=1)
        mean_edge_length[i] = np.mean(lengths)

    # Fallback for isolated nodes: use global mean edge length
    valid = mean_edge_length > 0
    if np.any(valid) and not np.all(valid):
        global_mean = np.mean(mean_edge_length[valid])
        mean_edge_length[~valid] = global_mean

    return neighbors, mean_edge_length


# Guassian-kernel smoothing function for node displacements

# Each node only smoothed based on the 'ring' around it - the 1st set of nodes it contacts
# Sigma factor controls the spatial radius of smoothing (based on local edge length). Increase for stronger smoothing
# Blend_alpha blends the smoothed displacement with original.
# e.g. a blend_alpha of 0.2 means the node displacement cannot change more than 20% from the unsmoothed one

def _smooth_nodal_displacement_1ring(vertex_disp, nodes, neighbors, mean_edge_length,
                                     sigma_factor=1.5, blend_alpha=0.3):

    N = vertex_disp.shape[0]
    smoothed = np.zeros_like(vertex_disp)

    for i in range(N):
        nbrs = neighbors[i]
        sigma_i = sigma_factor * mean_edge_length[i]
        if len(nbrs) == 0 or sigma_i < 1e-30:
            smoothed[i] = vertex_disp[i]
            continue

        dists = np.linalg.norm(nodes[nbrs] - nodes[i], axis=1)
        weights = np.exp(-dists**2 / (2.0 * sigma_i**2))

        # Include node i itself with weight 1
        all_weights = np.empty(len(nbrs) + 1)
        all_weights[0] = 1.0
        all_weights[1:] = weights

        all_disps = np.empty((len(nbrs) + 1, 3))
        all_disps[0] = vertex_disp[i]
        all_disps[1:] = vertex_disp[nbrs]

        w_sum = np.sum(all_weights) + 1e-20
        smoothed[i] = np.sum(all_weights[:, None] * all_disps, axis=0) / w_sum

    vertex_disp_final = (1.0 - blend_alpha) * vertex_disp + blend_alpha * smoothed
    return vertex_disp_final

# Facets within this angle of each other will be clustered into one face
FACE_THETA_DEG = 20


def _nodal_displacement_from_facets_clustered(nodes, facets, unit_normals, facet_disp, facet_area, face_theta_deg):
    """
    Compute nodal displacement by clustering incident facets by normal direction,
    then summing area-weighted cluster displacements. Avoids edge/corner pinching.
    """
    N_nodes = nodes.shape[0]
    vertex_disp = np.zeros((N_nodes, 3))

    # Build node -> incident facet indices
    node_facets = [[] for _ in range(N_nodes)]
    for f_id, (i, j, k) in enumerate(facets):
        node_facets[i].append(f_id)
        node_facets[j].append(f_id)
        node_facets[k].append(f_id)

    face_theta_rad = np.deg2rad(face_theta_deg)

    for i in range(N_nodes):
        incident = node_facets[i]
        if not incident:
            continue
        incident = np.array(incident)
        n_f = unit_normals[incident]
        d_f = facet_disp[incident]
        A_f = facet_area[incident]

        # Cluster: two facets in same cluster if angle(n_i, n_j) < face_theta_deg
        used = np.zeros(len(incident), dtype=bool)
        clusters = []
        while not np.all(used):
            seed = int(np.where(~used)[0][0])
            cluster = [seed]
            used[seed] = True
            added = True
            while added:
                added = False
                for j in range(len(incident)):
                    if used[j]:
                        continue
                    for c in cluster:
                        dot = np.clip(float(np.dot(n_f[j], n_f[c])), -1.0, 1.0)
                        if np.arccos(dot) < face_theta_rad:
                            cluster.append(j)
                            used[j] = True
                            added = True
                            break
            clusters.append(cluster)

        # Per cluster: area-weighted mean displacement; then sum over clusters
        for cluster in clusters:
            idx = np.array(cluster)
            a_sum = A_f[idx].sum()
            if a_sum <= 0:
                continue
            d_k = np.sum(d_f[idx] * A_f[idx, None], axis=0) / a_sum
            vertex_disp[i] += d_k

    return vertex_disp


def postprocess_PATO_solution(options, obj, time_to_read, assembly=None):
    """
    Postprocesses the PATO output

    Parameters
    ----------
	?????????????????????????
    """
    # Defaults for simulation summary CSV (overwritten when ablation outputs are computed)
    obj.pato.total_mdot_kg_s = 0.0
    obj.pato.max_recession_mm_s = 0.0

    #if options.pato.Ta_bc == 'ablation': postprocess_mass_inertia(obj, options, time_to_read)

    path = options.output_folder+"/PATO_"+str(obj.global_ID)+"/"

    #iteration_to_read = int(round((iteration+1)*options.dynamics.time_step/options.pato.time_step))

    n_proc = options.pato.n_cores

    # Wall temperature comes from the true boundary patch output
    surface_data = retrieve_surface_vtk_data(n_proc, path, time_to_read)
    # Subsurface sampling comes from the full volume output
    volume_data, _ = retrieve_volume_vtk_data(n_proc, path, time_to_read)

    # ---- Surface-only mapping for Tw / boundary fields ----
    surface_cell_data = surface_data.GetCellData()
    n_surface_cells = surface_data.GetNumberOfCells()

    temperature_surface = surface_cell_data.GetArray('Ta')
    temperature_surface = [temperature_surface.GetValue(i) for i in range(n_surface_cells)]
    temperature_surface = np.array(temperature_surface)
    obj.pato.Ta_prev = temperature_surface.copy()
    #write_Ta_from_previous(options, obj, temperature_surface)

    if options.pato.Ta_bc == "ablation":
        mDotVapor_surface = surface_cell_data.GetArray('mDotVapor')
        mDotVapor_cell = [mDotVapor_surface.GetValue(i) for i in range(n_surface_cells)]
        mDotVapor_cell = np.array(mDotVapor_cell)
        mDotMelt_surface = surface_cell_data.GetArray('mDotMelt')
        mDotMelt_cell = [mDotMelt_surface.GetValue(i) for i in range(n_surface_cells)]
        mDotMelt_cell = np.array(mDotMelt_cell)

    vtk_cell_centers = vtk.vtkCellCenters()
    vtk_cell_centers.SetInputData(surface_data)
    vtk_cell_centers.Update()
    vtk_cell_centers_data = vtk_cell_centers.GetOutput()
    points = vtk_cell_centers_data.GetPoints() if vtk_cell_centers_data else None
    if points is None:
        raise RuntimeError(
            "Surface VTK cell centers returned no points. VTK data may be empty or corrupted."
        )
    surface_COG = vtk_to_numpy(points.GetData())

    mapping = mapping_facetCOG_TITAN_PATO(obj.mesh.facet_COG, surface_COG)

    # retrieve wall solution from boundary patch
    obj.pato.temperature = temperature_surface[mapping]
    obj.temperature = obj.pato.temperature
    obj.pato.Tw_pato = obj.pato.temperature.copy()

    if options.pato.Ta_bc == "ablation":
        obj.pato.mDotVapor = mDotVapor_cell[mapping]
        obj.pato.mDotMelt = mDotMelt_cell[mapping]
        obj.pato.molten[obj.temperature >= obj.material.meltingTemperature] = 1

    # ---- Volume-only sampling for T_int / dx_s ----
    volume_cell_data = volume_data.GetCellData()
    n_volume_cells = volume_data.GetNumberOfCells()
    temperature_volume = volume_cell_data.GetArray('Ta')
    temperature_volume = [temperature_volume.GetValue(i) for i in range(n_volume_cells)]
    temperature_volume = np.array(temperature_volume)

    vtk_cell_centers = vtk.vtkCellCenters()
    vtk_cell_centers.SetInputData(volume_data)
    vtk_cell_centers.Update()
    vtk_cell_centers_data = vtk_cell_centers.GetOutput()
    points = vtk_cell_centers_data.GetPoints() if vtk_cell_centers_data else None
    if points is None:
        raise RuntimeError(
            "Volume VTK cell centers returned no points. VTK data may be empty or corrupted."
        )
    volume_COG = vtk_to_numpy(points.GetData())

    # Compute interior temperature T_int and distance dx_s from volume cells
    facet_COG = np.asarray(obj.mesh.facet_COG)
    tree = KDTree(volume_COG)

    # Query 2 nearest grid cells → second nearest is solid interior
    dists, idxs = tree.query(facet_COG, k=2)
    if dists.ndim == 1:   # ensure 2D arrays
        dists = dists[:, None]
        idxs  = idxs[:, None]

    interior_idx = idxs[:, 1]
    dx_s = dists[:, 1]
    T_int = temperature_volume[interior_idx]

    # Output path
    output_dir = "/home/navraj/TITAN/simulation/PATO_0/outputs_test"
    output_dir = os.path.join(options.output_folder, f"PATO_{obj.global_ID}", "outputs_test")
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"outputs_{time_to_read}.csv")

    # Known variables
    Tw   = obj.pato.temperature
    COG  = obj.mesh.facet_COG
    N    = len(Tw)

    rho_s = obj.material.density
    Tmelt = obj.material.meltingTemperature
    Lf    = obj.material.meltingHeat
    k_fun = getattr(obj.material, "heatConductivity", None)

    # q_conv from PATO
    q_conv_arr = np.asarray(getattr(obj.pato, "q_conv", np.zeros(N)), float)
    q_cond_arr = np.zeros(N, dtype=float)
    q_excess_arr = np.zeros(N, dtype=float)
    k_mid_arr = np.zeros(N, dtype=float)
    # Write CSV - for testing
    with open(fname, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow([
            "facet_id","Tw","x","y","z","rho_s","Tmelt","Lf","k_Tw",
            "q_conv","T_int","dx_s","k_mid","q_cond","q_excess","mdot","v_n","delta_t","d_n"
        ])
        
        mdot = np.zeros(N, dtype=float)
        v_n = np.zeros(N, dtype=float)
        d_n = np.zeros(N, dtype=float)
        delta_t = options.pato.time_step
        normals = obj.pato.facet_normal

        for i in range(N):
            k_Tw = float(k_fun(Tw[i])) if callable(k_fun) else 0.0

            # conductive heat flux
            if callable(k_fun) and dx_s[i] > 0:
                Tmid = 0.5 * (Tw[i] + T_int[i])
                Tq = min(Tmid, Tmelt)
                k_mid = float(k_fun(Tq))
                q_cond = -k_mid * (T_int[i]-Tw[i]) / dx_s[i]
            else:
                k_mid = 0.0
                q_cond = 0.0

            q_conv   = q_conv_arr[i]
            q_excess = q_conv - q_cond
            k_mid_arr[i] = k_mid
            q_cond_arr[i] = q_cond
            q_excess_arr[i] = q_excess

            # melting model
            if (Tmelt and Lf and rho_s) and (Tw[i] >= Tmelt) and (q_excess > 0):
                mdot[i] = q_excess / Lf
                v_n[i]  = mdot[i] / rho_s
                d_n[i]  = v_n[i] * delta_t
            else:
                mdot[i] = v_n[i] = 0

            w.writerow([
                i, float(Tw[i]), *COG[i], rho_s, Tmelt, Lf, k_Tw,
                q_conv, float(T_int[i]), float(dx_s[i]), k_mid,
                q_cond, q_excess, mdot[i], v_n[i], delta_t, d_n[i]
            ])

    # Subsurface / T_int vs Tw diagnostics
    print("[subsurface] count(T_int > Tw) =", int(np.sum(T_int > Tw)))
    print("[subsurface] count(T_int > Tw + 5K) =", int(np.sum(T_int > Tw + 5.0)))
    print("[subsurface] count(T_int > Tw + 20K) =", int(np.sum(T_int > Tw + 20.0)))
    worst = np.argsort(T_int - Tw)[-10:]
    print("[subsurface] worst 10 facets by (T_int - Tw):")
    for i in worst:
        print(
            f"  i={i}, Tw={Tw[i]:.3f}, T_int={T_int[i]:.3f}, dT={T_int[i]-Tw[i]:.3f}, "
            f"dx_s={dx_s[i]:.6e}, k_mid={k_mid_arr[i]:.6e}, "
            f"q_conv={q_conv_arr[i]:.6e}, q_cond={q_cond_arr[i]:.6e}, "
            f"q_excess={q_excess_arr[i]:.6e}, interior_idx={int(interior_idx[i])}"
        )

    # store arrays
    obj.pato.q_conv_field = q_conv_arr
    obj.pato.mdot_field   = mdot
    obj.pato.v_n_field    = v_n

    # Total mass loss rate (kg/s) = integral of mdot over surface area
    facet_areas = np.linalg.norm(normals, axis=1)
    total_mdot_kg_s = np.sum(mdot * facet_areas)
    obj.pato.total_mdot_kg_s = total_mdot_kg_s
    obj.pato.max_recession_mm_s = float(np.max(v_n)) * 1000.0  # m/s -> mm/s
    print('***** Total mass loss rate = {:.6e} kg/s *****'.format(total_mdot_kg_s))

    if assembly is not None and total_mdot_kg_s != 0:
        dt_titan = options.dynamics.time_step
        dm = -total_mdot_kg_s * dt_titan
        new_mass = obj.mass + dm
        if new_mass > 0:
            obj.material.density *= new_mass / obj.mass
            obj.mass = new_mass
            assembly.mesh.vol_density[assembly.mesh.vol_tag == obj.id] = obj.material.density
            assembly.compute_mass_properties()
            print(f'[Mass update] new mass = {obj.mass:.4f} kg (dm = {dm:.6e} kg)')

    # ============================================================
    # DEBUG ONLY: facet recession → vertex displacement → VTK
    # ============================================================

    # Time step (thermal timestep used by PATO)
    dt = options.pato.time_step

    # Distance each facet recedes over full TITAN step (0.25 s)
    d_n_step = (v_n * options.dynamics.time_step)  # recession (m) over TITAN timestep
    print("DEBUG DEBUG DEBUG DEBUG DEBUG")
    print("max v_n:", np.max(v_n))
    print("expected d_n:", np.max(v_n) * options.dynamics.time_step)
    print("Pato step: ", options.pato.time_step)
    print("Titan step: ", options.dynamics.time_step)

    print("max q_conv:", np.max(q_conv_arr))
    print("max q_cond:", np.max(q_cond_arr))
    print("max q_excess:", np.max(q_excess_arr))
    print("max mdot:", np.max(mdot))
    print("max v_n:", np.max(v_n))

    imax = int(np.argmax(v_n))
    q_cond_recomputed = (
        -k_mid_arr[imax] * (T_int[imax] - Tw[imax]) / dx_s[imax]
        if dx_s[imax] > 0 else 0.0
    )
    q_excess_recomputed = q_conv_arr[imax] - q_cond_recomputed
    print("imax facet (argmax v_n):", imax)
    print("Tw =", float(Tw[imax]))
    print("T_int =", float(T_int[imax]))
    print("dx_s =", float(dx_s[imax]))
    print("k_mid =", float(k_mid_arr[imax]))
    print("raw(T_int-Tw) =", float(T_int[imax] - Tw[imax]))
    print("q_conv =", float(q_conv_arr[imax]))
    print("q_cond(stored) =", float(q_cond_arr[imax]))
    print("q_cond(recomputed) =", float(q_cond_recomputed))
    print("q_excess(stored) =", float(q_excess_arr[imax]))
    print("q_excess(recomputed) =", float(q_excess_recomputed))
    print("mdot =", float(mdot[imax]))
    print("v_n =", float(v_n[imax]))

    # Required geometry
    nodes   = obj.pato.nodes            # (Nnodes, 3)
    facets  = obj.pato.facets           # (Nfacets, 3)
    normals = obj.pato.facet_normal     # (Nfacets, 3) area-scaled

    # Unit normals (facet_normal is area-scaled; normalise for correct displacement magnitude)
    unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-20)

    # Facet displacement vectors (inward); magnitude = recession distance d_n
    facet_disp = -d_n_step[:, None] * unit_normals

    facet_area = np.linalg.norm(normals, axis=1) + 1e-20

    # Nodal displacement: cluster incident facets by normal, sum cluster displacements
    vertex_disp = _nodal_displacement_from_facets_clustered(
        nodes, facets, unit_normals, facet_disp, facet_area, face_theta_deg=FACE_THETA_DEG
    )

    # 1-ring Gaussian smoothing + blend
    sigma_factor = getattr(options.pato, 'recession_sigma_factor', 1.0)
    blend_alpha  = getattr(options.pato, 'recession_smooth_alpha', 0.2)
    neighbors, mean_edge_length = _build_surface_1ring(nodes, facets)
    vertex_disp = _smooth_nodal_displacement_1ring(
        vertex_disp, nodes, neighbors, mean_edge_length,
        sigma_factor=sigma_factor, blend_alpha=blend_alpha
    )  

    obj.pato.vertex_disp_field = vertex_disp

    # Create displaced mesh
    nodes_recessed = nodes + (vertex_disp)


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

    if obj.density_ratio != 1:

        print('Ablation')

        obj.pato.mass_loss = obj.mass - new_mass if new_mass >= 0 else obj.mass
        print('mass loss:', obj.pato.mass_loss)
        obj.material.density *= density_ratio
        obj.mass = new_mass
    
        if (obj.material.density <= 0) or (obj.mass <= 0):
            print("MASS DEMISE OBJ: ", obj.name)
            obj.material.density = 0
            obj.mass = 0
    
        obj.inertia *= density_ratio

    print('OBJ: ', obj.global_ID, 'DENSITY: ', obj.material.density)
    print('OBJ: ', obj.global_ID, 'MASS: ', obj.mass)

    with open(options.output_folder + '/PATO_'+str(obj.global_ID)+'/data/constantProperties', 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        #lines[48] = 'mass ' + str(obj.mass) + ';\n'
        #lines[49] = 'density ' + str(obj.material.density) + ';\n'
        #lines[30] = 'rho_sub_n[0]    '+str(obj.material.density)+';\n'
        #f.seek(0)
        #f.writelines(lines)
        for i, line in enumerate(lines):
            if 'rho_sub_n[0]' in line:
                lines[i] = 'rho_sub_n[0]    '+str(obj.material.density)+';\n'
            if 'mass' in line:
                lines[i] = 'mass ' + str(obj.mass) + ';\n'
            if 'density' in line:
                lines[i] = 'density ' + str(obj.material.density) + ';\n'

        f.seek(0)  # Go back to the start of the file
        f.writelines(lines)
        f.truncate()  # Truncate the file to remove any leftover content

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

def retrieve_volume_vtk_data(n_proc, path, time_to_read):

    filename = [''] * n_proc
    for n in range(n_proc):
        filename[n] = path + "processor" + str(n) + "/VTK/" + "processor" + str(n) + "_" + str(time_to_read) + ".vtk"

    print('\n PATO solution filenames:', filename)

    # Check that VTK files exist before reading (foamToVTK may have crashed with FPE)
    missing = [fn for fn in filename if not os.path.isfile(fn)]
    if missing:
        raise FileNotFoundError(
            "PATO volume VTK files not found. foamToVTK may have crashed (e.g. FPE in FourierMaterialPropertiesModel). "
            "Missing: " + ", ".join(missing) + ". Check PATO/material config (Ta, conductivity) for invalid values."
        )

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
    volume_data = appendFilter.GetOutput()

    writer = vtk.vtkUnstructuredGridWriter()
    pato_output_folder = path + '/Output'
    if not os.path.exists(pato_output_folder): os.mkdir(pato_output_folder)
    time_to_write = str(float(time_to_read)).replace('.','').rjust(5,'0')
    writer.SetFileName(pato_output_folder+'/volume_solution_'+time_to_write+'.vtk')
    writer.SetInputData(volume_data)
    writer.Write()

    # extract geometry-filter skin (for diagnostics/optional use)
    extractSurface = vtk.vtkGeometryFilter()
    extractSurface.SetInputData(volume_data)
    extractSurface.Update()
    surface_data = extractSurface.GetOutput()

    return volume_data, surface_data

def compute_heat_conduction(assembly):

    print('Computing heat conduction between objects ...')

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