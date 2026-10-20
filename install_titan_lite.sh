#!/bin/bash
set -eu
echo "
[INSTALL] Warning! This installation file is experimental 
"
echo "
[INSTALL] This lite install does not include Mutation++, PATO or su2gmf
"
conda --version
if conda env list | awk '$1 == "titan" { found=1 } END { exit !found }'; then
    echo "[INSTALL] Conda environment 'titan' already exists. Rename it to continue. Aborting." >&2
    exit 1
fi
echo "
[INSTALL] Welcome to TITAN!"
echo "[INSTALL] Creating TITAN directory
"
git clone https://github.com/strath-ACE/TITAN -b rc_v02_atlas
cd TITAN
echo "
[INSTALL] Creating TITAN conda environment...
"
conda env create --name titan --file titan_env.yml
eval "$(conda shell.bash hook)"
conda activate titan
echo "
[INSTALL] Pip installing...
"
pip install .
export TITAN_PATH=$(realpath .)
echo "To finalise installation please add the following to your .bashrc..."
echo "export TITAN_PATH="$TITAN_PATH