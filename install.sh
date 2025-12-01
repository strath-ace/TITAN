#!/bin/bash
set -eu
echo "
[INSTALL] Warning! This installation file is experimental
"
conda --version
echo "
[INSTALL] Welcome to TITAN!"
echo "[INSTALL] Creating TITAN directory
"
git clone https://github.com/strath-ACE/TITAN -b develop
cd TITAN
echo "
[INSTALL] Creating TITAN conda environment...
"
conda env create --name titan --file titan_compatibility_env.yml
eval "$(conda shell.bash hook)"
conda activate titan
swig -version
echo "
[INSTALL] Cloning submodules...
"
git submodule update --init --recursive
cd ./Executables
echo "
[INSTALL] Installing su2gmf...
"
CFLAGS="-Wno-return-mismatch -Wno-implicit-function-declaration" pip install --config-settings editable_mode=compat -e amgio/su2gmf/
ln -s ./amgio/su2gmf/su2_to_gmf.py ./su2_to_gmf.py
ln -s ./amgio/su2gmf/gmf_to_su2.py ./gmf_to_su2.py
echo "
[INSTALL] Intalling mutation++...
"
cd ./mutationpp
python setup.py build
python setup.py install
conda deactivate
echo "
[INSTALL] Making PATO environment...
"
conda config --add channels conda-forge
conda config --add channels pato.devel
conda config --set channel_priority strict
conda create -y --name pato -c conda-forge -c pato.devel pato
echo "
[INSTALL] Finished, thanks for installing TITAN!
"
