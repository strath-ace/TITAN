#!/bin/bash
set -eu
echo "
[INSTALL] Warning! This installation file is experimental
"
conda --version
if conda env list | awk '$1 == "titan" { found=1 } END { exit !found }'; then
    echo "[INSTALL] Conda environment 'titan' already exists. Rename it to continue. Aborting." >&2
    exit 1
fi

if conda env list | awk '$1 == "pato" { found=1 } END { exit !found }'; then
    while true; do
        read -r -p "[INSTALL] Conda environment 'pato' already exists. Continue anyway? [y/n] " answer
        case "$answer" in
            [Yy]|[Yy][Ee][Ss]) break ;;
            [Nn]|[Nn][Oo]) echo "[INSTALL] Aborting." >&2; exit 1 ;;
            *) echo "Please answer y or n." >&2 ;;
        esac
    done
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
python mutation_patch.py
cd ./mutationpp
mkdir build install
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
make install
cd ..
pip install .
conda deactivate
echo "
[INSTALL] Making PATO environment...
"
conda config --add channels conda-forge
conda config --add channels pato.devel
conda config --set channel_priority strict
conda create -y --name pato -c conda-forge -c pato.devel pato
echo "[INSTALL] Finished, thanks for installing TITAN!"
cd ../../
export TITAN_PATH=$(realpath .)
echo "To finalise installation please add the following to your .bashrc..."
echo "export TITAN_PATH="$TITAN_PATH