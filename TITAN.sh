#!/bin/bash
set -eu
## This script automates the running of a TITAN simulation
export FILE=$(realpath "$1")
export REF=$2
ln -sf $FILE $TITAN_PATH/link_$REF.cfg
cd $TITAN_PATH
shift 2
python titan/run_TITAN.py -c link_$REF.cfg "$@"
