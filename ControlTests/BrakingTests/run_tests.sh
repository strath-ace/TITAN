#!/usr/bin/env bash
set -euo pipefail

for deg in $(seq 90 -5 60); do
  cfg="ControlTests/BrakingTests/${deg}deg.txt"
  echo "Running: python TITAN.py -c ${cfg}"
  python TITAN.py -c "${cfg}"
done
