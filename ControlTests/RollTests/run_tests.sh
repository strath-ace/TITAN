#!/usr/bin/env bash
set -euo pipefail

for deg in $(seq 0 10 90); do
  cfg="ControlTests/RollTests/${deg}deg.txt"
  echo "Running: python TITAN.py -c ${cfg}"
  python TITAN.py -c "${cfg}"
done
