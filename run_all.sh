#!/bin/bash
set -x
cd /home/claude/nanocubes-helix-simulation
python3 simulations/run_density_sweep.py
python3 simulations/run_belt_sweep.py
echo "TODO_LISTO"
