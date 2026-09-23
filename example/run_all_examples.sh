#!/bin/bash
# Run every example on one AiiDA computer.  usage: run_all_examples.sh <computer> <root> [--gpu] [--njobs N] [--cores N]
# Logs: <root>_<example>.log ; results: <root>/, <root>_scph/, <root>_qha/  (each driver reuses finished steps on rerun)
set -e
cd "$(dirname "$0")"
COMPUTER=$1; ROOT=$2; shift 2; EXTRA="$*"
C="--computer $COMPUTER $EXTRA"
run() { name=$1; shift; nohup python -u "$@" $C > "${ROOT}_${name}.log" 2>&1 & }
BEC="--borninfo-calculator sevennet-polar --dielectric-model anisonet --emax 900"
run Si     run_alamode_phonons.py --preset Si   --root $ROOT
run PbTe   run_alamode_phonons.py --preset PbTe --root $ROOT
run BaHfO3 run_alamode_phonons.py --structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 --nonanalytic 0 3 $BEC \
           --cubic --cubic-cutoff 8.0 --cubic-mag 0.04 --rta-qmesh 10 --root $ROOT
run BaZrO3 run_alamode_phonons.py --structure BaZrO3_Pm-3m.cif --supercell 2 2 2 --name BaZrO3 --nonanalytic 0 3 $BEC --root $ROOT
run ZrO2   run_alamode_phonons.py --structure ZrO2_P2_1c.cif --supercell 2 2 2 --relax full --idealize --name ZrO2 --nonanalytic 0 3 $BEC --root $ROOT
run BaTiO3_scph run_alamode_scph.py --root ${ROOT}_scph
run BaHfO3_scph run_alamode_scph.py --structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 --no-ref \
           --borninfo-calculator sevennet-polar --dielectric-model anisonet --root ${ROOT}_scph
run ZnO_qha run_alamode_qha.py --root ${ROOT}_qha
echo "launched 8 drivers on $COMPUTER (logs ${ROOT}_*.log)"
