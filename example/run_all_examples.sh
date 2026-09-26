#!/bin/bash
# Run every example on one AiiDA computer.  usage: run_all_examples.sh <computer> <root> [--gpu] [--njobs N] [--cores N]
# Logs: <root>_<example>.log ; results: <root>/, <root>_scph/, <root>_qha/  (each driver reuses finished steps on rerun)
set -e
cd "$(dirname "$0")"
COMPUTER=$1; ROOT=$2; shift 2; EXTRA="$*"
C="--computer $COMPUTER $EXTRA"
run() { name=$1; shift; nohup python -u "$@" $C > "${ROOT}_${name}.log" 2>&1 & }
# The ML packages are optional for the plugin: check them on the computer first and skip what cannot run.
python check_packages.py --computer $COMPUTER mattersim sevennet-polar anisonet | tee "${ROOT}_packages.log"
has() { grep -q "^$1: ok" "${ROOT}_packages.log"; }
has mattersim || { echo "MatterSim is not installed on $COMPUTER: nothing to run (see README, 'Optional packages')"; exit 1; }
if has sevennet-polar && has anisonet; then BEC="--borninfo-calculator sevennet-polar --dielectric-model anisonet --emax 900"; ZSTAR=1
else echo "SevenNet-Polar / AnisoNet missing on $COMPUTER: the Z* / eps_inf examples (BaHfO3, BaZrO3, ZrO2, Li3PO4, BaHfO3_scph) are skipped"; ZSTAR=; fi
if has anisonet; then EPS="--dielectric-model anisonet"; else EPS="--dielectric 3.0"; fi   # MgO / NaCl: literature Z*; eps_inf predicted or given
run Si     run_alamode_phonons.py --preset Si   --root $ROOT
run PbTe   run_alamode_phonons.py --preset PbTe --root $ROOT
[ -n "$ZSTAR" ] && run BaHfO3 run_alamode_phonons.py --structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 --nonanalytic 0 3 $BEC \
           --cubic --cubic-cutoff 8.0 --cubic-mag 0.04 --rta-qmesh 10 --root $ROOT
[ -n "$ZSTAR" ] && run BaZrO3 run_alamode_phonons.py --structure BaZrO3_Pm-3m.cif --supercell 2 2 2 --name BaZrO3 --nonanalytic 0 3 $BEC --root $ROOT
[ -n "$ZSTAR" ] && run ZrO2   run_alamode_phonons.py --structure ZrO2_P2_1c.cif --supercell 2 2 2 --relax full --idealize --name ZrO2 --nonanalytic 0 3 $BEC --root $ROOT
[ -n "$ZSTAR" ] && run Li3PO4 run_alamode_phonons.py --structure Li3PO4_Pnma.cif --supercell 2 1 2 --relax full --idealize --name Li3PO4 --nonanalytic 0 3 $BEC --emax 1300 --root $ROOT
run MgO    run_alamode_phonons.py --structure MgO_Fm-3m.cif --supercell 2 2 2 --name MgO --nonanalytic 0 3 --born-charges Mg:1.96 O:-1.96 $EPS --emax 900 --root $ROOT
run NaCl   run_alamode_phonons.py --structure NaCl_Fm-3m.cif --supercell 2 2 2 --name NaCl --nonanalytic 0 3 --born-charges Na:1.10 Cl:-1.10 ${EPS/3.0/2.3} --emax 400 --root $ROOT
run BaTiO3_scph run_alamode_scph.py --root ${ROOT}_scph
[ -n "$ZSTAR" ] && run BaHfO3_scph run_alamode_scph.py --structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 --no-ref \
           --borninfo-calculator sevennet-polar --dielectric-model anisonet --root ${ROOT}_scph
run ZnO_qha run_alamode_qha.py --root ${ROOT}_qha
echo "launched the drivers on $COMPUTER (logs ${ROOT}_*.log)"
