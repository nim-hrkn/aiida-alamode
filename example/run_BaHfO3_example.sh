#!/bin/bash
# BaHfO3 (cubic perovskite, 5 atoms): Born effective charges with SevenNet-Polar, harmonic phonons with and
# without the LO-TO correction, cubic IFCs and the RTA lattice thermal conductivity, and the SCPH
# (anharmonic, finite-temperature) phonons with structural relaxation.  Forces: MatterSim.
# All steps run through AiiDA (codes alm, anphon, displace, analyze_phonons, ase_runner @$AIIDA_ALAMODE_COMPUTER);
# finished steps are reused from run_v010*/BaHfO3/.node.json on rerun.
set -e
cd "$(dirname "$0")"
COMMON="--structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 --borninfo-calculator sevennet-polar --dielectric 4.9 --njobs 2"

# 1. Z* (alamode.bec_ase) -> BORNINFO; harmonic phonons NONANALYTIC 0 / 3;
# 2. cubic IFCs (cutoff 8 Bohr, 0.04 A) -> anphon RTA (10x10x10, NONANALYTIC 3) -> analyze_phonons
python run_alamode_phonons.py $COMMON --nonanalytic 0 3 --emax 900 \
    --cubic --cubic-cutoff 8.0 --cubic-mag 0.04 --rta-qmesh 10 --root run_v010

# 3. MD (300 K) + LASSO anharmonic IFCs (NORDER 3) -> SCPH + RELAX_STR 1, 50..700 K, NONANALYTIC 3
python run_alamode_scph.py $COMMON --no-ref --root run_v010_scph
