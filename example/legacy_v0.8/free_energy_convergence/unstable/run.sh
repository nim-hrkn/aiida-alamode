pwd=`pwd`
echo ${pwd}
script_home=/home/max/Documents/dxmag-phonon
data_home=/home/max/Documents/dxmag-phonon/example/free_energy_convergence/unstable,run
python ${script_home}/scripts/check_convergence_free_energy_v2.py --VASP ${data_home}/SPOSCAR --prim ${data_home}/relaxed_final_symm.POSCAR.vasp --dfset dfset --ninc 2 --verbosity 1

