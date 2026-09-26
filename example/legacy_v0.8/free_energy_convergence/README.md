## Example for automatic calculation of vibrational free energy

### Converged case (gen: 0, id: 1)

```bash
cd converged/
python ../../../scripts/check_convergence_free_energy.py --VASP SPOSCAR --prim relaxed_final_symm.POSCAR.vasp --dfset DFSET_random_latest --ninc 2 --verbosity 1
```

This calculation converges after 2 iterations (only 4 DFT training dataset). 
For more details of the convergence check, please see `free_energy_history.txt`.
The converged phonon dispersion, DOS, and thermodynamic function are plotted in separate PDF files generated in the working directory.


### Unstable (unconverged) case (gen: 0, id: 9)

```bash
cd unstable/
python ../../../scripts/check_convergence_free_energy.py --VASP SPOSCAR --prim relaxed_final_symm.POSCAR.vasp --dfset DFSET_random_latest --ninc 2 --verbosity 1
```

In this case, unstable phonons still exist even after 9th iteration. 
The script judges that this system is dynamically unstable and stops the iteration.
A file named `UNSTABLE` is created in the working directory.