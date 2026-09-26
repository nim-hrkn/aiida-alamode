# Legacy examples (aiida-alamode v0.7 - v0.9, 2022-2023)

Kept for reference only. The notebooks show the LAMMPS and Quantum ESPRESSO route of the plugin
as it was in v0.8 and **do not run unchanged** with the current package:

- they import the old package name `alamode_aiida` (now `aiida_alamode`);
- they use entry points that no longer exist: `alamode.lammps` (dangling since v0.9), `alamode.analyze_phonon`
  (now `alamode.analyze_phonons`), `alamode.pwx` / `alamode.dispall_pwx`;
- `tools/` uses `distutils` (removed in Python 3.12) and needs `aiida-lammps`.

| path | contents |
|---|---|
| `step1_make_structure.ipynb` … `step3C_postprocess.ipynb` | Si and GaN phonons with LAMMPS (steps 1-3: structure, harmonic, anharmonic) |
| `step2B_use_pwx_devel.ipynb` | the same harmonic step with pw.x (development state) |
| `qe/7080_alamode_all_aiida.ipynb` | the whole workflow with Quantum ESPRESSO |
| `tools/` | helper modules imported by the notebooks (NodeBank, LAMMPS support, structure input) |
| `lammps_input/` | LAMMPS data files and potentials (Si Stillinger-Weber / Tersoff, GaN Tersoff) |
| `free_energy_convergence/` | v0.9 (2023-02, from `example_dxmag/`): automatic convergence of the vibrational free energy (`alamode.anphon_calculator`, `alamode.check_convergence_freeenergy`) on a converged and an unstable case; its README refers to `scripts/check_convergence_free_energy.py`, which is not in this repository, and `check_convergence_freeenergy` needs the `alm` Python package |

The current way to bring your own DFT or LAMMPS forces is `alamode.displace_pf` (writes the displaced
structures in the format of your code) followed by `alamode.extract` (DFSET from the outputs); see the top-level README.
The current worked examples use MatterSim through the drivers in `example/`.
