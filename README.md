# alamode_aiida

the latest version is v0.9_free_energy_convergence



## v0.10: MatterSim / ASE-calculator replacement of the force calculations

The forces of the displaced structures can be computed with an ASE calculator (MatterSim by default;
MACE, CHGNet, SevenNet, ORB, EMT, ... through `aiida_alamode.ase_runner.CALCULATORS`)
instead of DFT or LAMMPS.  Install with `pip install -e .[mattersim]` and set up a code for the
console script `alamode-ase-runner` (e.g. `verdi code create core.code.installed --label ase_runner
--computer <computer> --filepath-executable $(which alamode-ase-runner) --default-calc-job-plugin alamode.forces_ase`).

CalcJobs.  Two kinds of prediction are kept apart: *forces* (energies, forces, stresses: `calculations/force_calcjob.py`, base `ForceCalculatorBaseCalculation`) and *dielectric properties* (Born effective charges, dielectric tensor: `calculations/dielectric_calcjob.py`, base `DielectricCalculatorBaseCalculation`).  The engine can be a DFT code (VASP, Quantum ESPRESSO: to be added as subclasses with the same output ports) or a machine-learning potential through ASE (`engine_base.py`: `AseRunnerBaseCalculation`, the `alamode-ase-runner` script).  

| entry point | input | output |
|---|---|---|
| `alamode.forces_ase` (`AseForcesCalculation`) | `structures` (TrajectoryData) | `arrays` (energies, forces, stresses, positions, cells) |
| `alamode.relax_ase` | `structure` | relaxed `structure` (volume only by default) |
| `alamode.md_ase` | supercell `structure`, temperature, ... | `displaced_structures` (sampled MD snapshots + random displacements, as `displace.py -md --random`) |
| `alamode.elastic_ase` | primitive `structure` | `strain_ifc_folder` (`elastic_constants.in`, `strain_force.in` for the QHA) |

The WorkChain `alamode.forces` (`ForcesWorkChain`; `forces_plugin` selects the forces CalcJob, default `alamode.forces_ase`) computes the forces of a
TrajectoryData in `njobs` scheduler jobs and returns `arrays` and the `dfset` (List of the DFSET lines,
Rydberg atomic units as `extract.py --QE`); `subtract_offset` subtracts the forces of the undisplaced
cell (needed for strained cells).

Other additions: `alamode.alm_cv` (elastic-net cross validation, `results['alpha_min']`), the anphon
CalcJob accepts `borninfo`, `fc2xml`, `extra_files` and any `MODE` (SCPH, QHA: the `param` sections are
written verbatim, all `{prefix}.*` files are returned in `output_folder`), `param['kpoint']` overrides the
automatic band path, and `param['interaction']` (NBODY) is kept.  `aiida_alamode.io.supercell.make_diagonal_supercell`
builds supercells whose translation 1 coincides with the primitive cell (required by NONANALYTIC = 3).

Fixed: the QE / VASP structure writers of `displace_pf` / `extract`, the cutoff dropped by `alm_opt`,
NAT in anphon inputs, the `nsample gridtype` arguments of `analyze_phonons` (alamode >= 1.5),
`withmpi` on aiida-core >= 2.3; aiida-lammps is optional.

Examples (`example/`, tutorial settings with MatterSim instead of DFT; results and figures under `run_*/<name>/`):

```
python run_alamode_phonons.py --preset Si            # harmonic phonons, cubic IFCs, RTA kappa (tutorial 3, 5-7)
python run_alamode_phonons.py --preset PbTe          # Born charges, NONANALYTIC 0-3
python run_alamode_phonons.py --structure X.cif --supercell 2 2 2 [--calculator mace]
python run_alamode_scph.py                           # BaTiO3: MD + LASSO anharmonic IFCs, SCPH structural relaxation (7.1, 7.4)
python run_alamode_qha.py                            # ZnO: strained IFCs, elastic constants, QHA thermal expansion (7.5)
```

### Born effective charges with SevenNet-Polar (v0.10)

Z* and eps_inf are separate jobs: `alamode.bec_ase` (`AseBornChargesCalculation`, SevenNet-Polar checkpoints from zenodo 10.5281/zenodo.21322761) and `alamode.epsinf_ase` (`AseDielectricTensorCalculation`, the electronic dielectric tensor from AnisoNet, github.com/virtualatoms/AnisoNet).  `alamode.borninfo` (`BornInfoWorkChain`) runs both (or takes a given `dielectric`) and writes the BORNINFO file for anphon; the engines are chosen by `bec_plugin` / `epsinf_plugin` (a VASP job giving both would skip the eps_inf job).  `run_alamode_phonons.py --borninfo-calculator sevennet-polar
--dielectric 6.7 --nonanalytic 0 3` uses it (cubic BaTiO3; ZrO2, BaZrO3 and the full BaHfO3 chain `example/run_BaHfO3_example.sh` in the docs).  See `docs/born_effective_charges.md`.

Running the MatterSim / SevenNet jobs on a remote GPU node through `core.ssh_async` + slurm (setup, ALAMODE build with conda + MKL, pitfalls, CPU vs GPU timings): `docs/remote_gpu_computer.md`.  A Claude Code skill summarizing the plugin, the drivers and the pitfalls is in `.claude/skills/aiida-alamode/SKILL.md`.
