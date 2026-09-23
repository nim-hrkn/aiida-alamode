# alamode_aiida

the latest version is v0.9_free_energy_convergence



## v0.10: MatterSim / ASE-calculator replacement of the force calculations

The forces of the displaced structures can be computed with an ASE calculator (MatterSim by default;
MACE, CHGNet, SevenNet, ORB, EMT, ... through `aiida_alamode.mattersim_runner.CALCULATORS`)
instead of DFT or LAMMPS.  Install with `pip install -e .[mattersim]` and set up a code for the
console script `alamode-mattersim` (e.g. `verdi code create core.code.installed --label mattersim
--computer <computer> --filepath-executable $(which alamode-mattersim) --default-calc-job-plugin alamode.mattersim`).

CalcJobs (`aiida_alamode.calculations.mattersim_calcjob`):

| entry point | input | output |
|---|---|---|
| `alamode.mattersim` | `structures` (TrajectoryData) | `arrays` (energies, forces, stresses, positions, cells) |
| `alamode.mattersim_relax` | `structure` | relaxed `structure` (volume only by default) |
| `alamode.mattersim_md` | supercell `structure`, temperature, ... | `displaced_structures` (sampled MD snapshots + random displacements, as `displace.py -md --random`) |
| `alamode.mattersim_elastic` | primitive `structure` | `strain_ifc_folder` (`elastic_constants.in`, `strain_force.in` for the QHA) |

The WorkChain `alamode.force_simulator_mattersim` (`ForcesMattersimWorkChain`) computes the forces of a
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

`alamode.mattersim_bec` (`MattersimBecCalculation`) computes the Born effective charges of the primitive
cell with a calculator that provides them (`{"name": "sevennet-polar"}`, checkpoints from
zenodo 10.5281/zenodo.21322761) and writes the BORNINFO file for anphon; the dielectric tensor comes from
the model or the `dielectric` input.  `run_alamode_phonons.py --borninfo-calculator sevennet-polar
--dielectric 6.7 --nonanalytic 0 3` uses it (cubic BaTiO3; ZrO2, BaZrO3 and the full BaHfO3 chain `example/run_BaHfO3_example.sh` in the docs).  See `docs/born_effective_charges.md`.

Running the MatterSim / SevenNet jobs on a remote GPU node through `core.ssh_async` + slurm (setup, ALAMODE build with conda + MKL, pitfalls, CPU vs GPU timings): `docs/remote_gpu_computer.md`.  A Claude Code skill summarizing the plugin, the drivers and the pitfalls is in `.claude/skills/aiida-alamode/SKILL.md`.
