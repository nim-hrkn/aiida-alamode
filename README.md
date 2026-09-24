# aiida-alamode

AiiDA plugin for [ALAMODE](https://alamode.readthedocs.io/) (alm, anphon, displace, analyze_phonons).
Every step of an ALAMODE calculation, from the displaced structures to the thermal conductivity,
runs as an AiiDA process, so each result keeps its full provenance.

Version 1.0.0.  The forces and the dielectric properties can come from DFT (VASP, Quantum ESPRESSO),
LAMMPS, or a machine-learning model through ASE (MatterSim by default).

```
structure ─ relax_ase ─ alm_suggest ─ displace_pf ─ forces ─ alm_opt ─ anphon ─ analyze_phonons
                                                                  │
                        bec_ase (Z*) + epsinf_ase (ε∞) ─ BORNINFO ┘   (LO-TO splitting)
```

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Forces from a machine-learning potential](#forces-from-a-machine-learning-potential)
4. [LO-TO splitting: Born effective charges Z\* and the dielectric constant ε∞](#lo-to-splitting-born-effective-charges-z-and-the-dielectric-constant-ε)
5. [Results of the examples](#results-of-the-examples)
6. [Other changes since v0.9](#other-changes-since-v09)
7. [Further documentation](#further-documentation)

## Install

```
pip install -e .[mattersim]            # MatterSim forces
pip install -e .[sevennet-polar]       # Born effective charges (SevenNet-Polar)
pip install -e .[equivar]              # Born effective charges (Equivar: torch + e3nn, weights from Mendeley Data)
pip install -e <clone of github.com/virtualatoms/AnisoNet>   # ε∞ (needs lightning, pymatgen)
```

Register one AiiDA code per ALAMODE program (`alm`, `anphon`, `displace`, `analyze_phonons`) and one for
the ASE runner:

```
verdi code create core.code.installed --label ase_runner --computer <computer> \
    --filepath-executable $(which alamode-ase-runner) --default-calc-job-plugin alamode.forces_ase
```

After editing the plugin, run `pip install -e . --no-deps` (entry points) and `verdi daemon restart`.

## Quick start

The drivers in `example/` reproduce the ALAMODE tutorials with MatterSim instead of DFT.
Results and figures go to `run_*/<name>/`, and finished steps are reused when a driver is run again.
Set the computer with `--computer <label>` or `export AIIDA_ALAMODE_COMPUTER=<label>`.

```
cd example
python run_alamode_phonons.py --preset Si     # harmonic phonons, cubic IFCs, RTA thermal conductivity (tutorial 3, 5-7)
python run_alamode_phonons.py --preset PbTe   # LO-TO splitting, NONANALYTIC 0-3
python run_alamode_scph.py                    # BaTiO3: MD + LASSO anharmonic IFCs, SCPH structural relaxation (7.1, 7.4)
python run_alamode_qha.py                     # ZnO: strained IFCs, elastic constants, QHA thermal expansion (7.5)

# any structure, any calculator
python run_alamode_phonons.py --structure X.cif --supercell 2 2 2 --calculator mace

# a polar material, with Z* and ε∞ both predicted (no literature value needed)
python run_alamode_phonons.py --structure BaHfO3_Pm-3m.cif --supercell 2 2 2 --name BaHfO3 \
    --borninfo-calculator sevennet-polar --dielectric-model anisonet --nonanalytic 0 3

bash run_BaHfO3_example.sh                    # BaHfO3: Z*, ε∞, phonons, κ, SCPH in one go
bash run_all_examples.sh <computer> <root>    # every example at once
```

`--supercell` multiplies the cell in the structure file.  The PbTe 4×4×4 is of the fcc *primitive*
cell (128 atoms), the Si 2×2×2 of the conventional cell.

## Forces from a machine-learning potential

Two kinds of prediction are kept apart, each with its own base class, so that a DFT engine can
later be added as a subclass with the same output ports:

- **Forces** (energies, forces, stresses): `calculations/force_calcjob.py`, base `ForceCalculatorBaseCalculation`.
- **Dielectric properties** (Z\*, ε∞): `calculations/dielectric_calcjob.py`, bases `DielectricCalculatorBaseCalculation` and `DielectricTensorBaseCalculation`.

The ASE engine (`engine_base.py`, script `alamode-ase-runner`) accepts `mattersim`, `mace`, `mace-off`,
`chgnet`, `sevennet`, `sevennet-polar`, `equivar`, `orb`, `emt`, `lj` (`aiida_alamode.ase_runner.CALCULATORS`).

| entry point | input | output |
|---|---|---|
| `alamode.forces_ase` | `structures` (TrajectoryData) | `arrays` (energies, forces, stresses, positions, cells) |
| `alamode.relax_ase` | `structure` | relaxed `structure` (volume only by default) |
| `alamode.md_ase` | supercell `structure`, temperature | `displaced_structures` (MD snapshots + random displacements, as `displace.py -md --random`) |
| `alamode.elastic_ase` | primitive `structure` | `strain_ifc_folder` (`elastic_constants.in`, `strain_force.in` for the QHA) |
| `alamode.bec_ase` | primitive `structure`, `calculator` | `born_effective_charges` |
| `alamode.epsinf_ase` | primitive `structure`, `dielectric_model` | `dielectric_tensor` (`epsilon_inf`) |
| `alamode.forces` (WorkChain) | TrajectoryData, `njobs`, `forces_plugin` | `arrays`, `dfset` (DFSET lines, Ry a.u.) |
| `alamode.borninfo` (WorkChain) | primitive `structure`, `bec`, `epsinf` or `dielectric` | `born_effective_charges`, `dielectric_tensor`, `borninfo` |

`alamode.forces` splits the structures over `njobs` scheduler jobs.  With `subtract_offset` it subtracts
the forces of the undisplaced cell, which strained cells need.

## LO-TO splitting: Born effective charges Z\* and the dielectric constant ε∞

In a polar crystal the long-range Coulomb field splits the longitudinal (LO) and transverse (TO)
optical modes at Γ.  The force constants of a finite supercell miss this, so anphon adds a
non-analytic correction (`NONANALYTIC = 1, 2, 3`).  The correction needs a BORNINFO file with two quantities:

- **Z\*, the Born effective charge tensor of each atom.** It is the polarization created when the atom moves, or equally the force on the atom in an electric field.
- **ε∞, the high-frequency (electronic) dielectric tensor.** It is the screening by the electrons alone, with the ions clamped. It is *not* the static dielectric constant ε₀, which also contains the ionic response and is much larger in polar crystals (MgO: ε∞ = 3.0, ε₀ = 9.8).

The LO frequency grows with Z\*²/ε∞, so an ε∞ that is too large shrinks the splitting, and a static ε₀
put in its place would nearly remove it.

### Where the two quantities come from

| quantity | source | how to use it |
|---|---|---|
| Z\* | SevenNet-Polar (ML) | `--borninfo-calculator sevennet-polar` (`alamode.bec_ase`) |
| Z\* | Equivar (ML) | `--borninfo-calculator equivar` (`alamode.bec_ase`) |
| ε∞ | AnisoNet (ML) | `--dielectric-model anisonet` (`alamode.epsinf_ase`) |
| ε∞ | literature or your own DFT value | `--dielectric 6.7` (1, 3 or 9 numbers) |
| both | an existing BORNINFO file | `--borninfo FILE` |
| both | DFT (VASP LEPSILON, QE ph.x) | a subclass of the dielectric base; `alamode.borninfo` then skips the ε∞ job |

SevenNet-Polar and Equivar give Z\* only.  AnisoNet (Lou & Ganose, arXiv:2405.07915, MIT licence) predicts the
full ε∞ tensor, including its anisotropy, in about 3 s on a CPU.

### Setting up AnisoNet

1. Install the package from a clone of github.com/virtualatoms/AnisoNet.
2. Download the weights `anisonet-stock.ckpt` (249 MB) from figshare 26270974.
3. Put the file at `~/models/anisonet/anisonet-stock.ckpt`, or point `ANISONET_CHECKPOINT` at it.

### Setting up Equivar

Equivar (Kutana, Shimizu, Watanabe & Asahi, Sci. Rep. 15, 16687 (2025)) is an equivariant GCNN trained on
the same DFPT data as SevenNet-Polar: ABO₃ perovskites (A = Ba, Ca, Sr, Pb; B = Ti, Zr, Hf), Li₃PO₄ and
ZrO₂, so it knows the ten elements **Ba, Ca, Hf, Li, O, P, Pb, Sr, Ti, Zr** and nothing else.

1. Nothing to install beyond torch and e3nn (`pip install -e .[equivar]`).  The published weights are
   TorchScript archives; `aiida_alamode.equivar` registers pure-torch stand-ins for the torch_scatter /
   torch_sparse operators they reference, so those packages are not needed.
2. Download `BM1.pt` (510k parameters, 2.5 MB) and `BM2.pt` (131k parameters, 1 MB) from
   Mendeley Data 10.17632/hx8kcpxh84.1 to `~/models/equivar/`, or point `EQUIVAR_MODEL` at one.
   BM1 is the default; `--borninfo-kwargs '{"model": "~/models/equivar/BM2.pt"}'` selects BM2.
3. Equivar gives Z\* only, so add `--dielectric-model anisonet` or `--dielectric E`.

### Example 1: driver

```
python run_alamode_phonons.py --structure BaTiO3_Pm-3m.cif --supercell 2 2 2 --name BaTiO3_bec \
    --nonanalytic 0 3 --borninfo-calculator sevennet-polar --dielectric-model anisonet
```

The driver relaxes the cell, computes the harmonic IFCs, runs `alamode.borninfo` on the primitive
cell, and draws the bands with and without the correction.  The log shows what went into BORNINFO
(values rounded here):

```
Born effective charges (diagonal) [e]: {'Ba': [2.72, 2.72, 2.72], 'Ti': [7.72, 7.72, 7.72], 'O': [-2.15, -2.15, -6.14]}
dielectric tensor (anisonet): [[6.3, 0, 0], [0, 6.3, 0], [0, 0, 6.3]]
```

Replace `--dielectric-model anisonet` by `--dielectric 6.7` to use a literature value instead.

### Example 2: Python, ε∞ only

```python
from aiida.engine import submit
from aiida.orm import Dict, StructureData, load_code
from aiida.plugins import CalculationFactory

calc = submit(CalculationFactory("alamode.epsinf_ase"),
              code=load_code("ase_runner@<computer>"),
              structure=StructureData(ase=atoms),                  # primitive cell
              dielectric_model=Dict({"name": "anisonet"}),
              metadata={"options": {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}})
# when finished:
eps_inf = calc.outputs.dielectric_tensor.get_array("epsilon_inf")  # (3, 3)
```

### Example 3: Python, the whole BORNINFO

```python
from aiida.plugins import WorkflowFactory

code = load_code("ase_runner@<computer>")
options = Dict({"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}})
wc = submit(WorkflowFactory("alamode.borninfo"),
            structure=prim,                                        # the same StructureData given to anphon
            bec={"code": code, "calculator": Dict({"name": "sevennet-polar"}), "options": options},
            epsinf={"code": code, "dielectric_model": Dict({"name": "anisonet"}), "options": options})
# or, instead of epsinf:  dielectric=List([6.7])
# outputs: born_effective_charges, dielectric_tensor, borninfo (SinglefileData for anphon), results
```

The engines are chosen by `bec_plugin` (default `alamode.bec_ase`) and `epsinf_plugin`
(default `alamode.epsinf_ase`).

### Pitfalls

- **Atom order.** BORNINFO lists Z\* in the order of the anphon `&position` block, which is the atom order of the structure. It is not the KD order of `&general`, which is sorted by atomic number. Give the Born-charge job the same StructureData as anphon.
- **NONANALYTIC = 3 (Ewald) needs a clean supercell.** `ase.build.make_supercell` may put the atoms of the first translation at another periodic image. Harmonic bands still look right, but the Ewald correction becomes badly wrong. Build supercells with `aiida_alamode.io.supercell.make_diagonal_supercell`.
- **AnisoNet normalisation.** The upstream prediction notebook recomputes `num_neighbors` from the structures being predicted, so one structure gives different values in different batches. The runner fixes it at the training-set value 34.956847.
- **Out-of-distribution materials.** Z\* of TiO₂ violates the acoustic sum rule by 0.3 to 1.2 e before it is enforced, and the values are less reliable. Neither SevenNet-Polar nor Equivar has learned Zn, Si or Te.
- **Equivar graph.** The upstream script builds the graph from minimum-image distances, so a cell shorter than 6 Å (twice the 3 Å cutoff) loses neighbours: cubic BaTiO₃ then gives Ti 4.4 instead of 7.9. The runner uses a full periodic neighbour list instead (`mic=False`), which reproduces the training targets to 0.02 e and is independent of the supercell.

`example/test_epsinf.py --computer <label>` checks ε∞ on nine materials against literature values.
Details and more materials: `docs/born_effective_charges.md`.

## Results of the examples

All numbers are from MatterSim forces, SevenNet-Polar (PS-M checkpoint) Z\* and AnisoNet ε∞, run on
2026-09-23.  CPU and GPU runs agree to the third decimal.

### What each example tests

| material | stable in the harmonic approximation? | physics tested | key result |
|---|---|---|---|
| Si | yes | dispersion, 3-phonon scattering, κ | κ(300 K) = 149 W/mK (exp. about 150, tutorial DFT 113) |
| PbTe | yes, soft TO | LO-TO splitting, non-analytic correction | Γ TO 1.57 THz (tutorial DFT 1.26) |
| BaHfO₃ | yes | Z\* and ε∞ from ML, κ, SCPH | κ(300 K) = 8.3 W/mK; stays cubic up to 700 K |
| BaZrO₃ | R-point rotation unstable | LO-TO with ML Z\* and ε∞ | R mode −1.38 THz, unrelated to the correction |
| m-ZrO₂ | yes | low-symmetry Z\*, full cell relaxation | weak imaginary mode (−0.8 THz) removed by NONANALYTIC 3 |
| BaTiO₃ | Γ-point unstable | ferroelectric transition, SCPH relaxation | tetragonal up to 350 K, cubic from 400 K (exp. 393 K) |
| ZnO | yes | anisotropic thermal expansion, QHA | strain at 300 K: u_xx = u_zz = 0.0032 (0.0025 at 0 K from zero-point motion) |

### Born effective charges Z\* (diagonal, after the sum rule, in e)

| material | A / B site | O (⊥ / ∥ to the B–O bond) | DFT literature |
|---|---|---|---|
| cubic BaTiO₃ | Ba 2.72 / Ti 7.72 | −2.15 / −6.14 | 2.75 / 7.16 / −2.11 / −5.69 |
| BaZrO₃ | Ba 2.72 / Zr 5.68 | −1.98 / −4.44 | 2.7 / 6.1 / −2.0 / −4.8 |
| BaHfO₃ | Ba 2.74 / Hf 5.42 | −1.99 / −4.19 | none found |
| m-ZrO₂ | Zr 5.54 / 5.44 / 5.01 | −2.5 to −2.8 (2 sites) | Zr 5.4 to 5.7 / O −2.3 to −3.2 |
| rutile TiO₂ | Ti 6.8 / 6.8 / 7.8 | −3.45 / −3.45 / −4.08 | 6.3 / 6.3 / 7.5 |

### Z\* from Equivar compared with SevenNet-Polar (diagonal, after the sum rule, in e)

Same primitive cells as above, 2026-09-24.  ASR = largest component of Σ Z\* before it is removed.

| material | BM1 | BM2 | SevenNet-PS-M |
|---|---|---|---|
| BaTiO₃ cubic | Ba 2.80 / Ti 7.82 / O −2.27, −6.08 (ASR 0.22) | 2.78 / 7.63 / −2.19, −6.04 (0.01) | 2.72 / 7.72 / −2.15, −6.14 (0.04) |
| BaZrO₃ | 2.52 / 5.69 / −1.66, −4.89 (0.17) | 2.59 / 5.70 / −1.69, −4.91 (0.20) | 2.72 / 5.68 / −1.98, −4.44 (0.03) |
| BaHfO₃ | 2.73 / 5.45 / −2.01, −4.16 (0.01) | 2.71 / 5.49 / −1.99, −4.22 (0.03) | 2.74 / 5.42 / −1.99, −4.19 (0.02) |
| ZrO₂ P2₁/c | Zr 5.53, 5.42, 4.91 / O −2.4 to −3.0 (0.21) | 5.55, 5.40, 4.96 / −2.4 to −3.0 (0.33) | 5.54, 5.44, 5.01 / −2.4 to −3.0 (0.10) |
| TiO₂ rutile (out of distribution) | Ti 6.54, 6.54, 7.11 (0.51) | 5.78, 5.78, 8.07 (3.8) | 6.82, 6.82, 7.93 (0.71) |

The three models agree to about 0.1 e inside the training distribution (the O∥ of BaZrO₃ differs by
0.45 e).  BM2 breaks down on rutile TiO₂; BM1 and SevenNet-Polar degrade more gently.

### High-frequency dielectric constant ε∞ (AnisoNet, eigenvalues)

| material | AnisoNet | literature |
|---|---|---|
| Si | 13.14 | exp. 11.7 |
| MgO | 3.13 | exp. 3.0 (static ε₀ 9.8) |
| NaCl | 2.67 | exp. 2.3 |
| BaHfO₃ | 4.69 | 4.6 to 4.9 |
| BaZrO₃ | 4.93 | 4.9 |
| cubic BaTiO₃ | 6.30 | 5.9 to 6.7 (LDA) |
| cubic SrTiO₃ | 6.55 | exp. 5.2, DFT 6.6 |
| m-ZrO₂ | 5.16 / 5.69 / 5.76 | 4.7 to 5.2 |
| rutile TiO₂ | 7.71 / 7.71 / 9.26 | 6.8 / 6.8 / 8.4 |

The predictions lie within 10 to 20 % of the literature and reproduce the direction of the anisotropy.

### Effect of BORNINFO on the phonons (NONANALYTIC 0 → 3)

| material | highest LO at Γ [THz] | soft mode at Γ | note |
|---|---|---|---|
| cubic BaTiO₃ | 13.5 → 19.7 | −6.96 ×3 → −6.96 ×2 + LO 4.89 | the ferroelectric instability remains |
| BaHfO₃ | 14.7 → 18.3 | none | no imaginary mode anywhere (R point 2.2 THz) |
| BaZrO₃ | 14.3 → 19.0 | none | |
| m-ZrO₂ | 21.6 → 23.8 | none | |

### Harmonic thermodynamics (anphon DOS runs)

`run_alamode_phonons.py` also draws `<name>_thermo.png`: C_v(T) against the Dulong–Petit limit 3N k_B,
S(T) and F(T) (zero-point energy included), per primitive cell, from the DOS run with the largest
NONANALYTIC value.  The numbers are in the driver's output after `thermo figure:`.

| material | N | ZPE [meV] | C_v / 3Nk_B at 100 / 300 / 1000 K | T where C_v = 0.9 × 3Nk_B | S(300 K) [k_B] |
|---|---|---|---|---|---|
| Si (DFT reference) | 2 | 121.4 (122.6) | 0.29 / 0.80 / 0.98 (0.31 / 0.80 / 0.98) | 450 K (460 K) | 4.57 (4.72) |
| PbTe (DFT reference) | 2 | 24.7 (25.2) | 0.92 / 0.99 / 1.00 (same) | 90 K (90 K) | 13.30 (13.16) |
| BaHfO₃ | 5 | 249.1 | 0.47 / 0.85 / 0.98 | 390 K | 15.95 |
| BaZrO₃ | 5 | 244.1 | 0.47 / 0.85 / 0.98 | 390 K | 16.35 |
| m-ZrO₂ | 12 | 796.1 | 0.28 / 0.77 / 0.97 | 500 K | 26.27 |

For Si, C_v(300 K) = 0.80 × 3Nk_B = 20.0 J/(mol·K), close to the measured C_p of 20.0 J/(mol·K).
The harmonic C_v approaches 3Nk_B from below and never exceeds it: anharmonicity and thermal expansion (C_p − C_v)
are not included.  anphon leaves imaginary modes out of the sums, so for BaZrO₃ (unstable R-point
rotation) the values miss those modes.

## Other changes since v0.9

- anphon `phonons_mode='dos'` now also gives the output `thermo` (ArrayData: `temperatures` [K],
  `heat_capacity` and `entropy` [k_B], `internal_energy` and `free_energy` [Ry], per primitive cell;
  units in the attribute `units`).

- `alamode.alm_cv`: elastic-net cross validation, `results['alpha_min']`.
- The anphon CalcJob accepts `borninfo`, `fc2xml`, `extra_files` and any `MODE` (SCPH, QHA). The `param` sections are written verbatim, and all `{prefix}.*` files come back in `output_folder`.
- `param['kpoint']` overrides the automatic band path, and `param['interaction']` (NBODY) is kept.
- Fixed: the QE / VASP structure writers of `displace_pf` / `extract`, the cutoff dropped by `alm_opt`, NAT in anphon inputs, the `nsample gridtype` arguments of `analyze_phonons` (ALAMODE ≥ 1.5), `withmpi` on aiida-core ≥ 2.3.
- aiida-lammps is optional. `alm_ALM` and `check_convergence_freeenergy` still need the `alm` Python package.

## Further documentation

| file | contents |
|---|---|
| `docs/born_effective_charges.md` | Z\* and ε∞ in detail, the BaHfO₃ chain, more results |
| `docs/examples_workflow.md` | what each driver does, step by step, and the provenance graph |
| `docs/examples_materials.md` | why each material was chosen and what it tests |
| `docs/remote_gpu_computer.md` | running on a remote GPU node through `core.ssh_async` + slurm, CPU vs GPU timings |
| `docs/slurm_invalidaccount.md` | the slurm InvalidAccount message and its fixes |
| `.claude/skills/aiida-alamode/SKILL.md` | a Claude Code skill summarizing the plugin, the drivers and the pitfalls |
