# aiida-alamode: notes for working in this repository

## Prerequisite outside the repository: `../alamode_test/`

The preset drivers compare the ML-potential results with the DFT reference data of the ALAMODE
tutorial. They read it from `alamode_test/` **next to this repository** (not inside it):

```
example/run_alamode_phonons.py: ALAMODE_TEST = os.path.join(HERE, "..", "..", "alamode_test")
```

| driver | reads |
|---|---|
| `run_alamode_phonons.py --preset Si` | `alamode_test/Si/reference/si222.xml`, `si222_cubic.xml.bz2` |
| `run_alamode_phonons.py --preset PbTe` | `alamode_test/PbTe/reference/super444_0.01.xml`, `PbTe.born` |
| `run_alamode_scph.py` (BaTiO3) | `alamode_test/BaTiO3/scph_relax/reference/` |
| `run_alamode_qha.py` (ZnO) | `alamode_test/ZnO/qha_relax/reference/` |

`alamode_test/` is the `example/` directory of the ALAMODE 1.5.0 source archive
(`alamode-v1.5.0.zip` -> `ttadano-alamode-*/example/`: BaTiO3, PbTe, Si, Si_LAMMPS, Si_OpenMX, SrTiO3, ZnO).
On a new host, unpack it there before running a preset:

```
cd <parent of this repository>
unzip -q ~/alamode-v1.5.0.zip 'ttadano-alamode-*/example/*'
mv ttadano-alamode-*/example alamode_test && rm -r ttadano-alamode-*
```

Without it the preset drivers fail when they reach the reference step (the ML part itself does not
need it; `--structure FILE` runs and `run_alamode_scph.py --no-ref` do not read it).

## Everyday commands

- Install for development: `pip install -e . --no-deps`; after changing entry points or plugin code,
  `verdi daemon restart`.
- Optional ML packages (MatterSim, SevenNet-Polar, AnisoNet, Equivar): see README section 2;
  `alamode-ase-runner --check` or `python example/check_packages.py --computer X` reports what a computer has.
- Drivers: `example/run_alamode_phonons.py`, `run_alamode_scph.py`, `run_alamode_qha.py`;
  `--computer <label>` (or `AIIDA_ALAMODE_COMPUTER`) selects the codes `alm@<label>`, `anphon@<label>`, ...
  Results and the `.node.json` cache go under `example/run_*/<name>/`.
- Remote computers (core.ssh_async + slurm): `docs/remote_gpu_computer.md`. Use `--gpu` only where a GPU exists.
- MCP server for LLM use: `docs/mcp_server.md`; agent skill: `.claude/skills/aiida-alamode/SKILL.md`.
- HTML report: `alamode-report <run dir | pk>` (`aiida_alamode/report.py`), discovered from the provenance graph.
  **When a CalcJob, calcfunction or WorkChain (or its input / output keys) is added or changed, update `report.py`**
  (`classify`, `describe_step`, the matching part of `collect`) and re-run the report on a phonons, LO-TO, SCPH and QHA run.
- `example/legacy_v0.8/` holds the 2022-2023 notebooks (LAMMPS / QE route, v0.9 free-energy convergence; old package name, removed entry points, missing script).
  They are reference material, not runnable examples; do not build on them.

## Conventions

- Host names and IP addresses stay out of the repository (docs are shared); use labels such as `host` / `gpu-node`.
- Commit author email: hkino@ism.ac.jp (repo-local git config).
