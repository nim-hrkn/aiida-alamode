# Copyright 2022 Hiori Kino
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ASE-calculator runner (MatterSim by default; also sevennet, sevennet-polar, mace, chgnet, orb, emt, ...):
compute total energy, forces and stress of displaced structures, relax cells, run MD, elastic constants,
Born effective charges.

This script runs on the computer side (inside a scheduler job) and must not import aiida.

usage: alamode-ase-runner job.json

job.json (mode "forces", the default):
    {"files": ["disp1.pw.in", ...], "input_format": "espresso-in",
     "calculator": {"name": "mattersim", "kwargs": {"load_path": "MatterSim-v1.0.0-1M.pth", "device": "auto"}},
     "output": "ase_results.json"}

"calculator" selects the ASE calculator:
    {"name": <one of CALCULATORS>, "kwargs": {...}}                      known calculators, or
    {"module": "mace.calculators", "callable": "mace_mp", "kwargs": {...}} any importable class / factory.
kwargs named "device" / "use_device" with the value "auto" become "cuda" if available, otherwise "cpu".
Without "calculator", the old keys "model" and "device" select MatterSim.

The output json contains, for each file in the same order,
    energy [eV], forces [eV/Angstrom], stress [eV/Angstrom^3] (3x3, ASE sign convention),
    positions [Angstrom], cell [Angstrom], symbols.

job.json (mode "relax"): relax the cell and the positions.
    {"mode": "relax", "files": ["structure.cif"], "input_format": "cif",
     "fmax": 1e-4, "steps": 500, "hydrostatic_strain": true,
     "model": ..., "device": ..., "output": "ase_results.json"}

The output json contains the relaxed structure in the same items as above and
    "nsteps", "converged", "initial_cell".

job.json (mode "md"): NVT (Langevin) molecular dynamics of the supercell, snapshots sampled as
displace.py -md ... -e start:end:interval --random --mag, i.e. a random displacement of a fixed
magnitude in a random direction is added to every atom of every sampled snapshot.
    {"mode": "md", "files": ["supercell.extxyz"], "input_format": "extxyz",
     "temperature": 300.0, "timestep_fs": 1.0, "nsteps": 5000, "sample": "1001:5000:50",
     "friction": 0.01, "random_mag": 0.04, "random_seed": 1, "disp_prefix": "disp",
     "calculator": ..., "output": "ase_results.json"}
The sampled, displaced structures are written as {disp_prefix}{NN}.extxyz (extended xyz)
and the trajectory (every interval steps) as md_traj.extxyz.

job.json (mode "bec"): Born effective charges (and the dielectric tensor when the model provides it).
    {"mode": "bec", "files": ["primitive.extxyz"], "input_format": "extxyz",
     "calculator": {"name": "sevennet-polar", "kwargs": {"model": ".../SevenNet-PS-M.pth"}},
     "output": "ase_results.json"}
The result has "born_effective_charges" (nat x 3 x 3, e; the calculator's convention, rows as VASP
BORN EFFECTIVE CHARGES) and "dielectric_tensor" (3 x 3, or null).

job.json (mode "elastic"): clamped-ion elastic constants and strain-force coupling of a cell,
for the anphon QHA structural optimization (STRAIN_IFC_DIR files elastic_constants.in and strain_force.in).
    {"mode": "elastic", "files": ["primitive.extxyz"], "input_format": "extxyz",
     "delta": 0.01, "strain_force_delta": 0.005, "calculator": ..., "output": "ase_results.json"}
The cell is deformed as h' = (1 + u) h with the fractional coordinates fixed; U(u) is the energy.
    V C_{ij}   = d^2 U / du_i du_j       (i, j over xx, xy, xz, yx, yy, yz, zx, zy, zz; 81 values, Ry)
    V C_{ijk}  = d^3 U / du_i du_j du_k  (729 values, Ry)
by central finite differences with the step delta; strain_force.in has the forces [Ry/Bohr] for
u_xx = u_yy = u_zz = strain_force_delta and u_yz = u_zx = u_xy = strain_force_delta / 2 (weight 1.0).
"""
import json
import os
import sys
import time

import numpy as np


def _set_num_threads():
    import torch
    nthreads = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if nthreads:
        torch.set_num_threads(int(nthreads))
    return torch.get_num_threads()


def _get_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# name -> (module, callable, default kwargs). The callable is a class or a factory function.
CALCULATORS = {
    "mattersim": ("mattersim.forcefield", "MatterSimCalculator",
                  {"load_path": "MatterSim-v1.0.0-1M.pth", "device": "auto"}),
    "mace": ("mace.calculators", "mace_mp", {"model": "medium", "device": "auto"}),
    "mace-off": ("mace.calculators", "mace_off", {"model": "medium", "device": "auto"}),
    "chgnet": ("chgnet.model", "CHGNetCalculator", {"use_device": "auto"}),
    "sevennet": ("sevenn.calculator", "SevenNetCalculator", {"model": "7net-0", "device": "auto"}),
    # SevenNet-Polar (github.com/AugustinLu/SevenNet-Polar): Born effective charges; checkpoints on
    # zenodo 10.5281/zenodo.21322761 (PS-*: BEC only, Ba Ca Hf Li O P Pb Sr Ti Zr; PM-*: + energy/forces, Li O P Zr)
    "sevennet-polar": ("sevenn.calculator", "SevenNetCalculator",
                       {"model": os.environ.get("SEVENNET_POLAR_MODEL",
                                                os.path.expanduser("~/models/sevennet-polar/SevenNet-PS-M.pth")),
                        "device": "auto"}),
    "orb": ("aiida_alamode.ase_runner", "_orb_calculator", {"model": "orb_v3_conservative_inf_omat", "device": "auto"}),
    "emt": ("ase.calculators.emt", "EMT", {}),
    "lj": ("ase.calculators.lj", "LennardJones", {}),
}


def _orb_calculator(model="orb_v3_conservative_inf_omat", device="cpu", **kwargs):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    orbff = getattr(pretrained, model)(device=device, **kwargs)
    return ORBCalculator(orbff, device=device)


def calculator_spec(job: dict) -> dict:
    """normalized {"module", "callable", "kwargs", "label"} from job["calculator"] (or model/device)."""
    spec = dict(job.get("calculator") or {"name": "mattersim",
                                          "kwargs": {"load_path": job.get("model", "MatterSim-v1.0.0-1M.pth"),
                                                     "device": job.get("device", "auto")}})
    kwargs = dict(spec.get("kwargs") or {})
    if "name" in spec:
        name = spec["name"].lower()
        if name not in CALCULATORS:
            raise ValueError(f"unknown calculator {name}. known: {sorted(CALCULATORS)}; "
                             "or give module / callable.")
        module, callable_, defaults = CALCULATORS[name]
        kwargs = {**defaults, **kwargs}
    else:
        name, module, callable_ = spec.get("label", spec["callable"]), spec["module"], spec["callable"]
    for key in ("device", "use_device"):
        if kwargs.get(key) == "auto":
            kwargs[key] = _get_device("auto")
    return {"name": name, "module": module, "callable": callable_, "kwargs": kwargs}


def make_calculator(job: dict):
    import importlib
    spec = calculator_spec(job)
    factory = getattr(importlib.import_module(spec["module"]), spec["callable"])
    return factory(**spec["kwargs"]), spec


def _atoms_to_dict(atoms, filename: str) -> dict:
    return {"filename": filename,
            "energy": float(atoms.get_potential_energy()),
            "forces": atoms.get_forces().tolist(),
            "stress": atoms.get_stress(voigt=False).tolist(),
            "positions": atoms.get_positions().tolist(),
            "cell": atoms.cell.array.tolist(),
            "symbols": atoms.get_chemical_symbols()}


def _write_extxyz_copy(filename: str, atoms):
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    copy = atoms.copy()
    copy.calc = SinglePointCalculator(copy, energy=atoms.get_potential_energy(), forces=atoms.get_forces(),
                                      stress=atoms.get_stress())
    ase.io.write(filename, copy, format="extxyz")


def _relax(atoms, job: dict) -> dict:
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    fmax = float(job.get("fmax", 1e-4))
    steps = int(job.get("steps", 500))
    initial_cell = atoms.cell.array.tolist()
    filtered = FrechetCellFilter(atoms, hydrostatic_strain=bool(job.get("hydrostatic_strain", True)))
    opt = BFGS(filtered, logfile="relax.log")
    converged = opt.run(fmax=fmax, steps=steps)
    return {"nsteps": opt.get_number_of_steps(), "converged": bool(converged),
            "fmax": fmax, "initial_cell": initial_cell}


def _random_displacements(rng, nat: int, mag: float):
    """each atom: |u| = mag [A], random direction (as GenDisplacement._get_random_displacements 'gauss')"""
    d = rng.normal(size=(nat, 3))
    return d / np.linalg.norm(d, axis=1)[:, None] * mag


def _md(atoms, job: dict) -> dict:
    """Langevin NVT MD; returns the sampled + randomly displaced structures and a summary."""
    import ase.io
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    temperature = float(job.get("temperature", 300.0))
    dt = float(job.get("timestep_fs", 1.0))
    nsteps = int(job.get("nsteps", 5000))
    start, end, interval = [int(x) for x in job.get("sample", "1001:5000:50").split(":")]
    mag = float(job.get("random_mag", 0.04))
    prefix = job.get("disp_prefix", "disp")
    rng = np.random.default_rng(job.get("random_seed", 1))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, dt * units.fs, temperature_K=temperature,
                   friction=float(job.get("friction", 0.01)) / units.fs, rng=rng, logfile="md.log")
    sampled = list(range(start, end + 1, interval))
    files, temps, epots = [], [], []
    traj = open("md_traj.extxyz", "w")
    for step in range(1, nsteps + 1):
        dyn.run(1)
        if step % interval == 0:
            temps.append(float(atoms.get_temperature()))
            epots.append(float(atoms.get_potential_energy()))
            ase.io.write(traj, atoms, format="extxyz")
        if step in sampled:
            snapshot = atoms.copy()
            snapshot.set_positions(snapshot.get_positions() + _random_displacements(rng, len(atoms), mag))
            filename = f"{prefix}{len(files) + 1:03d}.extxyz"
            snapshot.calc = None
            ase.io.write(filename, snapshot, format="extxyz")
            files.append(filename)
            print(f"step {step}: T = {temps[-1]:.1f} K, E = {epots[-1]:.4f} eV -> {filename}", flush=True)
    traj.close()
    return {"files": files, "nsnapshots": len(files), "temperature": temperature, "timestep_fs": dt,
            "nsteps": nsteps, "sample": job.get("sample", "1001:5000:50"), "random_mag": mag,
            "temperatures": temps, "potential_energies": epots,
            "mean_temperature": float(np.mean(temps[len(temps) // 5:])) if temps else None}


_STRAIN_COMPONENTS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
_BOHR = 0.5291772108
_RYDBERG = 13.60569253


def _strain_matrix(components: dict) -> np.ndarray:
    u = np.zeros((3, 3))
    for key, value in components.items():
        i, j = "xyz".index(key[0]), "xyz".index(key[1])
        u[i, j] += value
    return u


def _deformed(atoms, u: np.ndarray):
    """h' = (1 + u) h with fixed fractional coordinates (rows of the cell array are the lattice vectors)."""
    deformed = atoms.copy()
    deformed.set_cell(atoms.cell.array @ (np.eye(3) + u).T, scale_atoms=True)
    return deformed


def _elastic(atoms, job: dict) -> dict:
    """clamped-ion SOEC / TOEC and strain-force coupling by finite differences of the energy / forces."""
    from itertools import combinations_with_replacement
    calc = atoms.calc
    delta = float(job.get("delta", 0.01))
    delta_f = float(job.get("strain_force_delta", 0.005))
    cache = {}

    def energy(**components):
        key = tuple(sorted((k, round(v, 12)) for k, v in components.items() if v != 0.0))
        if key not in cache:
            d = _deformed(atoms, _strain_matrix(components))
            d.calc = calc
            cache[key] = float(d.get_potential_energy())
        return cache[key]

    n = len(_STRAIN_COMPONENTS)
    soec = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = _STRAIN_COMPONENTS[i], _STRAIN_COMPONENTS[j]
            if i == j:
                soec[i, i] = (energy(**{a: delta}) - 2 * energy() + energy(**{a: -delta})) / delta ** 2
            else:
                soec[i, j] = (energy(**{a: delta, b: delta}) - energy(**{a: delta, b: -delta})
                              - energy(**{a: -delta, b: delta}) + energy(**{a: -delta, b: -delta})) / (4 * delta ** 2)
                soec[j, i] = soec[i, j]

    def d3(i, j, k):
        """third derivative by the 8-point central difference (works for repeated indices too:
        the components are summed in _strain_matrix)."""
        total = 0.0
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    comps = {}
                    for idx, sgn in ((i, si), (j, sj), (k, sk)):
                        comps[_STRAIN_COMPONENTS[idx]] = comps.get(_STRAIN_COMPONENTS[idx], 0.0) + sgn * delta
                    total += si * sj * sk * energy(**comps)
        return total / (8 * delta ** 3)

    toec = np.zeros((n, n, n))
    for i, j, k in combinations_with_replacement(range(n), 3):
        value = d3(i, j, k)
        for p in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            toec[p] = value

    strain_force = []
    for comp, mag in (("xx", delta_f), ("yy", delta_f), ("zz", delta_f),
                      ("yz", delta_f), ("zx", delta_f), ("xy", delta_f)):
        if comp[0] == comp[1]:
            u = _strain_matrix({comp: mag})
        else:
            u = _strain_matrix({comp: mag / 2, comp[::-1]: mag / 2})
        d = _deformed(atoms, u)
        d.calc = calc
        forces = d.get_forces() * _BOHR / _RYDBERG   # eV/A -> Ry/Bohr
        strain_force.append({"component": comp, "magnitude": mag, "weight": 1.0, "forces": forces.tolist()})

    lines = ["SOEC"] + [f"{v / _RYDBERG:.10e}" for v in soec.ravel()] + \
            ["TOEC"] + [f"{v / _RYDBERG:.10e}" for v in toec.ravel()]
    with open("elastic_constants.in", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("strain_force.in", "w") as f:
        for entry in strain_force:
            f.write(f"{entry['component']} {entry['magnitude']} {entry['weight']}\n")
            for fx, fy, fz in entry["forces"]:
                f.write(f"{fx:.6f}  {fy:.6f}  {fz:.6f}\n")
    volume = float(atoms.get_volume()) / _BOHR ** 3
    return {"delta": delta, "strain_force_delta": delta_f, "volume_bohr3": volume,
            "soec_Ry": (soec / _RYDBERG).tolist(), "toec_Ry": (toec / _RYDBERG).tolist(),
            "soec_GPa": (soec / float(atoms.get_volume()) * 160.21766).tolist(),
            "strain_force": strain_force, "n_energy_evaluations": len(cache)}


def _bec(atoms) -> dict:
    """Born effective charges of every atom (and the dielectric tensor if the calculator gives one)."""
    calc = atoms.calc
    calc.calculate(atoms, properties=["born_effective_charges"])
    results = calc.results
    if "born_effective_charges" not in results:
        raise ValueError(f"the calculator {type(calc).__name__} does not provide born_effective_charges.")
    bec = np.asarray(results["born_effective_charges"], dtype=float).reshape(len(atoms), 3, 3)
    eps = results.get("dielectric_tensor")
    out = {"symbols": atoms.get_chemical_symbols(), "positions": atoms.get_positions().tolist(),
           "cell": atoms.cell.array.tolist(), "born_effective_charges": bec.tolist(),
           "dielectric_tensor": np.asarray(eps, dtype=float).reshape(3, 3).tolist() if eps is not None else None,
           "asr_residual": bec.sum(axis=0).tolist()}
    if "energy" in results:
        out["energy"] = float(results["energy"])
    for i, (s, z) in enumerate(zip(atoms.get_chemical_symbols(), bec)):
        print(f"{i + 1:4d} {s:2s} Z* diag = {np.round(np.diag(z), 3)}", flush=True)
    print("acoustic sum rule residual:", np.round(bec.sum(axis=0), 4).tolist(), flush=True)
    return out


def run(job: dict) -> dict:
    import ase.io

    nthreads = _set_num_threads()
    input_format = job.get("input_format", "espresso-in")
    mode = job.get("mode", "forces")

    t0 = time.time()
    calc, spec = make_calculator(job)
    t_load = time.time() - t0
    print(f"calculator: {spec}", flush=True)

    if mode == "md":
        atoms = ase.io.read(job["files"][0], format=input_format)
        atoms.calc = calc
        md = _md(atoms, job)
        md.update({"mode": mode, "calculator": spec, "num_threads": nthreads,
                   "time_model_load": t_load, "time_total": time.time() - t0, "structures": []})
        return md
    if mode == "bec":
        atoms = ase.io.read(job["files"][0], format=input_format)
        atoms.calc = calc
        result = _bec(atoms)
        result.update({"mode": mode, "calculator": spec, "num_threads": nthreads,
                       "time_model_load": t_load, "time_total": time.time() - t0, "structures": []})
        return result
    if mode == "elastic":
        atoms = ase.io.read(job["files"][0], format=input_format)
        atoms.calc = calc
        result = _elastic(atoms, job)
        result.update({"mode": mode, "calculator": spec, "num_threads": nthreads,
                       "time_model_load": t_load, "time_total": time.time() - t0, "structures": []})
        return result

    structures = []
    extra = {}
    for filename in job["files"]:
        atoms = ase.io.read(filename, format=input_format)
        atoms.calc = calc
        if mode == "relax":
            extra = _relax(atoms, job)
        elif mode != "forces":
            raise ValueError(f"unknown mode={mode}.")
        structures.append(_atoms_to_dict(atoms, filename))
        # keep a human-readable copy (energy, forces, stress only: calculators such as SevenNet-Polar leave
        # per-atom 3x3 tensors in the results, which the extxyz writer cannot store)
        _write_extxyz_copy(f"{os.path.splitext(filename)[0]}.calc.extxyz", atoms)
        forces = atoms.get_forces()
        print(f"{filename}: E = {atoms.get_potential_energy():.6f} eV, "
              f"|F|max = {abs(forces).max():.4e} eV/A, "
              f"a,b,c = {atoms.cell.lengths()}", flush=True)

    result = {"mode": mode, "calculator": spec, "num_threads": nthreads,
              "time_model_load": t_load, "time_total": time.time() - t0,
              "structures": structures}
    result.update(extra)
    return result


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        job = json.load(f)
    result = run(job)
    with open(job.get("output", "ase_results.json"), "w") as f:
        json.dump(result, f)
    print("JOB DONE.", flush=True)


if __name__ == "__main__":
    main()
