"""HTML report of an ALAMODE calculation, discovered from the AiiDA provenance graph.

    alamode-report <structure pk>            # everything computed from this structure (its descendants)
    alamode-report <root>/<name>             # a driver run directory: the root is the input cell in .node.json
    alamode-report 14542 -o Si_report.html --json

Starting from a structure node (or any node) the report follows the provenance graph: the structure lineage
upwards (input file -> input cell -> relaxed cell -> primitive cell -> supercells) and every process
downstream (relaxation, alm suggest / opt / cv, displacements, forces or MD, Born charges and eps_inf,
anphon band / DOS / RTA / SCPH / QHA, analyze_phonons, the figure calcfunctions).  Each process is
recognised by its entry point or function name and described from its own inputs and outputs, so the
report needs no driver-specific bookkeeping and works for any process that produced the nodes.

The page contains: the structure (full formula, space group, atoms, cell, Wyckoff sites), what was done
step by step, the relaxation, the force-constant fits, Z* and eps_inf, the Gamma-point frequencies for
every anphon band run (LO-TO splitting), the harmonic thermodynamics, the RTA thermal conductivity, the
SCPH / QHA results, the figures (SVG files from the AiiDA repository embedded inline; PNG as base64 when
no SVG exists: run the driver with --figure-format svg), a process provenance graph (graphviz, if
installed) and the table of the nodes.  Nothing is recomputed.
"""
import base64
import datetime
import html
import io
import json
import os
import re
import sys

import numpy as np

CM1_TO_THZ = 0.0299792458
RY_TO_MEV = 13605.693123
EV_A3_TO_GPA = 160.21766208
AMU_A3_TO_G_CM3 = 1.66053906660

CRYSTAL_SYSTEMS = ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                   (167, "trigonal"), (194, "hexagonal"), (230, "cubic"))

# Pauling electronegativities: the formula is written in increasing order (cations first), like pymatgen
ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.0,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.0, "K": 0.82,
    "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00, "Rb": 0.82,
    "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20,
    "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60, "Cs": 0.79,
    "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20,
    "Tb": 1.10, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30, "Ta": 1.50,
    "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33,
    "Bi": 2.02, "Po": 2.00, "At": 2.20, "Rn": 2.20, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10, "Th": 1.30, "Pa": 1.50,
    "U": 1.38, "Np": 1.36, "Pu": 1.28}


# ------------------------------------------------------------------------------------------ small helpers
def gamma_frequencies(bands_file):
    """the first k point of an anphon .bands file (Gamma on every ASE path), in THz, and the range on the path"""
    return gamma_from_bands_text(open(bands_file).read())


def gamma_from_bands_text(text):
    data = np.loadtxt(io.StringIO(text))
    freqs = data[:, 1:] * CM1_TO_THZ
    gamma = np.sort(freqs[0])
    return {"gamma_THz": [round(float(w), 3) for w in gamma],
            "gamma_highest_THz": round(float(gamma[-1]), 3),
            "path_min_THz": round(float(freqs.min()), 3),
            "path_max_THz": round(float(freqs.max()), 3),
            "imaginary_modes": bool(freqs.min() < -0.05)}


def crystal_system(number):
    for upper, name in CRYSTAL_SYSTEMS:
        if number <= upper:
            return name
    return "?"


def formula(atoms, reduce=True):
    """formula in increasing electronegativity order (Li3PO4, BaHfO3, ZnO); reduce: divide the counts by their gcd"""
    from math import gcd
    from functools import reduce as _reduce
    counts = {}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1
    g = _reduce(gcd, counts.values()) if reduce else 1
    order = sorted(counts, key=lambda s: (ELECTRONEGATIVITY.get(s, 9.0), s))
    return "".join(f"{s}{counts[s] // g if counts[s] // g > 1 else ''}" for s in order)


def describe_structure(structure, symprec=1e-3):
    """formula, symmetry (spglib), cell and sites of a StructureData"""
    import spglib
    atoms = structure.get_ase()
    cell = atoms.cell
    volume = float(atoms.get_volume())
    mass = float(atoms.get_masses().sum())
    info = {"pk": structure.pk, "full_formula": formula(atoms, reduce=False), "reduced_formula": formula(atoms),
            "natoms": len(atoms),
            "a_b_c_A": [round(x, 5) for x in cell.lengths().tolist()],
            "alpha_beta_gamma_deg": [round(x, 3) for x in cell.angles().tolist()],
            "volume_A3": round(volume, 4), "density_g_cm3": round(mass / volume * AMU_A3_TO_G_CM3, 4),
            "cell": [[round(x, 6) for x in row] for row in cell.tolist()]}
    try:
        ds = spglib.get_symmetry_dataset((cell.tolist(), atoms.get_scaled_positions().tolist(),
                                          atoms.get_atomic_numbers().tolist()), symprec=symprec)
        get = (lambda k: ds[k]) if isinstance(ds, dict) else (lambda k: getattr(ds, k))
        number = int(get("number"))
        info["symmetry"] = {"space_group": get("international"), "number": number, "hall": get("hall"),
                            "point_group": get("pointgroup"), "crystal_system": crystal_system(number),
                            "symprec": symprec, "n_operations": len(get("rotations"))}
        wyckoffs = list(get("wyckoffs"))
        equivalent = list(get("equivalent_atoms"))
    except Exception as exc:   # spglib failure must not kill the report
        info["symmetry"] = {"error": str(exc)}
        wyckoffs = [""] * len(atoms)
        equivalent = list(range(len(atoms)))
    sites = []
    for i, (sym, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions())):
        frac = [float(x) % 1.0 for x in pos]
        frac = [0.0 if x > 1 - 1e-5 or x < 1e-5 else round(x, 5) for x in frac]
        sites.append({"index": i + 1, "species": sym, "frac": frac, "wyckoff": wyckoffs[i], "orbit": int(equivalent[i]) + 1})
    info["sites"] = sites
    orbits = {}
    for s in sites:
        key = (s["orbit"], s["species"], s["wyckoff"])
        orbits[key] = orbits.get(key, 0) + 1
    info["wyckoff_positions"] = [f"{sp} {n}{w}" for (_, sp, w), n in sorted(orbits.items())]
    return info


def _fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        if x == 0:
            return "0"
        if abs(x) >= 1e5 or abs(x) < 1e-3:
            return f"{x:.{nd}e}"
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)):
        return ", ".join(_fmt(v, nd) for v in x)
    return html.escape(str(x))


def _value(node):
    """plain Python value of a Data node, or None"""
    if node is None:
        return None
    if hasattr(node, "get_dict"):
        return node.get_dict()
    if hasattr(node, "get_list"):
        return node.get_list()
    if hasattr(node, "value"):
        return node.value
    return None


def _results(proc):
    """the results Dict of a process: output 'results' (current plugin) or 'result' (old versions)"""
    for key in ("results", "result"):
        if key in proc.outputs and hasattr(proc.outputs[key], "get_dict"):
            return proc.outputs[key].get_dict()
    return {}


def _inp(proc, key, default=None):
    if key in proc.inputs:
        v = _value(proc.inputs[key])
        return default if v is None else v
    return default


def _text(node):
    """content of a SinglefileData (str)"""
    return node.get_content() if isinstance(node.get_content(), str) else node.get_content().decode()


# ------------------------------------------------------------------------------------------ discovery
def classify(proc):
    """(kind, sub) of a process node from its entry point / function name and its inputs"""
    t = proc.process_type or ""
    label = proc.process_label or ""
    if "relax_ase" in t or "mattersim_relax" in t:
        return "relax", _inp(proc, "hydrostatic_strain")
    if "forces_ase" in t or t.endswith("alamode.mattersim") or t.endswith("alamode.mattersim_forces"):
        return "forces", None
    if "md_ase" in t or "mattersim_md" in t:
        return "md", None
    if "elastic" in t:
        return "elastic", None
    if "bec_ase" in t or label == "bec_from_values":
        return "bec", None
    if "epsinf_ase" in t:
        return "epsinf", None
    if "borninfo" in t.lower() or label in ("make_borninfo", "BornInfoWorkChain"):
        return "borninfo", None
    if re.search(r"alamode\.alm(_|$)", t):
        return "alm", _inp(proc, "mode", "opt")
    if "alamode.displace" in t:
        return "displace", None
    if "alamode.anphon" in t:
        mode = _inp(proc, "mode", "phonons")
        return "anphon", (mode if mode != "phonons" else _inp(proc, "phonons_mode", "band"))
    if "analyze_phonons" in t:
        return "analyze", _inp(proc, "calc")
    if "img_file" in proc.outputs:
        return "figure", label
    from aiida.orm import StructureData
    if any(isinstance(proc.outputs[k], StructureData) for k in proc.outputs._get_keys()):
        return "structure_op", label
    return "other", label


def structure_role(structure):
    """what a StructureData is, from the process that made it"""
    c = structure.creator
    if c is None:
        return "input cell (stored directly)"
    kind, sub = classify(c)
    label = c.process_label or ""
    if kind == "relax":
        return "relaxed cell (" + ("volume" if sub else "full") + ")"
    if kind == "structure_op":
        if "primitive" in label:
            return "primitive cell (spglib)"
        if "supercell" in label:
            diag = _inp(c, "diag")
            return "supercell" + (f" {'×'.join(str(int(d)) for d in diag)}" if diag else "")
        if "idealize" in label:
            return "idealised cell (symmetrised)"
        if "strain" in label:
            return f"strained cell ({_inp(c, 'component')} {_inp(c, 'magnitude')})"
        return label.replace("_", " ")
    return f"from {label}"


def describe_step(proc, kind, sub):
    """one line saying what a process did, from its inputs and outputs"""
    r = _results(proc)
    calc = _inp(proc, "calculator") or {}
    calc_name = calc.get("name") if isinstance(calc, dict) else None
    model = os.path.basename(str(_inp(proc, "model", ""))) if _inp(proc, "model") else None
    who = f"{calc_name}" + (f" ({model})" if model and calc_name in (None, "mattersim") else "") if calc_name else (model or "")
    if kind == "relax":
        return (f"relax the {'cell volume (hydrostatic strain)' if sub else 'cell and the positions'} with {who}: "
                f"{r.get('nsteps', '?')} steps, fmax {_inp(proc, 'fmax')} eV/Å, "
                f"{'converged' if r.get('converged') else 'NOT converged'}; a, b, c → {_fmt(r.get('cell_lengths'), 4)} Å")
    if kind == "structure_op":
        return structure_role(proc.outputs[next(k for k in proc.outputs._get_keys())])
    if kind == "alm":
        opt = r.get("optimization", {})
        norder = _inp(proc, "norder")
        cut = _inp(proc, "cutoff")
        s = f"alm {sub}, NORDER {norder}, cutoff {json.dumps(cut) if cut else '-'}"
        if sub == "suggest":
            s += f": free IFCs {json.dumps(r.get('num_free_fcs,') or r.get('num_free_fcs'))}, displacement patterns {json.dumps(r.get('num_disp'))}"
        elif sub == "cv":
            s += f": elastic-net cross validation, α_min {r.get('alpha_min')}"
        else:
            s += f": {opt.get('LMODEL', '')} fit of {opt.get('num_param', '?')} parameters ({opt.get('num_free_param', '?')} free)"
            if opt.get("fitting_error") is not None:
                s += f", fitting error {opt['fitting_error']:.4f} %"
            if "fc2xml" in proc.inputs:
                s += ", harmonic IFCs fixed (FC2XML)"
        return s
    if kind == "displace":
        n = r.get("number_of_displacements") or _value(proc.outputs["result"]) if "result" in proc.outputs else r.get("number_of_displacements")
        return f"displaced structures: {n} patterns, displacement {_inp(proc, 'mag')} Å ({r.get('displacement_mode', 'finite displacement')})"
    if kind == "forces":
        n = None
        for key in ("arrays", "results"):
            if key in proc.outputs and hasattr(proc.outputs[key], "get_arraynames") and "energies" in proc.outputs[key].get_arraynames():
                n = int(len(proc.outputs[key].get_array("energies")))
        return f"forces of {n if n is not None else '?'} structures with {who}, {r.get('num_threads', '?')} threads"
    if kind == "md":
        files = r.get("files", [])
        return (f"MD with {who}: {_inp(proc, 'temperature')} K, {_inp(proc, 'nsteps')} steps of {_inp(proc, 'timestep')} fs, "
                f"{len(files)} snapshots ({_inp(proc, 'sample')}) + random displacements {_inp(proc, 'random_mag')} Å")
    if kind == "elastic":
        return f"elastic constants by finite differences (δ {_inp(proc, 'delta')}) with {who}; strained IFC inputs (strain {_inp(proc, 'strain_force_delta')})"
    if kind == "bec":
        if proc.process_label == "bec_from_values":
            return f"Born effective charges from given values {json.dumps(_inp(proc, 'values'))}"
        return f"Born effective charges Z* from the model {calc_name}" + (", ASR enforced" if _inp(proc, "enforce_asr") else "")
    if kind == "epsinf":
        dm = _inp(proc, "dielectric_model") or {}
        return f"dielectric tensor ε∞ from the model {dm.get('name', '?') if isinstance(dm, dict) else dm}"
    if kind == "borninfo":
        return f"BORNINFO file: Z* ({r.get('bec_source', 'model')}) and ε∞ ({r.get('epsilon_inf_source', '?')})"
    if kind == "anphon":
        param = _inp(proc, "param") or {}
        gen = param.get("general", {})
        na = gen.get("NONANALYTIC", 0)
        q = _inp(proc, "qmesh")
        if sub == "band":
            return f"anphon phonon bands, NONANALYTIC {na}" + (", BORNINFO given" if "borninfo" in proc.inputs else "")
        if sub == "dos":
            return f"anphon phonon DOS on a {'×'.join(str(x) for x in q) if q else '?'} q mesh and harmonic thermodynamics, NONANALYTIC {na}"
        if sub == "RTA":
            return f"anphon RTA lattice thermal conductivity on a {'×'.join(str(x) for x in q) if q else '?'} q mesh" + (" with κ spectrum" if _inp(proc, "kappa_spec") else "")
        if sub == "SCPH":
            scph = param.get("scph", {})
            return (f"anphon SCPH {gen.get('TMIN')}–{gen.get('TMAX')} K step {gen.get('DT')} K, KMESH_SCPH {scph.get('KMESH_SCPH')}"
                    + (", structural relaxation (RELAX_STR)" if scph.get("RELAX_STR") else "") + f", NONANALYTIC {na}")
        if sub == "QHA":
            qha = param.get("qha", {})
            return f"anphon QHA scheme {qha.get('QHA_SCHEME')} {gen.get('TMIN')}–{gen.get('TMAX')} K, KMESH_QHA {qha.get('KMESH_QHA')}, RELAX_STR {qha.get('RELAX_STR')}"
        return f"anphon {sub}"
    if kind == "analyze":
        p = _inp(proc, "param") or {}
        return {"tau": "phonon lifetimes", "cumulative": "cumulative κ vs mean free path", "kappa_boundary": "κ with boundary scattering"}.get(sub, f"analyze_phonons {sub}") + (f" at {p.get('temp')} K" if p.get("temp") else "")
    if kind == "figure":
        return f"figure {proc.outputs.img_file.filename}"
    return (proc.process_label or "?").replace("_", " ")


def discover(root_pk):
    """the structure lineage of root (upwards) and every process downstream of root"""
    from aiida.orm import load_node, Node, ProcessNode, StructureData, QueryBuilder
    root = load_node(root_pk)
    # lineage upwards: creator -> its structure input -> its creator ...
    lineage = []
    node = root
    for _ in range(20):
        if isinstance(node, StructureData):
            lineage.append(node)
        c = node.creator
        if c is None:
            break
        nxt = None
        for key in ("structure", "structure_org", "structure_file"):
            if key in c.inputs:
                nxt = c.inputs[key]
                break
        if nxt is None:
            src = c.inputs if isinstance(c, ProcessNode) else {}
            nxt = next((src[k] for k in src._get_keys() if isinstance(src[k], StructureData)), None) if src else None
        if nxt is None:
            break
        node = nxt
    lineage.reverse()
    top = lineage[0] if lineage else root
    qb = QueryBuilder().append(Node, filters={"id": top.pk}, tag="r").append(ProcessNode, with_ancestors="r", project=["*"])
    procs = {n.pk: n for n, in qb.all()}
    if isinstance(root, ProcessNode):
        procs[root.pk] = root
    procs = sorted(procs.values(), key=lambda n: (n.ctime, n.pk))
    return root, lineage, procs


# ------------------------------------------------------------------------------------------ collect
def collect(root_pk, name=None, run_dir=None):
    """everything the report shows, as plain Python (also the MCP tool's return value)"""
    import aiida
    from aiida.manage import get_manager
    if get_manager().get_profile() is None:
        aiida.load_profile(os.environ.get("AIIDA_PROFILE"))
    from aiida.orm import StructureData, CalcJobNode

    root, lineage, procs = discover(root_pk)
    steps = []
    by_kind = {}
    for p in procs:
        kind, sub = classify(p)
        by_kind.setdefault(kind, []).append((p, sub))
        row = {"pk": p.pk, "kind": kind, "sub": sub, "process": p.process_label, "ctime": p.ctime.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
               "state": p.process_state.value if p.process_state else "", "exit": p.exit_status,
               "computer": p.computer.label if isinstance(p, CalcJobNode) and p.computer is not None else None}
        try:
            row["what"] = describe_step(p, kind, sub)
        except Exception as exc:   # a description must never break the report
            row["what"] = f"{p.process_label} ({exc.__class__.__name__}: {exc})"
        t = _results(p).get("timing", {})
        if t.get("elapsed_seconds") is not None:
            row["elapsed_s"] = t["elapsed_seconds"]
        steps.append(row)
    data = {"root_pk": root.pk, "root_type": type(root).__name__, "run_dir": run_dir,
            "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "first_step": steps[0]["ctime"] if steps else None, "last_step": steps[-1]["ctime"] if steps else None,
            "steps": steps, "computers": sorted({s["computer"] for s in steps if s["computer"]})}
    if name is None:
        for p, _ in by_kind.get("figure", []):
            if "name" in p.inputs:
                name = _inp(p, "name")
                break

    # --- structures: the lineage plus every StructureData made downstream
    structures = []
    seen = set()
    made = []
    for p in procs:
        for k in p.outputs._get_keys():
            o = p.outputs[k]
            if isinstance(o, StructureData) and o.pk not in seen:
                made.append(o)
    for s in lineage + sorted(made, key=lambda n: n.ctime):
        if s.pk in seen:
            continue
        seen.add(s.pk)
        d = describe_structure(s)
        d["role"] = structure_role(s)
        d["is_root"] = s.pk == root.pk
        structures.append(d)
    data["structures"] = structures
    prim = next((s for s in structures if s["role"].startswith("primitive")), None)
    if prim is None:   # no find_primitive step: the smallest cell that is not a supercell
        cands = [s for s in structures if not s["role"].startswith("supercell")] or structures
        prim = min(cands, key=lambda s: s["natoms"]) if cands else None
    data["primary"] = prim
    data["formula"] = prim["reduced_formula"] if prim else "?"
    data["name"] = name or data["formula"]
    supercells = [s for s in structures if s["role"].startswith("supercell")]
    if supercells and prim:
        data["supercells"] = [{"pk": s["pk"], "natoms": s["natoms"], "role": s["role"], "multiplicity": round(s["natoms"] / prim["natoms"], 2)} for s in supercells]
    sf = None
    if lineage and lineage[0].creator is not None:
        c = lineage[0].creator
        for k in c.inputs._get_keys():
            if hasattr(c.inputs[k], "filename"):
                sf = c.inputs[k].filename
    data["structure_file"] = sf

    # --- calculator
    calc = {}
    for kind in ("relax", "forces", "md"):
        for p, _ in by_kind.get(kind, []):
            r = _results(p)
            calc = {"name": (_inp(p, "calculator") or {}).get("name"), "model": _inp(p, "model"), "details": r.get("calculator"), "num_threads": r.get("num_threads")}
            break
        if calc:
            break
    data["calculator"] = calc

    # --- relaxation
    relax = []
    for p, sub in by_kind.get("relax", []):
        r = _results(p)
        stress = np.array(r.get("stress", [[0.0]]))
        cell0 = np.array(r["initial_cell"]) if "initial_cell" in r else None
        relax.append({"pk": p.pk, "mode": "volume" if sub else "full", "converged": r.get("converged"), "nsteps": r.get("nsteps"),
                      "fmax_eV_A": r.get("fmax"), "energy_eV": r.get("energy"),
                      "initial_a_b_c_A": np.linalg.norm(cell0, axis=1).round(5).tolist() if cell0 is not None else None,
                      "final_a_b_c_A": [round(x, 5) for x in r.get("cell_lengths", [])],
                      "max_stress_GPa": round(float(np.abs(stress).max()) * EV_A3_TO_GPA, 4),
                      "input_formula": formula(p.inputs.structure.get_ase(), reduce=False) if "structure" in p.inputs else None})
    data["relaxation"] = relax

    # --- force constants
    fits = []
    for p, sub in by_kind.get("alm", []):
        if sub == "suggest":
            continue
        r = _results(p)
        fit = {"pk": p.pk, "mode": sub, "norder": _inp(p, "norder"), "cutoff": _inp(p, "cutoff"),
               "structure": formula(p.inputs.structure.get_ase(), reduce=False) if "structure" in p.inputs else None,
               "ndata": len([l for l in (_inp(p, "dfset") or []) if str(l).startswith("#")]) if isinstance(_inp(p, "dfset"), list) else None,
               "fc2_fixed": "fc2xml" in p.inputs}
        opt = r.get("optimization", {})
        fit.update({k: opt.get(k) for k in ("LMODEL", "num_param", "num_free_param", "fitting_error", "RSS")})
        if "alpha_min" in r:
            fit["alpha_min"] = r["alpha_min"]
            fit["LMODEL"] = "elastic-net CV"
        fits.append(fit)
    data["force_constants"] = fits
    data["suggest"] = [{"pk": p.pk, "norder": _inp(p, "norder"), "cutoff": _inp(p, "cutoff"),
                        "num_free_fcs": _results(p).get("num_free_fcs,") or _results(p).get("num_free_fcs"), "num_disp": _results(p).get("num_disp")}
                       for p, sub in by_kind.get("alm", []) if sub == "suggest"]

    # --- Born charges / eps_inf
    for p, _ in by_kind.get("borninfo", []):
        r = _results(p)
        if "epsilon_inf" in r:
            eps = np.array(r["epsilon_inf"])
            data["borninfo"] = {"pk": p.pk, "symbols": r["symbols"], "bec_diagonal": r["bec_diagonal"],
                                "bec_source": r.get("bec_source") or "model",
                                "asr_residual_max": round(float(np.abs(np.array(r["asr_residual"])).max()), 4),
                                "epsilon_inf_diagonal": np.diag(eps).round(4).tolist(), "epsilon_inf_source": r.get("epsilon_inf_source")}
    if "borninfo" not in data:
        for p, _ in by_kind.get("anphon", []):
            if "borninfo" in p.inputs:
                data["borninfo"] = {"pk": p.inputs.borninfo.pk, "source": "file", "filename": p.inputs.borninfo.filename}
                break

    # --- harmonic phonons: every anphon band run (Gamma frequencies from the .bands file in the repository)
    phonons = []
    for p, sub in by_kind.get("anphon", []):
        if sub != "band" or "phband_file" not in p.outputs:
            continue
        param = _inp(p, "param") or {}
        na = param.get("general", {}).get("NONANALYTIC", 0)
        try:
            g = gamma_from_bands_text(_text(p.outputs.phband_file))
        except Exception as exc:
            g = {"error": str(exc)}
        fcs = p.inputs.fcsxml if "fcsxml" in p.inputs else None
        phonons.append({"pk": p.pk, "NONANALYTIC": na, "prefix": _inp(p, "prefix"), "ifc_file": fcs.filename if fcs is not None else None,
                        "ifc_pk": fcs.pk if fcs is not None else None,
                        "ifc_from": (fcs.creator.process_label if fcs is not None and fcs.creator is not None else "stored file"), **g})
    data["phonons"] = phonons
    # figure summaries give the reference (DFT) values and the DOS integral
    for p, _ in by_kind.get("figure", []):
        if p.process_label == "phonon_figure" and "summary" in p.outputs:
            data["phonon_figure_summary"] = {"pk": p.pk, **p.outputs.summary.get_dict()}
    ms = [ph for ph in phonons if ph.get("ifc_from") != "stored file" and "gamma_highest_THz" in ph]
    if len({ph["NONANALYTIC"] for ph in ms}) > 1:
        lo, hi = min(ms, key=lambda x: x["NONANALYTIC"]), max(ms, key=lambda x: x["NONANALYTIC"])
        data["lo_to_shift_THz"] = {"from": f"NA{lo['NONANALYTIC']}", "to": f"NA{hi['NONANALYTIC']}",
                                   "highest_gamma_mode": [lo["gamma_highest_THz"], hi["gamma_highest_THz"]]}

    # --- thermodynamics from the DOS runs' thermo arrays (per primitive cell)
    thermo = []
    for p, sub in by_kind.get("anphon", []):
        if sub != "dos":
            continue
        a = p.outputs.thermo if "thermo" in p.outputs else None
        if a is None and "thermo_file" in p.outputs:
            try:
                arr = np.loadtxt(io.StringIO(_text(p.outputs.thermo_file)))
                cols = {"temperatures": arr[:, 0], "heat_capacity": arr[:, 1], "entropy": arr[:, 2], "free_energy": arr[:, 4]}
            except Exception:
                cols = None
        elif a is not None:
            cols = {k: a.get_array(k) for k in ("temperatures", "heat_capacity", "entropy", "free_energy")}
        else:
            cols = None
        if not cols:
            continue
        T = cols["temperatures"]
        natom = len(p.inputs.structure.sites) if "structure" in p.inputs else None
        fcs = p.inputs.fcsxml if "fcsxml" in p.inputs else None
        entry = {"pk": p.pk, "NONANALYTIC": (_inp(p, "param") or {}).get("general", {}).get("NONANALYTIC", 0), "natom": natom,
                 "ifc_from": (fcs.creator.process_label if fcs is not None and fcs.creator is not None else "stored file"),
                 "zero_point_energy_meV": round(float(cols["free_energy"][0] * RY_TO_MEV), 3), "at": {}}
        for Tk in (100.0, 300.0, 1000.0):
            i = int(np.argmin(np.abs(T - Tk)))
            if abs(T[i] - Tk) < 1e-6:
                entry["at"][f"{int(Tk)}K"] = {"Cv_kB": round(float(cols["heat_capacity"][i]), 4),
                                              "Cv_over_dulong_petit": round(float(cols["heat_capacity"][i] / (3 * natom)), 4) if natom else None,
                                              "S_kB": round(float(cols["entropy"][i]), 4), "F_meV": round(float(cols["free_energy"][i] * RY_TO_MEV), 3)}
        if natom:
            above = np.where(cols["heat_capacity"] >= 0.9 * 3 * natom)[0]
            entry["T_Cv_90pct_dulong_petit_K"] = float(T[above[0]]) if len(above) else None
        thermo.append(entry)
    data["thermo"] = thermo
    for p, _ in by_kind.get("figure", []):
        if p.process_label == "thermo_figure" and "summary" in p.outputs:
            data["thermo_figure_summary"] = {"pk": p.pk, **p.outputs.summary.get_dict()}

    # --- thermal conductivity from the RTA runs' .kl files
    kappa = []
    for p, sub in by_kind.get("anphon", []):
        if sub != "RTA" or "kl_file" not in p.outputs:
            continue
        try:
            arr = np.loadtxt(io.StringIO(_text(p.outputs.kl_file)))
            T = arr[:, 0]
            diag = arr[:, [1, 5, 9]]
            table = {}
            for Tk in (100, 200, 300, 500, 1000):
                i = int(np.argmin(np.abs(T - Tk)))
                if abs(T[i] - Tk) < 1e-6:
                    table[str(Tk)] = {"xx": round(float(diag[i, 0]), 3), "yy": round(float(diag[i, 1]), 3), "zz": round(float(diag[i, 2]), 3),
                                      "mean": round(float(diag[i].mean()), 3)}
            fcs = p.inputs.fcsxml if "fcsxml" in p.inputs else None
            kappa.append({"pk": p.pk, "qmesh": _inp(p, "qmesh"), "nmodes": _results(p).get("nmodes"), "kappa_WmK": table,
                          "ifc_from": (fcs.creator.process_label if fcs is not None and fcs.creator is not None else "stored file"),
                          "T_range_K": [float(T[0]), float(T[-1])]})
        except Exception as exc:
            kappa.append({"pk": p.pk, "error": str(exc)})
    data["kappa"] = kappa
    for p, _ in by_kind.get("figure", []):
        if p.process_label == "kappa_figure" and "summary" in p.outputs:
            data["kappa_figure_summary"] = {"pk": p.pk, **p.outputs.summary.get_dict()}
    data["analyze"] = [{"pk": p.pk, "calc": sub, **_results(p)} for p, sub in by_kind.get("analyze", [])]

    # --- SCPH / QHA
    for p, sub in by_kind.get("anphon", []):
        if sub == "SCPH":
            param = _inp(p, "param") or {}
            data.setdefault("scph", []).append({"pk": p.pk, "general": param.get("general"), "scph": param.get("scph"), "relax": param.get("relax"),
                                                "files": _results(p).get("files"), "elapsed_s": _results(p).get("timing", {}).get("elapsed_seconds")})
        if sub == "QHA":
            param = _inp(p, "param") or {}
            data.setdefault("qha", []).append({"pk": p.pk, "general": param.get("general"), "qha": param.get("qha"), "files": _results(p).get("files")})
    for p, _ in by_kind.get("figure", []):
        if p.process_label == "scph_figure" and "summary" in p.outputs:
            data["scph_figure_summary"] = {"pk": p.pk, **p.outputs.summary.get_dict()}
        if p.process_label == "qha_figure" and "summary" in p.outputs:
            data["qha_figure_summary"] = {"pk": p.pk, **p.outputs.summary.get_dict()}
    for p, _ in by_kind.get("elastic", []):
        r = _results(p)
        data["elastic"] = {"pk": p.pk, **{k: v for k, v in r.items() if k in ("delta", "strain_force_delta", "volume_bohr3", "soec_Ry", "soec_GPa", "elastic_constants_GPa")}}
    data["md"] = [{"pk": p.pk, "temperature_K": _inp(p, "temperature"), "nsteps": _inp(p, "nsteps"), "timestep_fs": _inp(p, "timestep"),
                   "sample": _inp(p, "sample"), "random_mag_A": _inp(p, "random_mag"), "nsnapshots": len(_results(p).get("files", []))}
                  for p, _ in by_kind.get("md", [])]

    # --- figures: img_file outputs of the figure calcfunctions, SVG preferred per file stem
    figs = {}
    for p, _ in by_kind.get("figure", []):
        for k in p.outputs._get_keys():
            o = p.outputs[k]
            if not k.startswith("img_file") or not hasattr(o, "filename"):
                continue
            stem, ext = os.path.splitext(o.filename)
            ext = ext.lstrip(".").lower()
            if ext not in ("svg", "png"):
                continue
            cur = figs.get(stem)
            if cur is None or (ext == "svg" and cur["format"] != "svg") or (ext == cur["format"] and o.ctime > cur["ctime"]):
                figs[stem] = {"stem": stem, "filename": o.filename, "format": ext, "pk": o.pk, "ctime": o.ctime,
                              "title": p.process_label.replace("_", " "), "process_pk": p.pk, "node": o}
    data["figures"] = [{k: v for k, v in f.items() if k not in ("node", "ctime")} for f in sorted(figs.values(), key=lambda f: f["ctime"])]
    data["_figure_nodes"] = {f["stem"]: f["node"] for f in figs.values()}

    # --- graph
    data["graph_svg"] = process_graph_svg(procs)
    return data


def process_graph_svg(procs):
    """process-only provenance graph (an edge where an output of one process is an input of another), as SVG"""
    try:
        import graphviz
    except ImportError:
        return None
    from aiida.common.links import LinkType
    pks = {p.pk for p in procs}
    g = graphviz.Digraph(graph_attr={"rankdir": "TB", "bgcolor": "transparent", "nodesep": "0.2", "ranksep": "0.35"},
                         node_attr={"fontname": "Helvetica", "fontsize": "9", "margin": "0.05,0.03"},
                         edge_attr={"fontname": "Helvetica", "fontsize": "7", "color": "#888888"})
    for p in procs:
        kind, sub = classify(p)
        color = {"anphon": "#ffd8a8", "alm": "#d0ebff", "forces": "#d3f9d8", "md": "#d3f9d8", "relax": "#d3f9d8",
                 "figure": "#e5dbff", "borninfo": "#fff3bf", "bec": "#fff3bf", "epsinf": "#fff3bf"}.get(kind, "#f1f3f5")
        shape = "box" if p.node_type.endswith("CalcJobNode.") else "ellipse" if "CalcFunctionNode" in p.node_type else "hexagon"
        label = f"{p.process_label}" + (f"\\n{sub}" if sub and kind in ("anphon", "alm", "analyze") else "") + f"\\npk {p.pk}"
        g.node(str(p.pk), label, shape=shape, style="filled", fillcolor=color)
    edges = set()
    for p in procs:
        for link in p.base.links.get_incoming(link_type=(LinkType.INPUT_CALC, LinkType.INPUT_WORK)).all():
            for c in link.node.base.links.get_incoming(link_type=(LinkType.CREATE, LinkType.RETURN)).all():
                if c.node.pk in pks and c.node.pk != p.pk:
                    edges.add((c.node.pk, p.pk, link.link_label))
        for link in p.base.links.get_incoming(link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)).all():
            if link.node.pk in pks:
                edges.add((link.node.pk, p.pk, "CALL"))
    for a, b, lab in sorted(edges):
        g.edge(str(a), str(b), label=lab, style="dashed" if lab == "CALL" else "solid")
    try:
        svg = g.pipe(format="svg").decode()
    except Exception:
        return None
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, flags=re.S)
    svg = re.sub(r"<!--.*?-->\s*", "", svg, flags=re.S)
    return svg.strip()


# ------------------------------------------------------------------------------------------ render
CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d0d7de; --head: #f6f8fa; --accent: #0969da; --warn: #9a3412; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --head: #161b22; --accent: #58a6ff; --warn: #f0883e; } }
:root[data-theme="dark"] { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --head: #161b22; --accent: #58a6ff; --warn: #f0883e; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, "Segoe UI", Roboto, "Noto Sans", Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6em; margin: 0 0 4px; } h1 small { font-weight: normal; color: var(--muted); font-size: 0.6em; margin-left: 8px; }
h2 { font-size: 1.2em; border-bottom: 1px solid var(--line); padding-bottom: 4px; margin: 36px 0 12px; }
h3 { font-size: 1em; margin: 18px 0 6px; color: var(--muted); }
p.meta { color: var(--muted); margin: 0 0 16px; font-size: 0.92em; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.card { flex: 1 1 150px; min-width: 150px; border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; background: var(--head); }
.card .k { font-size: 0.8em; color: var(--muted); } .card .v { font-size: 1.15em; font-weight: 600; word-break: break-word; }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 0.92em; max-width: 100%; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 4px 10px; text-align: left; white-space: nowrap; vertical-align: top; }
th { background: var(--head); font-weight: 600; } td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.wrap { white-space: normal; min-width: 22em; }
.warn { color: var(--warn); font-weight: 600; }
ol.steps { padding-left: 1.6em; } ol.steps li { margin: 3px 0; } ol.steps .pk { color: var(--muted); font-size: 0.85em; }
figure { margin: 12px 0 24px; } figure svg, figure img { max-width: 100%; height: auto; display: block; background: #fff; border-radius: 6px; }
figure.graph svg { background: transparent; }
figcaption { color: var(--muted); font-size: 0.85em; margin-top: 4px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }
details summary { cursor: pointer; color: var(--accent); }
footer { color: var(--muted); font-size: 0.85em; margin-top: 40px; border-top: 1px solid var(--line); padding-top: 8px; }
"""


class _Raw(str):
    """a cell that is already HTML"""


def _table(headers, rows, numeric_from=1, wrap=()):
    out = ["<table><thead><tr>"]
    for i, h in enumerate(headers):
        out.append(f'<th{" class=num" if i >= numeric_from else ""}>{html.escape(str(h))}</th>')
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, v in enumerate(row):
            cls = "num" if i >= numeric_from else ""
            if i in wrap:
                cls = "wrap"
            out.append(f'<td{" class=" + cls if cls else ""}>{v if isinstance(v, _Raw) else _fmt(v)}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _pk(pk):
    return _Raw(f"<code>{pk}</code>") if pk is not None else _Raw("–")


def _embed_figure(fig, node):
    raw = node.get_content(mode="rb") if "mode" in node.get_content.__code__.co_varnames else node.get_content()
    if isinstance(raw, str):
        raw = raw.encode("utf-8") if fig["format"] == "svg" else raw.encode("latin-1")
    if fig["format"] == "svg":
        text = raw.decode("utf-8")
        text = re.sub(r"<\?xml[^>]*\?>\s*", "", text)
        text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text, flags=re.S)
        body = text.strip()
        cap = f"{html.escape(fig['filename'])} (SVG from the AiiDA repository, node {fig['pk']}, made by {html.escape(fig['title'])} pk {fig['process_pk']})"
    else:
        body = f'<img alt="{html.escape(fig["title"])}" src="data:image/png;base64,{base64.b64encode(raw).decode()}">'
        cap = f"{html.escape(fig['filename'])} (PNG, node {fig['pk']}; run the driver with --figure-format svg for a vector figure)"
    return f"<figure>{body}<figcaption>{cap}</figcaption></figure>"


def render(data):
    """the HTML page"""
    d = data
    fig_nodes = d.get("_figure_nodes", {})
    figs = {f["stem"]: f for f in d.get("figures", [])}
    h = []
    w = h.append
    prim = d.get("primary") or {}
    sym = prim.get("symmetry", {})
    calc = d.get("calculator") or {}
    calc_name = calc.get("name") or "?"
    kinds = {s["kind"] for s in d["steps"]}
    subs = {(s["kind"], s["sub"]) for s in d["steps"]}
    what = "phonons"
    if ("anphon", "SCPH") in subs:
        what = "SCPH"
    elif ("anphon", "QHA") in subs:
        what = "QHA"
    elif ("anphon", "RTA") in subs:
        what = "phonons and thermal conductivity"
    title = f"{d['formula']} {what}"
    w('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">')
    w(f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>")
    w(f"<h1>{html.escape(d['formula'])} <small>{html.escape(d['name'])} · {html.escape(what)} · from node {d['root_pk']}</small></h1>")
    w(f"<p class=meta>{len(d['steps'])} processes from {d.get('first_step') or '?'} to {d.get('last_step') or '?'}"
      + (f" · run directory <code>{html.escape(d['run_dir'])}</code>" if d.get("run_dir") else "") + f" · report {d['generated']}</p>")

    # --- cards
    cards = []
    if prim:
        cards.append(("Full formula (primitive cell)", prim["full_formula"]))
        cards.append(("Space group", f"{sym.get('space_group', '?')} (No. {sym.get('number', '?')})" if "number" in sym else "?"))
        cards.append(("Crystal system / point group", f"{sym.get('crystal_system', '?')} / {sym.get('point_group', '?')}"))
        cards.append(("Atoms (primitive)", str(prim["natoms"])))
        if prim.get("wyckoff_positions"):
            cards.append(("Wyckoff positions", ", ".join(prim["wyckoff_positions"])))
    if d.get("supercells"):
        s = d["supercells"][0]
        cards.append(("Supercell atoms", f"{s['natoms']} (×{s['multiplicity']:g})" + (f" + {len(d['supercells']) - 1} more" if len(d["supercells"]) > 1 else "")))
    model = calc.get("model")
    cards.append(("Force calculator", calc_name + (f" ({os.path.basename(str(model))})" if model else "")))
    if d.get("computers"):
        cards.append(("Computer", ", ".join(d["computers"])))
    if "lo_to_shift_THz" in d:
        lo = d["lo_to_shift_THz"]
        cards.append((f"Highest Γ mode {lo['from']} → {lo['to']}", f"{lo['highest_gamma_mode'][0]:.2f} → {lo['highest_gamma_mode'][1]:.2f} THz"))
    elif d.get("phonons"):
        ph = d["phonons"][-1]
        if "gamma_highest_THz" in ph:
            cards.append((f"Highest Γ mode (NA{ph['NONANALYTIC']})", f"{ph['gamma_highest_THz']:.2f} THz"))
    if d.get("kappa") and d["kappa"][0].get("kappa_WmK", {}).get("300"):
        cards.append(("κ (RTA, 300 K)", f"{d['kappa'][0]['kappa_WmK']['300']['mean']:.1f} W/mK"))
    if d.get("thermo"):
        cards.append(("Zero-point energy", f"{d['thermo'][-1]['zero_point_energy_meV']:.1f} meV / cell"))
    if d.get("scph_figure_summary"):
        s = d["scph_figure_summary"]
        cards.append(("Polar structure up to", f"{s['T_polar_max_K']} K" if s.get("T_polar_max_K") else "none (stays centrosymmetric)"))
    imag = [ph for ph in d.get("phonons", []) if ph.get("imaginary_modes")]
    if imag:
        cards.append(("Imaginary modes", f"yes ({', '.join('NA' + str(p['NONANALYTIC']) for p in imag)})"))
    w("<div class=cards>" + "".join(f"<div class=card><div class=k>{html.escape(k)}</div><div class=v>{html.escape(v)}</div></div>" for k, v in cards) + "</div>")

    # --- structure
    w("<h2>Structure</h2>")
    rows = []
    for s in d["structures"]:
        ss = s.get("symmetry", {})
        rows.append([_Raw(html.escape(s["role"]) + (" <b>(root)</b>" if s["is_root"] else "")), _pk(s["pk"]), s["full_formula"], s["natoms"],
                     f"{ss.get('space_group', '?')} ({ss.get('number', '?')})", _fmt(s["a_b_c_A"], 4), _fmt(s["alpha_beta_gamma_deg"], 2),
                     s["volume_A3"], s["density_g_cm3"]])
    w(_table(["cell", "pk", "formula", "atoms", "space group", "a, b, c [Å]", "α, β, γ [°]", "V [Å³]", "ρ [g/cm³]"], rows, numeric_from=3))
    if d.get("structure_file"):
        w(f"<p class=meta>input structure file: <code>{html.escape(d['structure_file'])}</code></p>")

    # --- relaxation
    if d.get("relaxation"):
        w("<h2>Relaxation</h2>")
        w(_table(["cell", "mode", "pk", "converged", "steps", "fmax [eV/Å]", "E [eV]", "a, b, c before [Å]", "a, b, c after [Å]", "max |σ| after [GPa]"],
                 [[r["input_formula"], r["mode"], _pk(r["pk"]), _Raw("yes" if r["converged"] else '<span class=warn>no</span>'), r["nsteps"], r["fmax_eV_A"],
                   r["energy_eV"], _fmt(r["initial_a_b_c_A"], 4), _fmt(r["final_a_b_c_A"], 4), r["max_stress_GPa"]] for r in d["relaxation"]], numeric_from=4))
        det = calc.get("details") or {}
        if det:
            w(f"<p class=meta>calculator: <code>{html.escape(str(det.get('module', '')))}.{html.escape(str(det.get('callable', '')))}</code> "
              f"{html.escape(json.dumps(det.get('kwargs', {})))}, threads {calc.get('num_threads')}</p>")

    # --- MD
    if d.get("md"):
        w("<h2>Molecular dynamics (anharmonic training set)</h2>")
        w(_table(["pk", "T [K]", "steps", "Δt [fs]", "sampled steps", "snapshots", "random displacement [Å]"],
                 [[_pk(m["pk"]), m["temperature_K"], m["nsteps"], m["timestep_fs"], m["sample"], m["nsnapshots"], m["random_mag_A"]] for m in d["md"]]))

    # --- force constants
    if d.get("force_constants") or d.get("suggest"):
        w("<h2>Force constants (alm)</h2>")
        if d.get("suggest"):
            w(_table(["suggest pk", "NORDER", "cutoff [Bohr]", "free IFCs", "displacement patterns"],
                     [[_pk(s["pk"]), s["norder"], _Raw(html.escape(json.dumps(s["cutoff"]))), _Raw(html.escape(json.dumps(s["num_free_fcs"]))),
                       _Raw(html.escape(json.dumps(s["num_disp"])))] for s in d["suggest"]]))
        rows = []
        for f in d["force_constants"]:
            rows.append([f["mode"], _pk(f["pk"]), f["structure"], f["norder"], _Raw(html.escape(json.dumps(f["cutoff"])) if f.get("cutoff") else "–"),
                         f.get("ndata"), _Raw("yes" if f["fc2_fixed"] else "no"), f.get("LMODEL"), f.get("num_param"), f.get("num_free_param"),
                         _Raw(f"{f['fitting_error']:.4f} %") if f.get("fitting_error") is not None else _Raw("–"), f.get("RSS"), f.get("alpha_min")])
        if rows:
            w(_table(["step", "pk", "supercell", "NORDER", "cutoff [Bohr]", "structures in DFSET", "FC2 fixed", "model", "params", "free", "fitting error", "RSS", "α_min (CV)"], rows, numeric_from=3))

    # --- Born charges
    if "borninfo" in d:
        b = d["borninfo"]
        w("<h2>Born effective charges and ε∞ (LO-TO correction)</h2>")
        if b.get("source") == "file":
            w(f"<p>BORNINFO given as a stored file <code>{html.escape(str(b.get('filename')))}</code> (pk {b['pk']}).</p>")
        else:
            w(_table(["species", "Z* xx", "Z* yy", "Z* zz"], [[s, *z] for s, z in zip(b["symbols"], b["bec_diagonal"])]))
            w(f"<p class=meta>Z* source: <b>{html.escape(str(b['bec_source']))}</b>, max acoustic-sum-rule residual {b['asr_residual_max']} e · "
              f"ε∞ diagonal {_fmt(b['epsilon_inf_diagonal'], 3)} from <b>{html.escape(str(b['epsilon_inf_source']))}</b> · BORNINFO pk {b['pk']}</p>")

    # --- phonons
    if d.get("phonons"):
        w("<h2>Harmonic phonons</h2>")
        rows = []
        for ph in d["phonons"]:
            if "error" in ph:
                rows.append([f"NA{ph['NONANALYTIC']}", _pk(ph["pk"]), ph["ifc_from"], _Raw(f'<span class=warn>{html.escape(ph["error"])}</span>'), None, None, None, None])
                continue
            rows.append([f"NA{ph['NONANALYTIC']}", _pk(ph["pk"]), ph["ifc_from"], _fmt(ph["gamma_THz"], 2), ph["gamma_highest_THz"], ph["path_min_THz"],
                         ph["path_max_THz"], _Raw('<span class=warn>yes</span>' if ph["imaginary_modes"] else "no")])
        w(_table(["NONANALYTIC", "anphon pk", "IFCs from", "Γ frequencies [THz]", "highest Γ [THz]", "min on path [THz]", "max on path [THz]", "imaginary"], rows, numeric_from=3))
        if "lo_to_shift_THz" in d:
            lo = d["lo_to_shift_THz"]
            w(f"<p>LO-TO splitting: the highest Γ mode moves from {lo['highest_gamma_mode'][0]:.3f} THz ({lo['from']}) to {lo['highest_gamma_mode'][1]:.3f} THz ({lo['to']}).</p>")
        fs = d.get("phonon_figure_summary")
        if fs:
            rows = []
            for na, s in sorted((k, v) for k, v in fs.items() if k.startswith("NA")):
                for who, label in (("mattersim", calc_name), ("ref", "reference")):
                    if f"{who}_gamma_THz" in s:
                        rows.append([na, label, _fmt(sorted(s[f"{who}_gamma_THz"]), 2), s[f"{who}_max_THz"], s.get(f"dos_integral_{who}")])
            w("<h3>Comparison in the figure (reference IFCs, e.g. DFT, through the same anphon step)</h3>")
            w(_table(["NONANALYTIC", "IFCs", "Γ frequencies [THz]", "max [THz]", "∫DOS (3 × atoms)"], rows, numeric_from=2))
        if "phband_phdos" in figs or any(k.endswith("phband_phdos") for k in figs):
            key = next(k for k in figs if k.endswith("phband_phdos"))
            w(_embed_figure(figs[key], fig_nodes[key]))

    # --- thermodynamics
    if d.get("thermo"):
        w("<h2>Harmonic thermodynamics</h2>")
        rows = []
        for t in d["thermo"]:
            at = t["at"]
            g = lambda k, f: at[k][f] if k in at else None
            rows.append([f"NA{t['NONANALYTIC']}", _pk(t["pk"]), t["ifc_from"], t["natom"], t["zero_point_energy_meV"],
                         g("100K", "Cv_over_dulong_petit"), g("300K", "Cv_over_dulong_petit"), g("1000K", "Cv_over_dulong_petit"),
                         g("100K", "S_kB"), g("300K", "S_kB"), g("1000K", "S_kB"), g("100K", "F_meV"), g("300K", "F_meV"), g("1000K", "F_meV"),
                         t.get("T_Cv_90pct_dulong_petit_K")])
        w(_table(["NONANALYTIC", "anphon pk", "IFCs from", "atoms", "ZPE [meV]", "C_v/3Nk_B 100 K", "300 K", "1000 K", "S [k_B] 100 K", "300 K", "1000 K",
                  "F [meV] 100 K", "300 K", "1000 K", "T(C_v = 0.9·3Nk_B) [K]"], rows, numeric_from=3))
        w("<p class=meta>per primitive cell, from the anphon DOS run's thermo output (harmonic; F includes the zero-point energy)</p>")
        ts = d.get("thermo_figure_summary")
        if ts and "ref" in ts:
            r = ts["ref"]
            at = r["at"]
            w(f"<p class=meta>reference IFCs in the figure: ZPE {r['zero_point_energy_meV']:.1f} meV, C_v/3Nk_B at 100 / 300 / 1000 K = "
              f"{at['100K']['Cv_over_dulong_petit']:.3f} / {at['300K']['Cv_over_dulong_petit']:.3f} / {at['1000K']['Cv_over_dulong_petit']:.3f}, "
              f"S(300 K) {at['300K']['S_kB']:.2f} k_B</p>")
        key = next((k for k in figs if k.endswith("_thermo")), None)
        if key:
            w(_embed_figure(figs[key], fig_nodes[key]))

    # --- kappa
    if d.get("kappa"):
        w("<h2>Lattice thermal conductivity (RTA)</h2>")
        temps = ["100", "200", "300", "500", "1000"]
        rows = []
        for k in d["kappa"]:
            if "error" in k:
                rows.append([_pk(k["pk"]), _Raw(f'<span class=warn>{html.escape(k["error"])}</span>')])
                continue
            rows.append([_pk(k["pk"]), k["ifc_from"], "×".join(str(x) for x in k["qmesh"]) if k.get("qmesh") else "?", k.get("nmodes"),
                         *[k["kappa_WmK"][T]["mean"] if T in k["kappa_WmK"] else None for T in temps],
                         _fmt([k["kappa_WmK"]["300"][c] for c in ("xx", "yy", "zz")], 2) if "300" in k["kappa_WmK"] else "–"])
        w(_table(["anphon pk", "IFCs from", "q mesh", "modes", *[f"κ({T} K) [W/mK]" for T in temps], "κ_xx, κ_yy, κ_zz (300 K)"], rows, numeric_from=3))
        ks = d.get("kappa_figure_summary")
        if ks:
            rows = []
            for who, label in (("ms", calc_name), ("ref", "reference")):
                if who in ks:
                    v = ks[who]
                    rows.append([label, *[v["kappa_WmK"].get(T) for T in temps], v.get("L50_nm"), v.get("kappa_boundary_1mm_WmK"), v.get("spectrum_integral_WmK")])
            w("<h3>Comparison in the figure (analyze_phonons at 300 K)</h3>")
            w(_table(["IFCs", *[f"κ({T} K)" for T in temps], "L₅₀ [nm]", "κ with 1 mm boundary", "∫κ spectrum"], rows))
            w("<p class=meta>L₅₀: mean free path below which half of κ accumulates</p>")
        key = next((k for k in figs if k.endswith("_kappa")), None)
        if key:
            w(_embed_figure(figs[key], fig_nodes[key]))

    # --- SCPH
    if d.get("scph"):
        w("<h2>SCPH (self-consistent phonons)</h2>")
        for s in d["scph"]:
            gen, sc = s.get("general") or {}, s.get("scph") or {}
            w(f"<p>anphon SCPH pk {s['pk']}: {gen.get('TMIN')}–{gen.get('TMAX')} K step {gen.get('DT')} K, KMESH_SCPH {sc.get('KMESH_SCPH')}, "
              f"KMESH_INTERPOLATE {sc.get('KMESH_INTERPOLATE')}, MAXITER {sc.get('MAXITER')}, MIXALPHA {sc.get('MIXALPHA')}"
              + (", structural relaxation (RELAX_STR = %s)" % sc.get("RELAX_STR") if sc.get("RELAX_STR") else "") + (f"; {s['elapsed_s']:.0f} s" if s.get("elapsed_s") else "") + "</p>")
        fs = d.get("scph_figure_summary")
        if fs:
            w(f"<p>B-site atom {html.escape(str(fs.get('B_site')))}; highest temperature with a polar structure: "
              f"<b>{fs['T_polar_max_K'] if fs.get('T_polar_max_K') else 'none'}</b>" + (" K" if fs.get("T_polar_max_K") else "") + f" (figure calcfunction pk {fs['pk']}).</p>")
            rows = [[T, u, F] for T, u, F in zip(fs["T_K"], fs.get("Ti_z_Bohr", []), fs.get("F_total_meV", []))]
            w(_table(["T [K]", f"{fs.get('B_site')} z displacement [Bohr]", "F [meV / cell]"], rows, numeric_from=0))
        key = next((k for k in figs if k.endswith("_scph_relax")), None)
        if key:
            w(_embed_figure(figs[key], fig_nodes[key]))

    # --- QHA
    if d.get("qha"):
        w("<h2>QHA (quasi-harmonic thermal expansion)</h2>")
        w(_table(["anphon pk", "QHA_SCHEME", "T range [K]", "KMESH_QHA", "RELAX_STR"],
                 [[_pk(q["pk"]), (q.get("qha") or {}).get("QHA_SCHEME"), f"{(q.get('general') or {}).get('TMIN')}–{(q.get('general') or {}).get('TMAX')}",
                   (q.get("qha") or {}).get("KMESH_QHA"), (q.get("qha") or {}).get("RELAX_STR")] for q in d["qha"]]))
        fs = d.get("qha_figure_summary")
        if fs:
            q = {k: v for k, v in fs.items() if k != "pk"}
            schemes = list(q)
            temps = q[schemes[0]]["T_K"] if schemes else []
            rows = [[T, *[_fmt(x, 5) for sch in schemes for x in (q[sch]["u_xx"][i], q[sch]["u_zz"][i])]] for i, T in enumerate(temps)]
            w(_table(["T [K]", *[f"{sch} {c}" for sch in schemes for c in ("u_xx", "u_zz")]], rows, numeric_from=0))
            w(f"<p class=meta>thermal strain relative to the 0 K relaxed cell (figure calcfunction pk {fs['pk']})</p>")
        if "elastic" in d:
            e = d["elastic"]
            w(f"<h3>Elastic constants (pk {e['pk']})</h3><details><summary>values</summary><pre>"
              + html.escape(json.dumps({k: v for k, v in e.items() if k != "pk"}, indent=1, default=str)[:6000]) + "</pre></details>")
        key = next((k for k in figs if k.endswith("_thermal_strain")), None)
        if key:
            w(_embed_figure(figs[key], fig_nodes[key]))

    # --- other figures not shown above
    shown = {k for k in figs if any(k.endswith(suf) for suf in ("phband_phdos", "_thermo", "_kappa", "_scph_relax", "_thermal_strain"))}
    rest = [k for k in figs if k not in shown]
    if rest:
        w("<h2>Other figures</h2>")
        for k in rest:
            w(_embed_figure(figs[k], fig_nodes[k]))

    # --- nodes
    w("<h2>Processes</h2>")
    w("<details><summary>" + f"{len(d['steps'])} process nodes</summary>")
    rows = [[_pk(s["pk"]), s["process"], s["kind"] + (f" / {s['sub']}" if s.get("sub") not in (None, "", True, False) else ""), s["state"], s.get("exit"),
             s.get("computer") or "", s["ctime"], _Raw(html.escape(s["what"]))] for s in d["steps"]]
    w(_table(["pk", "process", "kind", "state", "exit", "computer", "created", "what"], rows, numeric_from=99, wrap=(7,)))
    w("</details>")
    if d.get("graph_svg"):
        w(f"<details><summary>process graph</summary><figure class=graph>{d['graph_svg']}<figcaption>boxes: CalcJobs, ellipses: calcfunctions; an edge: an output of one is an input of the other</figcaption></figure></details>")
    w(f"<footer>aiida-alamode report · generated {d['generated']} from node {d['root_pk']} ({html.escape(d['root_type'])})"
      + (f" · <code>{html.escape(d['run_dir'])}</code>" if d.get("run_dir") else "") + "; <code>verdi node graph generate &lt;pk&gt;</code> draws the full provenance of any node</footer>")
    w("</main></body></html>")
    return "\n".join(h)


# ------------------------------------------------------------------------------------------ entry points
def root_of_run_dir(run_dir):
    """the root node of a driver run: the input cell recorded in .node.json (unit0), else the structure file"""
    node_json = os.path.join(run_dir, ".node.json")
    if not os.path.isfile(node_json):
        raise FileNotFoundError(f"{node_json} not found: give <root>/<name> of a run, or a node pk")
    with open(node_json) as f:
        pks = json.load(f)
    for key in ("unit0", "structure_file", "prim", "structure"):
        if key in pks:
            return pks[key]
    raise KeyError(f"{node_json} records no structure (keys: {list(pks)})")


def write_report(target, out=None):
    """target: a run directory or a node pk; returns the path of the HTML file
    (default <run_dir>/<name>_report.html, or ./<formula>_pk<pk>_report.html for a pk)"""
    if isinstance(target, int) or str(target).isdigit():
        data = collect(int(target))
        out = out or f"{data['formula']}_pk{data['root_pk']}_report.html"
    else:
        run_dir = os.path.abspath(target)
        data = collect(root_of_run_dir(run_dir), name=os.path.basename(run_dir.rstrip("/")), run_dir=run_dir)
        out = out or os.path.join(run_dir, f"{data['name']}_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(data))
    return out


def public(data):
    """the collected data without node objects and the graph (JSON-serialisable)"""
    return {k: v for k, v in data.items() if k not in ("_figure_nodes", "graph_svg")}


def summary(data):
    """the key numbers of a report in a few hundred characters (for an LLM); every value with the pk it came from.
    The full collected data is public(data); the MCP writes it next to the HTML as <name>_report.json"""
    d = data
    prim = d.get("primary") or {}
    sym = prim.get("symmetry", {})
    s = {"root_pk": d["root_pk"], "formula": d.get("formula"), "name": d.get("name"), "run_dir": d.get("run_dir"),
         "processes": len(d["steps"]), "first_step": d.get("first_step"), "last_step": d.get("last_step"),
         "failed": [{"pk": x["pk"], "process": x["process"], "state": x["state"], "exit": x.get("exit")} for x in d["steps"]
                    if x["state"] != "finished" or x.get("exit") not in (0, None)],
         "kinds": sorted({x["kind"] + (f"/{x['sub']}" if x.get("sub") not in (None, "", True, False) else "") for x in d["steps"]})}
    if prim:
        s["primary"] = {"pk": prim["pk"], "full_formula": prim["full_formula"], "natoms": prim["natoms"], "role": prim.get("role"),
                        "space_group": sym.get("space_group"), "number": sym.get("number"), "wyckoff_positions": prim.get("wyckoff_positions"),
                        "a_b_c_A": prim.get("a_b_c_A"), "alpha_beta_gamma_deg": prim.get("alpha_beta_gamma_deg")}
    if d.get("supercells"):
        s["supercells"] = [{"pk": x["pk"], "natoms": x["natoms"], "role": x["role"]} for x in d["supercells"]]
    calc = d.get("calculator") or {}
    if calc:
        s["calculator"] = {"name": calc.get("name"), "model": os.path.basename(str(calc["model"])) if calc.get("model") else None}
    if d.get("relaxation"):
        s["relaxation"] = [{"pk": r["pk"], "mode": r["mode"], "converged": r["converged"], "nsteps": r["nsteps"],
                            "final_a_b_c_A": r["final_a_b_c_A"]} for r in d["relaxation"]]
    if d.get("force_constants"):
        s["force_constants"] = [{k: fc.get(k) for k in ("pk", "mode", "norder", "ndata", "LMODEL", "num_free_param", "fitting_error", "alpha_min") if fc.get(k) is not None}
                                for fc in d["force_constants"]]
    if d.get("borninfo"):
        b = d["borninfo"]
        per = {}
        for sym_, z in zip(b.get("symbols", []), b.get("bec_diagonal", [])):
            per.setdefault(sym_, []).append(sum(z) / 3)
        s["borninfo"] = {"pk": b.get("pk"), "bec_source": b.get("bec_source"), "mean_Z_per_species": {k: round(sum(v) / len(v), 3) for k, v in per.items()},
                         "epsilon_inf_diagonal": b.get("epsilon_inf_diagonal"), "epsilon_inf_source": b.get("epsilon_inf_source")}
    if d.get("phonons"):
        s["phonons"] = [{"pk": p["pk"], "NONANALYTIC": p["NONANALYTIC"], "gamma_highest_THz": p.get("gamma_highest_THz"),
                         "path_max_THz": p.get("path_max_THz"), "imaginary_modes": p.get("imaginary_modes")} for p in d["phonons"]]
    if "lo_to_shift_THz" in d:
        s["lo_to_shift_THz"] = d["lo_to_shift_THz"]
    if d.get("thermo"):
        s["thermo"] = [{"pk": t["pk"], "NONANALYTIC": t["NONANALYTIC"], "zero_point_energy_meV": t["zero_point_energy_meV"],
                        "at": t.get("at"), "T_Cv_90pct_dulong_petit_K": t.get("T_Cv_90pct_dulong_petit_K")} for t in d["thermo"]]
    if d.get("kappa"):
        s["kappa"] = [{"pk": k["pk"], "qmesh": k.get("qmesh"), "kappa_mean_WmK": {T: v["mean"] for T, v in k.get("kappa_WmK", {}).items()}} for k in d["kappa"]]
    if d.get("scph_figure_summary"):
        f = d["scph_figure_summary"]
        s["scph"] = {"pk": f["pk"], "B_site": f.get("B_site"), "T_polar_max_K": f.get("T_polar_max_K"), "T_K": f.get("T_K"),
                     "B_site_displacement_Bohr": f.get(f"{f.get('B_site')}_z_Bohr")}
    if d.get("qha_figure_summary"):
        f = d["qha_figure_summary"]
        s["qha"] = {"pk": f["pk"]}
        for scheme, v in f.items():
            if isinstance(v, dict) and "T_K" in v:
                s["qha"][scheme] = {"T_max_K": v["T_K"][-1], **{k: x[-1] for k, x in v.items() if k != "T_K" and isinstance(x, list) and x}}
    if d.get("elastic"):
        s["elastic"] = {"pk": d["elastic"]["pk"]}
    if d.get("md"):
        s["md"] = d["md"]
    s["figures"] = [{"pk": f["pk"], "filename": f["filename"]} for f in d.get("figures", [])]
    return s


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="a structure (or any node) pk, or <root>/<name> of a driver run (contains .node.json)")
    parser.add_argument("-o", "--output", help="HTML file (default <run_dir>/<name>_report.html or ./<formula>_pk<pk>_report.html)")
    parser.add_argument("--json", action="store_true", help="print the collected data as JSON instead of writing HTML")
    parser.add_argument("--summary", action="store_true", help="print the key numbers as JSON instead of writing HTML")
    args = parser.parse_args(argv)
    if args.json or args.summary:
        target = args.target
        if str(target).isdigit():
            data = collect(int(target))
        else:
            run_dir = os.path.abspath(target)
            data = collect(root_of_run_dir(run_dir), name=os.path.basename(run_dir.rstrip("/")), run_dir=run_dir)
        json.dump(summary(data) if args.summary else public(data), sys.stdout, indent=1, default=str)
        print()
        return
    print(write_report(args.target, args.output))


if __name__ == "__main__":
    main()
