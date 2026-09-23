"""BaTiO3 (alamode tutorial 7.1 + 7.4) with an ASE calculator instead of DFT, through AiiDA:
anharmonic IFCs from MD + random displacements + elastic-net CV, then SCPH structural optimization.

    relax (volume)  ->  primitive / 2x2x2 supercell  ->  harmonic IFCs (alm_suggest / displace_pf / forces / alm_opt)
      ->  MD at T (md_ase: sampled snapshots + random 0.04 A)  ->  forces + DFSET (alamode.forces)
      ->  alm cv (alm_cv: NORDER = 3, NBODY 2 3 3, LASSO, CV = 4)  ->  alm opt (L1_ALPHA of the minimum CV score)
      ->  anphon SCPH + RELAX_STR = 1 (T = TMIN..TMAX)  ->  figure (atomic displacements, free energies)

usage:
    python run_alamode_scph.py                       # cubic BaTiO3, tutorial settings
    python run_alamode_scph.py --structure X.cif --supercell 2 2 2 --name X --init-disp "1 0 0 0.002" ...
    python run_alamode_scph.py --calculator mace --calculator-kwargs '{"model": "medium"}'

The helpers (NodeBank, calcfunctions, ...) are shared with run_alamode_phonons.py.
codes: alm, anphon, displace, ase_runner @<computer>.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ase.io
import ase.formula
from ase import Atoms

from run_alamode_phonons import (NodeBank, wait, run_cached, read_structure, find_primitive, idealize_structure,
                                 make_supercell_structure, submit_alm, submit_displace, submit_forces, submit_anphon,
                                 submit_ase, HERE, ALAMODE_TEST, BOHR)
from aiida.engine import calcfunction, submit
from aiida.orm import load_code, Str, Dict, Float, Int, List, Bool
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_alamode.io.alm_input import AlmPrefixMaker

StructureData = DataFactory('core.structure')
SinglefileData = DataFactory('core.singlefile')
FolderData = DataFactory('core.folder')

RY_TO_MEV = 13605.693
REF_DIR = os.path.join(ALAMODE_TEST, "BaTiO3", "scph_relax", "reference")


def cubic_batio3(a=3.9855493692679786):
    """cubic perovskite BaTiO3 (PBEsol a of the tutorial). O order: 2 planar (x, y) then the apical (z),
    as the tutorial's &displace pattern (Ti +z, planar O -z, apical O -2z) assumes."""
    return Atoms(symbols=["Ba", "Ti", "O", "O", "O"], cell=np.eye(3) * a, pbc=True,
                 scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]])


@calcfunction
def scph_figure(cwd: Str, name: Str, calc_label: Str, output_folder: FolderData, prefix: Str, structure: StructureData,
                ref_atom_disp: SinglefileData = None, ref_thermo: SinglefileData = None) -> dict:
    """atomic displacements (z) and free energies vs T from the SCPH RELAX_STR run; tutorial reference dashed."""
    def load(text):
        return np.loadtxt(text.splitlines())
    disp = load(output_folder.get_object_content(f"{prefix.value}.atom_disp"))
    thermo = load(output_folder.get_object_content(f"{prefix.value}.scph_thermo"))
    # temperatures where the structure iteration diverged (nan) are dropped
    disp = disp[np.isfinite(disp).all(axis=1)]
    thermo = thermo[np.isfinite(thermo).all(axis=1)]
    disp = disp[np.argsort(disp[:, 0])]
    thermo = thermo[np.argsort(thermo[:, 0])]
    nat = (disp.shape[1] - 1) // 3
    symbols = [site.kind_name for site in structure.sites]   # ABO3: A, B, O(1), O(2), O(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    labels = {0: f"{symbols[0]}(z)", 1: f"{symbols[1]}(z)", 2: f"{symbols[2]}(1,2, z)", 4: f"{symbols[4]}(3, z)"} if nat >= 5 \
        else {i: f"{sym}(z)" for i, sym in enumerate(symbols)}
    for i, lab in labels.items():
        if i < nat:
            ax.plot(disp[:, 0], disp[:, 3 + 3 * i], color=f"C{i}", lw=1.5, label=f"{lab} {calc_label.value}")
    if ref_atom_disp is not None:
        ref = load(ref_atom_disp.get_content())
        ref = ref[np.argsort(ref[:, 0])]
        for i, lab in labels.items():
            ax.plot(ref[:, 0], ref[:, 3 + 3 * i], color=f"C{i}", lw=1.2, ls="--", label=f"{lab} DFT reference" if i == 1 else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Atomic displacement (Bohr)")
    ax.set_title(f"{name.value} SCPH structural relaxation", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    T = thermo[:, 0]
    curves = [("total", thermo[:, 5], "C0"), ("U0", thermo[:, 4], "C1"), ("F_vib", thermo[:, 2] + thermo[:, 3], "C2")]
    for lab, y, c in curves:
        ax.plot(T, y * RY_TO_MEV, color=c, lw=1.5, label=f"{lab} {calc_label.value}")
    if ref_thermo is not None:
        rt = load(ref_thermo.get_content())
        for lab, y, c in [("total", rt[:, 5], "C0"), ("U0", rt[:, 4], "C1"), ("F_vib", rt[:, 2] + rt[:, 3], "C2")]:
            ax.plot(rt[:, 0], y * RY_TO_MEV, color=c, lw=1.2, ls="--", label="DFT reference" if lab == "total" else None)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Free energy (meV / primitive cell)")
    ax.set_title(f"{name.value} SCPH free energy", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = os.path.join(cwd.value, f"{name.value}_scph_relax.png")
    fig.savefig(target, dpi=150)
    plt.close(fig)
    # transition temperature: the highest T whose B-site displacement is a sizable fraction of the
    # low-T (saturated) value; the seed displacement of the high-symmetry phase is ~1e-3 Bohr
    ti_z = np.abs(disp[:, 6]) if nat > 1 else np.zeros(len(disp))
    polar = disp[ti_z > max(0.005, 0.25 * ti_z.max()), 0]
    summary = {"T_K": disp[:, 0].tolist(), "B_site": symbols[1] if nat > 1 else None,
               "Ti_z_Bohr": disp[:, 6].tolist() if nat > 1 else [],
               "T_polar_max_K": float(polar.max()) if len(polar) else None,
               "F_total_meV": (thermo[:, 5] * RY_TO_MEV).tolist()}
    return {"img_file": SinglefileData(target), "summary": Dict(summary)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", help="structure file of the high-symmetry cell (default: cubic BaTiO3 written here)")
    parser.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2], metavar="N")
    parser.add_argument("--name", help="run name (default BaTiO3 or the formula, + _calculator)")
    parser.add_argument("--calculator", default="mattersim")
    parser.add_argument("--calculator-kwargs", default="{}")
    parser.add_argument("--calc-label")
    parser.add_argument("--relax", choices=["volume", "none"], default="volume")
    parser.add_argument("--mag", type=float, default=0.01, help="harmonic displacement [A]")
    # MD + random displacements (tutorial: 300 K, 1 fs, 5000 steps, 1001:5000:50 -> 80 configurations, 0.04 A)
    parser.add_argument("--md-temperature", type=float, default=300.0)
    parser.add_argument("--md-timestep", type=float, default=1.0)
    parser.add_argument("--md-steps", type=int, default=5000)
    parser.add_argument("--md-sample", default="1001:5000:50")
    parser.add_argument("--random-mag", type=float, default=0.04)
    parser.add_argument("--random-seed", type=int, default=1)
    # anharmonic fit (tutorial: NORDER 3, NBODY 2 3 3, cutoff None 15 9 Bohr, LASSO CV 4, alpha 1e-8..1e-2 x 30)
    parser.add_argument("--nbody", default="2 3 3")
    parser.add_argument("--cutoff", type=float, nargs=2, default=[15.0, 9.0], metavar="BOHR", help="cubic, quartic")
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--cv-alpha", type=float, nargs=3, default=[1.0e-8, 0.01, 30], metavar="X", help="min max n")
    parser.add_argument("--l1-alpha", type=float, help="skip the CV and use this L1_ALPHA")
    # SCPH (tutorial: 50..400 K step 25, KMESH_SCPH 4, KMESH_INTERPOLATE 2, k mesh 8; MatterSim needs TMAX 700:
    # with TMAX 400 the structure loop at 75 K did not converge in 1000 iterations)
    parser.add_argument("--tmin", type=float, default=100.0, help="lowest T; at 50 K the MatterSim BaTiO3 SCPH structure loop can diverge (anphon aborts with std::length_error)")
    parser.add_argument("--tmax", type=float, default=700.0, help="the cooling must start well above T_c (MatterSim: ~350-400 K)")
    parser.add_argument("--dt", type=float, default=50.0)
    parser.add_argument("--kmesh-scph", type=int, default=4)
    parser.add_argument("--kmesh-interpolate", type=int, default=2)
    parser.add_argument("--qmesh", type=int, default=8)
    parser.add_argument("--mixbeta-coord", type=float, default=0.2, help="&relax MIXBETA_COORD (smaller is more stable)")
    parser.add_argument("--max-str-iter", type=int, default=1000)
    parser.add_argument("--add-hess-diag", type=float, default=0.0, help="&relax ADD_HESS_DIAG [cm^-1] (larger is more stable)")
    parser.add_argument("--mixalpha", type=float, default=0.2, help="&scph MIXALPHA (tutorial 0.2, anphon default 0.1)")
    parser.add_argument("--maxiter", type=int, default=500, help="&scph MAXITER")
    parser.add_argument("--init-disp", default="0 0 1.5e-5; 0 0 2.5e-3; 0 0 -1.9e-3; 0 0 -1.9e-3; 0 0 -3.8e-3",
                        help="&displace initial displacements per atom [Bohr], ';' separated (tutorial values)")
    parser.add_argument("--no-ref", action="store_true", help="do not plot the tutorial's DFT reference")
    parser.add_argument("--borninfo", help="BORNINFO file (dielectric tensor and Born charges of the primitive cell)")
    parser.add_argument("--borninfo-calculator", metavar="NAME",
                        help="compute the Born charges with this calculator (alamode.bec_ase, e.g. sevennet-polar)")
    parser.add_argument("--borninfo-kwargs", default="{}", help="JSON kwargs of --borninfo-calculator")
    parser.add_argument("--dielectric-model", metavar="NAME",
                        help="predict eps_inf with this model when the Born-charge calculator has none (e.g. anisonet)")
    parser.add_argument("--dielectric", type=float, nargs="+", metavar="E", help="dielectric tensor for --borninfo-calculator")
    parser.add_argument("--nonanalytic", type=int, default=3, help="NONANALYTIC of the SCPH run when Born charges are given")
    parser.add_argument("--computer", default=os.environ.get("AIIDA_ALAMODE_COMPUTER", "localhost"),
                        help="AiiDA computer label of the codes alm@..., anphon@..., ... (env AIIDA_ALAMODE_COMPUTER)")
    parser.add_argument("--gpu", action="store_true", help="request one GPU (#SBATCH --gres=gpu:1) for the MatterSim / SevenNet jobs")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--njobs", type=int, default=2)
    parser.add_argument("--root", default=os.path.join(HERE, "run_alamode_scph"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)
    args.calculator_kwargs = json.loads(args.calculator_kwargs)
    args.borninfo_kwargs = json.loads(args.borninfo_kwargs)
    if args.calc_label is None:
        args.calc_label = "MatterSim-v1.0.0-1M" if args.calculator == "mattersim" and not args.calculator_kwargs \
            else args.calculator + ("" if not args.calculator_kwargs else " " + json.dumps(args.calculator_kwargs))
    if args.structure is None:
        args.structure = os.path.join(HERE, "BaTiO3_Pm-3m.cif")
        if not os.path.isfile(args.structure):
            ase.io.write(args.structure, cubic_batio3(), format="cif")
    return args


def main():
    args = parse_args()
    input_atoms = ase.io.read(args.structure)
    formula = ase.formula.Formula(input_atoms.get_chemical_formula()).reduce()[0].format("metal")
    name = args.name or formula + ("" if args.calculator == "mattersim" else f"_{args.calculator}")
    root = os.path.join(args.root, name)
    dirs = {k: os.path.join(root, k) for k in ["relax", "harmonic", "md", "anharmonic", "scph"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    bank = NodeBank(root, args.force)
    print(f"{name}: {args.structure} ({len(input_atoms)} atoms), supercell {args.supercell}")

    code_alm = load_code(f"alm@{args.computer}")
    code_anphon = load_code(f"anphon@{args.computer}")
    code_displace = load_code(f"displace@{args.computer}")
    code_ase = load_code(f"ase_runner@{args.computer}")   # the alamode-ase-runner script
    opt_calc = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": args.cores},
                "max_wallclock_seconds": 4 * 3600}
    if args.gpu:
        opt_calc["custom_scheduler_commands"] = "#SBATCH --gres=gpu:1"
    opt_serial = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 2},
                  "max_wallclock_seconds": 3600}
    if args.gpu:   # relax / Born charges / elastic constants also on the GPU
        opt_serial["custom_scheduler_commands"] = "#SBATCH --gres=gpu:1"
    opt_anphon = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": args.cores},
                  "max_wallclock_seconds": 12 * 3600,
                  "environment_variables": {"OMP_NUM_THREADS": str(args.cores)}}

    calc_spec = {"name": args.calculator, "kwargs": args.calculator_kwargs}
    calculator = run_cached(bank, "calculator", lambda: Dict(calc_spec))
    if calculator.get_dict() != calc_spec:
        raise SystemExit(f"{root} was made with {calculator.get_dict()}; use another --name.")

    # --- structures
    structure_file = run_cached(bank, "structure_file", lambda: SinglefileData(os.path.abspath(args.structure)))
    unit0 = run_cached(bank, "unit0", lambda: read_structure(structure_file, Str("")))
    if args.relax == "none":
        unit = unit0
    else:
        relax = run_cached(bank, "relax_volume",
                           lambda: submit_ase("relax_ase", code_ase, unit0, calculator, opt_serial,
                                                    cwd=Str(dirs["relax"]), hydrostatic_strain=Bool(True)))
        unit = relax.outputs.structure
        print(f"relaxed cell [A]: {np.round(relax.outputs.results['cell_lengths'], 5)} (input {np.round(unit0.cell_lengths, 5)})")
    symprec = Float(1e-3)
    # the relaxed cell is symmetry-idealized (alm finds 0 free IFCs on a cell with ~1e-6 noise);
    # the supercells are built from this primitive cell, as in the tutorial.
    prim = run_cached(bank, "prim", lambda: idealize_structure(unit, symprec))
    unit = prim
    supercell = run_cached(bank, "supercell", lambda: make_supercell_structure(unit, List(args.supercell)))
    print(f"primitive cell: {[s.kind_name for s in prim.sites]}, supercell: {len(supercell.sites)} atoms")

    # --- harmonic IFCs (tutorial 7.1 step 1; the same steps as run_alamode_phonons.py)
    norder1 = Int(1)
    prefix1 = Str(AlmPrefixMaker(name=name, kmesh=args.supercell, norder=1).prefix)
    cwd_h = Str(dirs["harmonic"])
    alm_suggest = run_cached(bank, "alm_suggest", lambda: submit_alm(code_alm, "suggest", supercell, prefix1, norder1, cwd_h))
    displace = run_cached(bank, "displace",
                          lambda: submit_displace(code_displace, supercell, alm_suggest.outputs.pattern, args.mag, norder1, cwd_h))
    print(f"harmonic: {displace.outputs.results['number_of_displacements']} displaced structures")
    forces_h = run_cached(bank, "forces_h",
                          lambda: submit_forces(code_ase, displace.outputs.displaced_structures, supercell, calculator,
                                                args.njobs, opt_calc, cwd_h))
    alm_opt_h = run_cached(bank, "alm_opt_h",
                           lambda: submit_alm(code_alm, "opt", supercell, prefix1, norder1, cwd_h, dfset=forces_h.outputs.dfset))
    print("harmonic alm opt:", alm_opt_h.outputs.results["optimization"])
    fc2xml = alm_opt_h.outputs.input_ANPHON

    # --- MD + random displacements (tutorial step 1-2: AIMD, displace.py -md --random)
    cwd_md = Str(dirs["md"])
    md = run_cached(bank, "md",
                    lambda: submit_ase("md_ase", code_ase, supercell, calculator, opt_calc, cwd=cwd_md,
                                             temperature=Float(args.md_temperature), timestep=Float(args.md_timestep),
                                             nsteps=Int(args.md_steps), sample=Str(args.md_sample),
                                             random_mag=Float(args.random_mag), random_seed=Int(args.random_seed)))
    ndata = md.outputs.results["nsnapshots"]
    print(f"MD: {ndata} snapshots, mean T = {md.outputs.results['mean_temperature']:.1f} K")
    forces_md = run_cached(bank, "forces_md",
                           lambda: submit_forces(code_ase, md.outputs.displaced_structures, supercell, calculator,
                                                 args.njobs, opt_calc, cwd_md))
    dfset_md = forces_md.outputs.dfset

    # --- anharmonic IFCs: elastic-net CV, then optimize with the best alpha (tutorial steps 3-4)
    norder3 = Int(3)
    prefix3 = Str(prefix1.value.replace("_harmonic", "_anharm"))
    cwd_a = Str(dirs["anharmonic"])
    cutoff3 = Dict({"*-*": [None, args.cutoff[0], args.cutoff[1]]})
    optimize = {"LMODEL": "elastic-net", "NDATA": ndata, "L1_RATIO": 1.0, "STANDARDIZE": 1,
                "CV_MINALPHA": args.cv_alpha[0], "CV_MAXALPHA": args.cv_alpha[1], "CV_NALPHA": int(args.cv_alpha[2])}
    if args.l1_alpha is None:
        param_cv = Dict({"interaction": {"NBODY": args.nbody},
                         "optimize": {**optimize, "CV": args.cv, "CONV_TOL": 1.0e-8}})
        alm_cv = run_cached(bank, "alm_cv",
                            lambda: submit_alm(code_alm, "cv", supercell, prefix3, norder3, cwd_a, cutoff=cutoff3, dfset=dfset_md,
                                               fc2xml=fc2xml, param=param_cv))
        alpha = alm_cv.outputs.results["alpha_min"]
        print(f"CV: minimum CV score at alpha = {alpha:.5g}")
    else:
        alpha = args.l1_alpha
    param_opt = Dict({"interaction": {"NBODY": args.nbody},
                      "optimize": {**optimize, "CV": 0, "L1_ALPHA": alpha, "CONV_TOL": 1.0e-9}})
    alm_opt_a = run_cached(bank, "alm_opt_a",
                           lambda: submit_alm(code_alm, "opt", supercell, prefix3, norder3, cwd_a, cutoff=cutoff3, dfset=dfset_md,
                                              fc2xml=fc2xml, param=param_opt))
    print("anharmonic alm opt:", alm_opt_a.outputs.results["optimization"])
    fcsxml = alm_opt_a.outputs.input_ANPHON

    # --- Born charges (optional): BORNINFO file or alamode.bec_ase on the primitive cell
    borninfo = None
    if args.borninfo:
        borninfo = run_cached(bank, "borninfo", lambda: SinglefileData(os.path.abspath(args.borninfo)))
    elif args.borninfo_calculator:
        # Z* (alamode.bec_ase) and eps_inf (alamode.epsinf_ase with a dielectric model, or a given value)
        # of the primitive cell -> BORNINFO (BornInfoWorkChain); same atom order as the anphon &position
        bec_calculator = run_cached(bank, "bec_calculator",
                                    lambda: Dict({"name": args.borninfo_calculator, "kwargs": args.borninfo_kwargs}))
        inputs = dict(structure=prim, bec=dict(code=code_ase, calculator=bec_calculator, cwd=Str(dirs["scph"]), options=Dict(opt_serial)))
        if args.dielectric_model:
            inputs["epsinf"] = dict(code=code_ase, dielectric_model=Dict({"name": args.dielectric_model}), cwd=Str(dirs["scph"]),
                                    options=Dict(opt_serial))
        if args.dielectric:
            inputs["dielectric"] = List(args.dielectric)
        bec = run_cached(bank, "borninfo_wc", lambda: submit(WorkflowFactory("alamode.borninfo"), **inputs))
        r = bec.outputs.results
        print("Born effective charges (diagonal) [e]:", {s: np.round(d, 3).tolist() for s, d in zip(r["symbols"], r["bec_diagonal"])})
        print(f"dielectric tensor ({r['epsilon_inf_source']}):", np.round(r["epsilon_inf"], 3).tolist())
        borninfo = bec.outputs.borninfo

    # --- SCPH with structural relaxation (tutorial 7.4)
    a_prim = prim.cell_lengths[0] / BOHR
    disp_lines = ["0", f"{a_prim:.10f}", "1.0 0.0 0.0", "0.0 1.0 0.0", "0.0 0.0 1.0"]
    disp_lines += [" ".join(x.split()) for x in args.init_disp.split(";")]
    param_scph = Dict({
        "general": {"TMIN": args.tmin, "TMAX": args.tmax, "DT": args.dt,
                    **({"NONANALYTIC": args.nonanalytic} if borninfo is not None else {})},
        "scph": {"SELF_OFFDIAG": 1, "MAXITER": args.maxiter, "MIXALPHA": args.mixalpha,
                 "KMESH_INTERPOLATE": " ".join([str(args.kmesh_interpolate)] * 3),
                 "KMESH_SCPH": " ".join([str(args.kmesh_scph)] * 3), "RELAX_STR": 1},
        "relax": {"RELAX_ALGO": 2, "MAX_STR_ITER": args.max_str_iter, "COORD_CONV_TOL": 1.0e-5, "MIXBETA_COORD": args.mixbeta_coord,
                  "SET_INIT_STR": 3, "COOLING_U0_INDEX": 5, "COOLING_U0_THR": 0.005,
                  "ADD_HESS_DIAG": args.add_hess_diag},
        "displace": disp_lines,
    })
    prefix_scph = Str(f"{name}_scph")
    scph = run_cached(bank, "scph",
                      lambda: submit_anphon(code_anphon, prim, fcsxml, "SCPH", prefix_scph, Str(dirs["scph"]), norder=3,
                                            qmesh=List([args.qmesh] * 3), param=param_scph, borninfo=borninfo,
                                            options=opt_anphon))
    print("SCPH outputs:", scph.outputs.results["files"])

    # --- figure
    refs = {}
    if not args.no_ref:
        refs["ref_atom_disp"] = run_cached(bank, "ref_atom_disp",
                                           lambda: SinglefileData(os.path.join(REF_DIR, "cBTO222_scph.atom_disp")))
        refs["ref_thermo"] = run_cached(bank, "ref_thermo",
                                        lambda: SinglefileData(os.path.join(REF_DIR, "cBTO222_scph.scph_thermo")))
    figure = run_cached(bank, "figure",
                        lambda: scph_figure(Str(root), Str(name), Str(args.calc_label), scph.outputs.output_folder,
                                            prefix_scph, prim, **refs)["img_file"])
    summary = figure.base.links.get_incoming().one().node.outputs.summary.get_dict()
    print("figure:", os.path.join(root, figure.filename))
    print(f"{summary['B_site']} z displacement [Bohr] vs T [K]:")
    for t, u in zip(summary["T_K"], summary["Ti_z_Bohr"]):
        print(f"  {t:6.0f}  {u: .5f}")
    print("highest T with a polar (tetragonal) structure:", summary["T_polar_max_K"], "K")
    print(f"done. provenance: verdi node graph generate {figure.pk}")


if __name__ == "__main__":
    main()
