"""ZnO (alamode tutorial 7.5) with an ASE calculator instead of DFT, through AiiDA:
QHA-based structural optimization (thermal expansion of wurtzite ZnO).

    relax (cell + positions)  ->  primitive cell
      ->  harmonic IFCs of the 4x4x2 supercell                                  (FC2XML)
      ->  anharmonic IFCs of the 3x3x2 supercell: MD at 500 K + random 0.04 A, LASSO CV, optimize   (FCSXML)
      ->  harmonic IFCs of the 4x4x2 supercell under 6 strains (xx, yy, zz: 0.005; yz, zx, xy: 0.0025)
      ->  clamped-ion elastic constants (SOEC, TOEC) and strain-force coupling (mattersim_elastic)
      ->  anphon QHA + RELAX_STR = 2 for QHA_SCHEME 0 (full), 1 (ZSISA), 2 (v-ZSISA)  ->  thermal strain figure

usage:
    python run_alamode_qha.py                      # wurtzite ZnO, tutorial settings
    python run_alamode_qha.py --structure X.cif --supercell-harm 4 4 2 --supercell-anharm 3 3 2 --name X
    python run_alamode_qha.py --calculator mace --calculator-kwargs '{"model": "medium"}'

codes: alm, anphon, displace, mattersim @<computer>.
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
from ase.build import bulk

from run_alamode_phonons import (NodeBank, wait, run_cached, read_structure, find_primitive, idealize_structure,
                                 make_supercell_structure, submit_alm, submit_displace, submit_forces, submit_anphon,
                                 submit_mattersim, HERE, ALAMODE_TEST, BOHR)
from aiida.engine import calcfunction
from aiida.orm import load_code, Str, Dict, Float, Int, List, Bool
from aiida.plugins import DataFactory
from aiida_alamode.io.alm_input import AlmPrefixMaker

StructureData = DataFactory('core.structure')
SinglefileData = DataFactory('core.singlefile')
FolderData = DataFactory('core.folder')

REF_DIR = os.path.join(ALAMODE_TEST, "ZnO", "qha_relax", "reference")
STRAINS = ["xx", "yy", "zz", "yz", "zx", "xy"]
SCHEMES = {0: "qha", 1: "qha_zsisa", 2: "qha_vzsisa"}
SCHEME_LABELS = {0: "full QHA", 1: "ZSISA", 2: "v-ZSISA"}


@calcfunction
def strain_structure(structure: StructureData, component: Str, magnitude: Float) -> StructureData:
    """h' = (1 + u) h with fixed fractional coordinates. Diagonal: u_ii = magnitude;
    off-diagonal ij: u_ij = u_ji = magnitude / 2 (the alamode strain_harmonic.in convention)."""
    atoms = structure.get_ase()
    comp = component.value
    i, j = "xyz".index(comp[0]), "xyz".index(comp[1])
    u = np.zeros((3, 3))
    if i == j:
        u[i, i] = magnitude.value
    else:
        u[i, j] = u[j, i] = magnitude.value / 2
    atoms.set_cell(atoms.cell.array @ (np.eye(3) + u).T, scale_atoms=True)
    return StructureData(ase=atoms)


@calcfunction
def make_strain_ifc_folder(elastic: FolderData, magnitude: Float, **xmls) -> FolderData:
    """STRAIN_IFC_DIR contents: elastic_constants.in, strain_force.in, strain_harmonic.in and the xml files.
    xmls: {component: SinglefileData of the strained harmonic IFCs}"""
    folder = FolderData()
    for name in elastic.list_object_names():
        folder.put_object_from_filelike(__import__("io").StringIO(elastic.get_object_content(name)), name)
    lines = []
    for comp in STRAINS:
        xml = xmls[comp]
        folder.put_object_from_filelike(__import__("io").StringIO(xml.get_content()), xml.filename)
        lines.append(f"{comp} {magnitude.value} 1.0 {xml.filename}")
    folder.put_object_from_filelike(__import__("io").StringIO("\n".join(lines) + "\n"), "strain_harmonic.in")
    return folder


@calcfunction
def qha_figure(cwd: Str, name: Str, calc_label: Str, **outputs) -> dict:
    """thermal strain u_xx (= u_yy) and u_zz vs T for each QHA scheme; tutorial reference dashed.
    outputs: scheme{n} (FolderData of the anphon QHA run, containing {prefix}.umn_tensor), ref{n} (SinglefileData)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    summary = {}
    for n, color in zip(sorted(SCHEMES), ["C0", "C1", "C2"]):
        key = f"scheme{n}"
        if key not in outputs:
            continue
        folder = outputs[key]
        fname = [f for f in folder.list_object_names() if f.endswith(".umn_tensor")][0]
        data = np.loadtxt(folder.get_object_content(fname).splitlines())
        data = data[np.argsort(data[:, 0])]
        ax.plot(data[:, 0], data[:, 1], color=color, lw=1.6, label=f"u_xx {SCHEME_LABELS[n]} ({calc_label.value})")
        ax.plot(data[:, 0], data[:, 9], color=color, lw=1.6, ls=":", label=f"u_zz {SCHEME_LABELS[n]}")
        summary[SCHEME_LABELS[n]] = {"T_K": data[:, 0].tolist(), "u_xx": data[:, 1].tolist(), "u_zz": data[:, 9].tolist()}
        if f"ref{n}" in outputs:
            ref = np.loadtxt(outputs[f"ref{n}"].get_content().splitlines())
            ref = ref[np.argsort(ref[:, 0])]
            ax.plot(ref[:, 0], ref[:, 1], color=color, lw=1.0, ls="--", alpha=0.7,
                    label="DFT reference (dashed: u_xx, dash-dot: u_zz)" if n == 0 else None)
            ax.plot(ref[:, 0], ref[:, 9], color=color, lw=1.0, ls="-.", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Thermal strain")
    ax.set_title(f"{name.value} thermal expansion (QHA structural optimization)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    target = os.path.join(cwd.value, f"{name.value}_thermal_strain.png")
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"img_file": SinglefileData(target), "summary": Dict(summary)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", help="structure file (default: wurtzite ZnO, the tutorial's PBEsol cell)")
    parser.add_argument("--supercell-harm", type=int, nargs=3, default=[4, 4, 2], metavar="N")
    parser.add_argument("--supercell-anharm", type=int, nargs=3, default=[3, 3, 2], metavar="N")
    parser.add_argument("--name")
    parser.add_argument("--calculator", default="mattersim")
    parser.add_argument("--calculator-kwargs", default="{}")
    parser.add_argument("--calc-label")
    parser.add_argument("--relax", choices=["full", "volume", "none"], default="full")
    parser.add_argument("--mag", type=float, default=0.01)
    parser.add_argument("--md-temperature", type=float, default=500.0)
    parser.add_argument("--md-timestep", type=float, default=1.0)
    parser.add_argument("--md-steps", type=int, default=5000)
    parser.add_argument("--md-sample", default="1001:5000:50")
    parser.add_argument("--random-mag", type=float, default=0.04)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--nbody", default="2 3 3")
    parser.add_argument("--cutoff", type=float, nargs=2, default=[12.0, 8.0], metavar="BOHR", help="cubic, quartic")
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--cv-alpha", type=float, nargs=3, default=[1.0e-8, 0.01, 30], metavar="X")
    parser.add_argument("--l1-alpha", type=float)
    parser.add_argument("--strain", type=float, default=0.005, help="strain of the strained harmonic IFCs and strain_force.in")
    parser.add_argument("--elastic-delta", type=float, default=0.01, help="finite-difference step of the elastic constants")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=100.0)
    parser.add_argument("--kmesh-qha", type=int, nargs=3, default=[4, 4, 2])
    parser.add_argument("--kmesh-interpolate", type=int, nargs=3, default=[4, 4, 2])
    parser.add_argument("--qmesh", type=int, default=8)
    parser.add_argument("--schemes", type=int, nargs="+", default=[0, 1, 2], help="QHA_SCHEME values (0 full, 1 ZSISA, 2 v-ZSISA)")
    parser.add_argument("--no-ref", action="store_true")
    parser.add_argument("--computer", default="mygarden5")
    parser.add_argument("--gpu", action="store_true", help="request one GPU (#SBATCH --gres=gpu:1) for the MatterSim / SevenNet jobs")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--njobs", type=int, default=2)
    parser.add_argument("--root", default=os.path.join(HERE, "run_alamode_qha"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)
    args.calculator_kwargs = json.loads(args.calculator_kwargs)
    if args.calc_label is None:
        args.calc_label = "MatterSim-v1.0.0-1M" if args.calculator == "mattersim" and not args.calculator_kwargs \
            else args.calculator + ("" if not args.calculator_kwargs else " " + json.dumps(args.calculator_kwargs))
    if args.structure is None:
        args.structure = os.path.join(HERE, "ZnO_P6_3mc.cif")
        if not os.path.isfile(args.structure):
            # the tutorial's PBEsol cell: a = 24.4596 / 4 Bohr, c = 19.7465 / 2 Bohr
            ase.io.write(args.structure, bulk("ZnO", "wurtzite", a=24.45955161866495 / 4 * BOHR,
                                              c=19.74654961812053 / 2 * BOHR), format="cif")
    return args


def main():
    args = parse_args()
    input_atoms = ase.io.read(args.structure)
    formula = ase.formula.Formula(input_atoms.get_chemical_formula()).reduce()[0].format("metal")
    name = args.name or formula + ("" if args.calculator == "mattersim" else f"_{args.calculator}")
    root = os.path.join(args.root, name)
    dirs = {k: os.path.join(root, k) for k in ["relax", "harmonic", "md", "anharmonic", "elastic", "qha"]
            + [f"strain_{c}" for c in STRAINS]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    bank = NodeBank(root, args.force)
    print(f"{name}: {args.structure} ({len(input_atoms)} atoms), supercells {args.supercell_harm} / {args.supercell_anharm}")

    code_alm = load_code(f"alm@{args.computer}")
    code_anphon = load_code(f"anphon@{args.computer}")
    code_displace = load_code(f"displace@{args.computer}")
    code_mattersim = load_code(f"mattersim@{args.computer}")
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
        relax = run_cached(bank, f"relax_{args.relax}",
                           lambda: submit_mattersim("mattersim_relax", code_mattersim, unit0, calculator, opt_serial,
                                                    cwd=Str(dirs["relax"]), hydrostatic_strain=Bool(args.relax == "volume")))
        unit = relax.outputs.structure
        r = relax.outputs.results
        print(f"relaxed cell [A]: {np.round(r['cell_lengths'], 5)} angles {np.round(r['cell_angles'], 3)} "
              f"(input {np.round(unit0.cell_lengths, 5)})")
    symprec = Float(1e-3)
    # the relaxed cell is symmetry-idealized (alm finds 0 free IFCs on a cell with ~1e-6 noise);
    # the supercells are built from this primitive cell, as in the tutorial.
    prim = run_cached(bank, "prim", lambda: idealize_structure(unit, symprec))
    unit = prim
    nat_prim = len(prim.sites)
    print(f"primitive cell: {[s.kind_name for s in prim.sites]}")

    def harmonic_ifcs(tag, unit_structure, diag, cwd, prefix):
        """alm suggest -> displace -> forces (offset subtracted) -> DFSET -> alm opt of the supercell diag x unit_structure.
        Returns the alm opt node (outputs.input_ANPHON is the xml)."""
        norder1 = Int(1)
        cwd = Str(cwd)
        supercell = run_cached(bank, f"{tag}_supercell", lambda: make_supercell_structure(unit_structure, List(diag)))
        suggest = run_cached(bank, f"{tag}_alm_suggest", lambda: submit_alm(code_alm, "suggest", supercell, prefix, norder1, cwd))
        displace = run_cached(bank, f"{tag}_displace",
                              lambda: submit_displace(code_displace, supercell, suggest.outputs.pattern, args.mag, norder1, cwd))
        # forces of the undisplaced supercell are subtracted (a strained cell is not at equilibrium)
        forces = run_cached(bank, f"{tag}_forces",
                            lambda: submit_forces(code_mattersim, displace.outputs.displaced_structures, supercell, calculator,
                                                  args.njobs, opt_calc, cwd, subtract_offset=True))
        opt = run_cached(bank, f"{tag}_alm_opt",
                         lambda: submit_alm(code_alm, "opt", supercell, prefix, norder1, cwd, dfset=forces.outputs.dfset))
        print(f"{tag}: {len(supercell.sites)} atoms, {displace.outputs.results['number_of_displacements']} displacements, "
              f"fitting error {opt.outputs.results['optimization'].get('fitting_error')} %")
        return supercell, opt

    # --- 1. harmonic IFCs of the large supercell (FC2XML)
    kh = "x".join(map(str, args.supercell_harm))
    _, opt_h = harmonic_ifcs("harm", unit, args.supercell_harm, dirs["harmonic"], Str(f"{name}_k{kh}_harmonic"))
    fc2xml = opt_h.outputs.input_ANPHON

    # --- 2. anharmonic IFCs of the small supercell: MD + random -> CV -> opt (FCSXML)
    ka = "x".join(map(str, args.supercell_anharm))
    cwd_md = Str(dirs["md"])
    supercell_a = run_cached(bank, "anharm_supercell", lambda: make_supercell_structure(unit, List(args.supercell_anharm)))
    # the harmonic IFCs of the small supercell are fixed in the anharmonic fit (FC2XML must be commensurate)
    _, opt_ha = harmonic_ifcs("harm_small", unit, args.supercell_anharm, dirs["anharmonic"], Str(f"{name}_k{ka}_harmonic"))

    md = run_cached(bank, "md",
                    lambda: submit_mattersim("mattersim_md", code_mattersim, supercell_a, calculator, opt_calc, cwd=cwd_md,
                                             temperature=Float(args.md_temperature), timestep=Float(args.md_timestep),
                                             nsteps=Int(args.md_steps), sample=Str(args.md_sample),
                                             random_mag=Float(args.random_mag), random_seed=Int(args.random_seed)))
    ndata = md.outputs.results["nsnapshots"]
    print(f"MD ({len(supercell_a.sites)} atoms, {args.md_temperature} K): {ndata} snapshots, "
          f"mean T = {md.outputs.results['mean_temperature']:.1f} K")
    forces_md = run_cached(bank, "forces_md",
                           lambda: submit_forces(code_mattersim, md.outputs.displaced_structures, supercell_a, calculator,
                                                 args.njobs, opt_calc, cwd_md))
    dfset_md = forces_md.outputs.dfset
    norder3 = Int(3)
    prefix3 = Str(f"{name}_k{ka}_anharm")
    cwd_a = Str(dirs["anharmonic"])
    cutoff3 = Dict({"*-*": [None, args.cutoff[0], args.cutoff[1]]})
    optimize = {"LMODEL": "elastic-net", "NDATA": ndata, "L1_RATIO": 1.0, "STANDARDIZE": 1,
                "CV_MINALPHA": args.cv_alpha[0], "CV_MAXALPHA": args.cv_alpha[1], "CV_NALPHA": int(args.cv_alpha[2])}
    if args.l1_alpha is None:
        param_cv = Dict({"interaction": {"NBODY": args.nbody}, "optimize": {**optimize, "CV": args.cv, "CONV_TOL": 1.0e-8}})
        alm_cv = run_cached(bank, "alm_cv", lambda: submit_alm(code_alm, "cv", supercell_a, prefix3, norder3, cwd_a, cutoff=cutoff3,
                                                                dfset=dfset_md, fc2xml=opt_ha.outputs.input_ANPHON, param=param_cv))
        alpha = alm_cv.outputs.results["alpha_min"]
        print(f"CV: minimum CV score at alpha = {alpha:.5g}")
    else:
        alpha = args.l1_alpha
    param_opt = Dict({"interaction": {"NBODY": args.nbody}, "optimize": {**optimize, "CV": 0, "L1_ALPHA": alpha, "CONV_TOL": 1.0e-9}})
    alm_opt_a = run_cached(bank, "alm_opt_a", lambda: submit_alm(code_alm, "opt", supercell_a, prefix3, norder3, cwd_a, cutoff=cutoff3,
                                                                  dfset=dfset_md, fc2xml=opt_ha.outputs.input_ANPHON, param=param_opt))
    print("anharmonic alm opt:", alm_opt_a.outputs.results["optimization"])
    fcsxml = alm_opt_a.outputs.input_ANPHON

    # --- 3. strained harmonic IFCs (large supercell), 6 strains
    strained_xml = {}
    for comp in STRAINS:
        unit_s = run_cached(bank, f"unit_{comp}", lambda: strain_structure(unit, Str(comp), Float(args.strain)))
        tag = f"h{comp}"
        mag = f"{args.strain:g}".replace(".", "")
        _, opt_s = harmonic_ifcs(tag, unit_s, args.supercell_harm, dirs[f"strain_{comp}"],
                                 Str(f"{name}_k{kh}_harmonic_{comp}_{mag}"))
        strained_xml[comp] = opt_s.outputs.input_ANPHON

    # --- 4. elastic constants and strain-force coupling of the primitive cell
    elastic = run_cached(bank, "elastic",
                         lambda: submit_mattersim("mattersim_elastic", code_mattersim, prim, calculator, opt_serial,
                                                  cwd=Str(dirs["elastic"]), delta=Float(args.elastic_delta),
                                                  strain_force_delta=Float(args.strain)))
    C = np.array(elastic.outputs.results["soec_GPa"])
    print(f"elastic constants [GPa]: C11 {C[0, 0]:.1f} C12 {C[0, 4]:.1f} C13 {C[0, 8]:.1f} C33 {C[8, 8]:.1f} "
          f"C44 {C[5, 5]:.1f} C66 {C[1, 1]:.1f}")
    strain_ifc = run_cached(bank, "strain_ifc",
                            lambda: make_strain_ifc_folder(elastic.outputs.strain_ifc_folder, Float(args.strain), **strained_xml))

    # --- 5. anphon QHA + RELAX_STR = 2 for each scheme
    def submit_qha(scheme):
        param = Dict({
            "general": {"TMIN": args.tmin, "TMAX": args.tmax, "DT": args.dt},
            "qha": {"KMESH_INTERPOLATE": " ".join(map(str, args.kmesh_interpolate)),
                    "KMESH_QHA": " ".join(map(str, args.kmesh_qha)), "RELAX_STR": 2, "QHA_SCHEME": scheme},
            "relax": {"RELAX_ALGO": 2, "MAX_STR_ITER": 1000, "COORD_CONV_TOL": 1.0e-5, "MIXBETA_COORD": 0.5,
                      "CELL_CONV_TOL": 1.0e-5, "MIXBETA_CELL": 0.7, "SET_INIT_STR": 2, "ADD_HESS_DIAG": 0.0,
                      "RENORM_3TO2ND": 2, "RENORM_2TO1ST": 2, "RENORM_34TO1ST": 0, "STRAIN_IFC_DIR": "./"},
            "displace": ["1"] + ["0.0 0.0 0.0"] * nat_prim,
            "strain": ["0.0 0.0 0.0"] * 3,
        })
        return submit_anphon(code_anphon, prim, fcsxml, "QHA", Str(f"{name}_{SCHEMES[scheme]}"), Str(dirs["qha"]), norder=3,
                             qmesh=List([args.qmesh] * 3), param=param, fc2xml=fc2xml, extra_files=strain_ifc,
                             options=opt_anphon)

    qha = {}
    pending = []
    for scheme in args.schemes:
        label = f"qha_{scheme}"
        node = bank.load(label)
        if node is None:
            node = submit_qha(scheme)
            print(f"submitted {label}: {node}")
            pending.append((label, node))
        qha[scheme] = node
    wait([node for _, node in pending])
    for label, node in pending:
        bank.dump(label, node)

    # --- 6. figure
    outputs = {f"scheme{s}": qha[s].outputs.output_folder for s in args.schemes}
    if not args.no_ref:
        for s in args.schemes:
            ref = os.path.join(REF_DIR, f"ZnO_{SCHEMES[s]}.umn_tensor")
            if os.path.isfile(ref):
                outputs[f"ref{s}"] = run_cached(bank, f"ref_umn_{s}", lambda ref=ref: SinglefileData(ref))
    figure = run_cached(bank, "figure",
                        lambda: qha_figure(Str(root), Str(name), Str(args.calc_label), **outputs)["img_file"])
    summary = figure.base.links.get_incoming().one().node.outputs.summary.get_dict()
    print("figure:", os.path.join(root, figure.filename))
    for scheme, v in summary.items():
        print(f"{scheme}: T [K] / u_xx / u_zz")
        for t, uxx, uzz in zip(v["T_K"], v["u_xx"], v["u_zz"]):
            print(f"  {t:6.0f}  {uxx: .5f}  {uzz: .5f}")
    print(f"done. provenance: verdi node graph generate {figure.pk}")


if __name__ == "__main__":
    main()
