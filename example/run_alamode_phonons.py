"""Phonons of any crystal with ALAMODE + an ASE calculator (MatterSim by default), through AiiDA (aiida-alamode).

Input: a structure file readable by ASE (CIF, POSCAR, ...), the supercell size, and optionally the calculator.

    relax (alamode.relax_ase)  ->  primitive cell / supercell (calcfunctions)  ->  alm suggest (alamode.alm_suggest)
      ->  displace.py (alamode.displace_pf)  ->  forces + DFSET (alamode.forces)
      ->  alm opt (alamode.alm_opt)  ->  anphon band / DOS (alamode.anphon)  ->  C_v, S, F (T) figure
      [->  the same anphon step on reference IFCs  ->  comparison figure]
    --cubic: alm suggest NORDER=2  ->  displace  ->  forces / DFSET  ->  alm opt (FC2XML fixed)
      ->  anphon RTA (kappa, kappa spectrum)  ->  analyze_phonons (tau, cumulative, boundary)  ->  figure

Every step is a CalcJob on the computer (slurm) or a calcfunction, so the whole chain is in the
provenance graph. Finished nodes are recorded in <root>/<name>/.node.json and reused on rerun.

usage:
    python run_alamode_phonons.py --structure Si_Fd-3m.cif --supercell 2 2 2
    python run_alamode_phonons.py --structure POSCAR --supercell 3 3 2 --relax full --name ZnO
    python run_alamode_phonons.py --preset PbTe          # alamode tutorial settings (a = 6.45 A, 4x4x4, Born charges, NA0-3)
    python run_alamode_phonons.py --preset Si            # alamode tutorial settings (2x2x2 conventional, cubic IFCs, RTA)
    python run_alamode_phonons.py --structure Si.cif --supercell 2 2 2 --cubic --cubic-cutoff 7.5
    python run_alamode_phonons.py --preset Si --calculator mace --calculator-kwargs '{"model": "medium"}'
    python run_alamode_phonons.py --preset Si --calculator emt   # any name in aiida_alamode.ase_runner.CALCULATORS

codes: alm, anphon, displace, analyze_phonons, ase_runner @<computer>.  verdi daemon and RabbitMQ must be running.

"""
import argparse
import json
import os
import sys
from time import sleep

sys.stdout.reconfigure(line_buffering=True)   # progress lines reach a redirected log immediately

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ase.io
import ase.formula
import spglib
from ase import Atoms
from ase.build import bulk

import aiida
aiida.load_profile()

from aiida.engine import calcfunction, submit
from aiida.orm import load_code, load_node, Str, Dict, Float, Int, List, Bool, ProcessNode
from aiida.plugins import DataFactory, WorkflowFactory, CalculationFactory
from aiida_alamode.io.alm_input import AlmPrefixMaker
from aiida_alamode.io.supercell import make_diagonal_supercell
from aiida_alamode.calculations.anphon_calcjob import thermo_to_arraydata

StructureData = DataFactory('core.structure')
SinglefileData = DataFactory('core.singlefile')
ArrayData = DataFactory('core.array')
FolderData = DataFactory('core.folder')
TrajectoryData = DataFactory('core.array.trajectory')

BOHR = 0.5291772108   # alamode tools/interface/QE.py
CM1_TO_THZ = 0.0299792458
RY_TO_MEV = 13605.693123
HERE = os.path.dirname(os.path.abspath(__file__))
ALAMODE_TEST = os.path.join(HERE, "..", "..", "alamode_test")
KPATH_TUTORIAL = ["1",   # G-X-G-L, the path of the alamode tutorial (Si, PbTe)
                  "G 0.0 0.0 0.0 X 0.5 0.5 0.0 51",
                  "X 0.5 0.5 1.0 G 0.0 0.0 0.0 51",
                  "G 0.0 0.0 0.0 L 0.5 0.5 0.5 51"]


# the alamode tutorial settings (alamode_test/{Si,PbTe}), DFT replaced by MatterSim; ref_xml = the DFT IFCs.
PRESETS = {
    "Si": dict(structure="Si_Fd-3m.cif", make=lambda: bulk("Si", "diamond", a=5.431, cubic=True),
               supercell=[2, 2, 2], mag=0.01, emax=550, delta_e=1.0, nonanalytic=[0], borninfo=None,
               kpath="tutorial", ref_xml="Si/reference/si222.xml", ref_label="DFT (QE, PBE)",
               # cubic IFCs up to the 2nd neighbours (7.30 Bohr with the MatterSim a; the tutorial uses 7.3 for a = 5.40 A)
               cubic=True, cubic_cutoff=7.5, cubic_mag=0.04, rta_qmesh=10,
               ref_cubic_xml="Si/reference/si222_cubic.xml.bz2"),
    "PbTe": dict(structure="PbTe_Fm-3m_primitive.cif", make=lambda: bulk("PbTe", "rocksalt", a=6.45),
                 supercell=[4, 4, 4],   # 4x4x4 of the fcc primitive cell (128 atoms), as in the tutorial
                 mag=0.01, emax=150, delta_e=0.5, nonanalytic=[0, 1, 2, 3],   # the 4 NONANALYTIC methods of the tutorial
                 borninfo="PbTe/reference/PbTe.born", kpath="tutorial",
                 ref_xml="PbTe/reference/super444_0.01.xml", ref_label="DFT (VASP, PBEsol)"),
}


class NodeBank:
    """label -> pk in <root>/.node.json, to reuse finished nodes across runs."""

    def __init__(self, root, force=False):
        self.filepath = os.path.join(root, ".node.json")
        self.force = force
        self.dic = {}
        if os.path.isfile(self.filepath):
            with open(self.filepath) as f:
                self.dic = json.load(f)

    def load(self, label):
        if self.force or label not in self.dic:
            return None
        node = load_node(self.dic[label])
        print(f"reuse {label}: {node}")
        return node

    def dump(self, label, node):
        node.store()
        self.dic[label] = node.pk
        with open(self.filepath, "w") as f:
            json.dump(self.dic, f, indent=1)
        print(f"save  {label}: {node}")


def wait(nodes, sec=5):
    nodes = [nodes] if not isinstance(nodes, (list, tuple)) else nodes
    while not all(node.is_terminated for node in nodes):
        sleep(sec)
    for node in nodes:
        if not node.is_finished_ok:
            raise RuntimeError(f"{node} failed. see: verdi process report {node.pk}")


def run_cached(bank, label, factory):
    """factory() submits (or runs a calcfunction) and returns the node; reused if in the bank."""
    node = bank.load(label)
    if node is None:
        node = factory()
        if isinstance(node, ProcessNode) and not node.is_terminated:
            print(f"submitted {label}: {node}")
            wait(node)
        bank.dump(label, node)
    return node


# ---------------------------------------------------------------- calcfunctions

@calcfunction
def read_structure(structure_file: SinglefileData, fmt: Str) -> StructureData:
    """structure file (CIF, POSCAR, ...) -> StructureData"""
    from ase.io.formats import filetype
    fmt = fmt.value or filetype(structure_file.filename, read=False)   # ase can't guess from a handle
    with structure_file.open() as handle:
        atoms = ase.io.read(handle, format=fmt)
    atoms.set_pbc(True)
    return StructureData(ase=atoms)


@calcfunction
def find_primitive(structure: StructureData, symprec: Float) -> StructureData:
    """primitive cell by spglib in the cartesian frame of the input (no idealization: spglib.find_primitive
    rotates the lattice to a standard orientation, which breaks the mapping between anphon's primitive cell
    and the supercell in the IFC xml). The species order of the input is kept (BORNINFO must follow it)."""
    atoms = structure.get_ase()
    cell, positions, numbers = spglib.standardize_cell(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
        to_primitive=True, no_idealize=True, symprec=symprec.value)
    prim = Atoms(numbers=numbers, cell=cell, scaled_positions=positions, pbc=True)
    return StructureData(ase=prim)


@calcfunction
def idealize_structure(structure: StructureData, symprec: Float) -> StructureData:
    """symmetry-idealized primitive cell in the cartesian frame of the input.

    A relaxed structure carries ~1e-6 noise that alm's equivalence test does not tolerate (it then
    finds the space group but 0 free IFCs). spglib idealizes cell and positions in its standard
    orientation; the result is rotated back onto the non-idealized primitive cell (Kabsch fit of the
    lattice vectors, which differ only by a rotation).
    """
    atoms = structure.get_ase()
    spg = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    cell_i, pos_i, num_i = spglib.standardize_cell(spg, to_primitive=True, no_idealize=False, symprec=symprec.value)
    cell_n, _, _ = spglib.standardize_cell(spg, to_primitive=True, no_idealize=True, symprec=symprec.value)
    # rotation Q with cell_n ~ cell_i @ Q  (rows are lattice vectors)
    u, _, vt = np.linalg.svd(cell_i.T @ cell_n)
    q = u @ vt
    if np.linalg.det(q) < 0:
        u[:, -1] *= -1
        q = u @ vt
    # alm's equivalence test does not tolerate even ~1e-8 noise: snap q to an exact axis-aligned
    # matrix when it is one (identity, permutations, sign flips, 90-degree rotations).
    if np.abs(q - np.round(q)).max() < 1e-4:
        q = np.round(q)
    cell = cell_i @ q
    if np.abs(cell - cell_n).max() > 1e-2:
        raise ValueError("idealized and input primitive cells are not related by a rotation: "
                         f"{np.round(cell, 4)} vs {np.round(cell_n, 4)}")
    return StructureData(ase=Atoms(numbers=num_i, cell=cell, scaled_positions=pos_i, pbc=True))


@calcfunction
def primitive_from_fcsxml(fcsxml: SinglefileData, symprec: Float) -> StructureData:
    """primitive cell of the supercell stored in an alm IFC xml, in the frame of that xml.

    anphon maps its primitive cell onto the xml supercell in cartesian coordinates, so a reference
    xml made elsewhere (e.g. by DFT) must be paired with a primitive cell in its own frame.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(fcsxml.get_content())
    struct = root.find("Structure")
    cell = np.array([[float(x) for x in struct.find(f"LatticeVector/a{i}").text.split()] for i in (1, 2, 3)]) * BOHR
    symbols, positions = [], []
    for pos in struct.find("Position"):
        symbols.append(pos.attrib["element"])
        positions.append([float(x) for x in pos.text.split()])
    atoms = Atoms(symbols=symbols, cell=cell, scaled_positions=positions, pbc=True)
    cell, positions, numbers = spglib.standardize_cell(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
        to_primitive=True, no_idealize=True, symprec=symprec.value)
    return StructureData(ase=Atoms(numbers=numbers, cell=cell, scaled_positions=positions, pbc=True))


@calcfunction
def make_supercell_structure(structure: StructureData, diag: List) -> StructureData:
    # not ase.build.make_supercell: translation 1 must coincide with the primitive cell (see supercell.py)
    return StructureData(ase=make_diagonal_supercell(structure.get_ase(), diag.get_list()))



@calcfunction
def select_patterns(pattern: List, indices: List) -> List:
    """subset of the alm suggest patterns (index 0: harmonic, 1: cubic, ...), e.g. only the cubic one"""
    patterns = pattern.get_list()
    return List(list=[patterns[i] for i in indices.get_list()])


# ---------------------------------------------------------------- submit helpers (shared by the drivers)

ALM_OPTIONS = {"resources": {"num_machines": 1, "tot_num_mpiprocs": 1}, "max_wallclock_seconds": 4 * 3600}


def submit_alm(code, kind, structure, prefix, norder, cwd=None, cutoff=None, dfset=None, fc2xml=None, param=None,
               options=None):
    """alm suggest / opt / cv (alamode.alm_suggest, alamode.alm_opt, alamode.alm_cv)"""
    builder = CalculationFactory(f"alamode.alm_{kind}").get_builder()
    builder.code = code
    builder.structure = structure
    builder.prefix = prefix
    builder.norder = norder
    if cwd is not None:
        builder.cwd = cwd
    if cutoff is not None:
        builder.cutoff = cutoff
    if dfset is not None:
        builder.dfset = dfset
    if fc2xml is not None:
        builder.fc2xml = fc2xml
    if param is not None:
        builder.param = param
    builder.metadata.options = options or ALM_OPTIONS
    return submit(builder)


def submit_displace(code, structure, pattern, mag, norder, cwd=None, prefix="disp"):
    """displace.py -pf on the supercell (alamode.displace_pf, QE template)"""
    builder = CalculationFactory("alamode.displace_pf").get_builder()
    builder.code = code
    builder.format = Str("QE")
    builder.structure_org = structure
    builder.pattern = pattern
    builder.mag = Float(mag)
    builder.norder = norder
    builder.prefix = Str(prefix)
    if cwd is not None:
        builder.cwd = cwd
    builder.metadata.options = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}
    return submit(builder)


def submit_forces(code, structures, structure_org, calculator, njobs, options, cwd=None, subtract_offset=False):
    """forces of the displaced structures and the DFSET (alamode.forces)"""
    inputs = dict(code=code, structures=structures, structure_org=structure_org, calculator=calculator,
                  njobs=Int(njobs), options=Dict(options), subtract_offset=Bool(subtract_offset))
    if cwd is not None:
        inputs["cwd"] = cwd
    return submit(WorkflowFactory("alamode.forces"), **inputs)


def submit_anphon(code, structure, fcsxml, mode, prefix, cwd=None, norder=1, phonons_mode=None, qmesh=None,
                  param=None, borninfo=None, fc2xml=None, extra_files=None, kappa_spec=0, options=None):
    """anphon (alamode.anphon): mode phonons (band / dos), RTA, or generic (SCPH, QHA, ...)"""
    builder = CalculationFactory("alamode.anphon").get_builder()
    builder.code = code
    builder.structure = structure
    builder.fcsxml = fcsxml
    builder.mode = Str(mode)
    builder.prefix = prefix
    builder.norder = Int(norder)
    if cwd is not None:
        builder.cwd = cwd
    if phonons_mode is not None:
        builder.phonons_mode = Str(phonons_mode)
    if qmesh is not None:
        builder.qmesh = qmesh
    if param is not None:
        builder.param = param
    if borninfo is not None:
        builder.borninfo = borninfo
    if fc2xml is not None:
        builder.fc2xml = fc2xml
    if extra_files is not None:
        builder.extra_files = extra_files
    builder.kappa_spec = Int(kappa_spec)
    builder.metadata.options = options or {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}
    return submit(builder)


def submit_ase(entry, code, structure, calculator, options, cwd=None, **inputs):
    """alamode.relax_ase / md_ase / elastic_ase / bec_ase of a structure (ASE-calculator engine)"""
    builder = CalculationFactory(f"alamode.{entry}").get_builder()
    builder.code = code
    builder.structure = structure
    builder.calculator = calculator
    if cwd is not None:
        builder.cwd = cwd
    for key, value in inputs.items():
        setattr(builder, key, value)
    builder.metadata.options = options
    return submit(builder)


def _load_bands(content):
    lines = content.splitlines()
    labels = lines[0].split()[1:]
    ticks = np.array(lines[1].split()[1:], dtype=float)
    data = np.loadtxt(lines)
    scale = data[-1, 0]
    return labels, ticks / scale, data[:, 0] / scale, data[:, 1:] * CM1_TO_THZ


def _load_dos(content):
    data = np.loadtxt(content.splitlines())
    return data[:, 0] * CM1_TO_THZ, data[:, 1] / CM1_TO_THZ


@calcfunction
def phonon_figure(cwd: Str, name: Str, calc_label: Str, ref_label: Str, nonanalytic: List, qmesh: Int, **files) -> dict:
    """bands for each NONANALYTIC value and the DOS; MatterSim, and the reference IFCs if given.

    files: band_ms_NA{n}, dos_ms_NA{n} and optionally band_ref_NA{n}, dos_ref_NA{n} (SinglefileData).
    """
    nas = nonanalytic.get_list()
    has_ref = f"band_ref_NA{nas[0]}" in files
    fig, axes = plt.subplots(1, len(nas) + 1, figsize=(5 * len(nas) + 4, 4.8),
                             gridspec_kw={"width_ratios": [1] * len(nas) + [0.8]})
    summary = {}
    for ax, na in zip(axes, nas):
        labels, ticks, k, w = _load_bands(files[f"band_ms_NA{na}"].get_content())
        ymax = w.max()
        if has_ref:
            _, _, k_ref, w_ref = _load_bands(files[f"band_ref_NA{na}"].get_content())
            ax.plot(k_ref, w_ref, color="0.6", lw=1.2, ls="--")
            ax.plot([], [], color="0.6", lw=1.2, ls="--", label=f"{ref_label.value} reference")
            ymax = max(ymax, w_ref.max())
        ax.plot(k, w, color="C0", lw=1.5)
        ax.plot([], [], color="C0", lw=1.5, label=calc_label.value)
        for t in ticks:
            ax.axvline(t, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([r"$\Gamma$" if s == "G" else s for s in labels])
        ax.set_xlim(0, 1)
        ax.set_ylim(min(0, w.min() - 0.1), ymax * 1.05)
        ax.set_ylabel("Frequency (THz)")
        title = {0: "without LO-TO correction", 3: "with LO-TO correction (Ewald)"}.get(na, f"NONANALYTIC={na}")
        ax.set_title(f"{name.value} phonons, {title} (NA{na})", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        summary[f"NA{na}"] = {"mattersim_min_THz": float(w.min()), "mattersim_max_THz": float(w.max()),
                              "mattersim_gamma_THz": np.sort(w[0]).tolist()}
        if has_ref:
            summary[f"NA{na}"].update({"ref_min_THz": float(w_ref.min()), "ref_max_THz": float(w_ref.max()),
                                       "ref_gamma_THz": np.sort(w_ref[0]).tolist()})

    ax = axes[-1]
    for na, ls in zip(nas, ["-", ":", "-.", "--"]):
        e, d = _load_dos(files[f"dos_ms_NA{na}"].get_content())
        if has_ref:
            e_ref, d_ref = _load_dos(files[f"dos_ref_NA{na}"].get_content())
            ax.plot(e_ref, d_ref, color="0.5", lw=1.2, ls=ls, label=f"{ref_label.value}, NA{na}")
            summary[f"NA{na}"]["dos_integral_ref"] = float(np.trapezoid(d_ref, e_ref))
        ax.plot(e, d, color="C0", lw=1.5, ls=ls, label=f"{calc_label.value}, NA{na}")
        summary[f"NA{na}"]["dos_integral_mattersim"] = float(np.trapezoid(d, e))
    ax.set_xlim(0, axes[0].get_ylim()[1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("DOS (states / THz / primitive cell)")
    nq = qmesh.value
    ax.set_title(f"{name.value} phonon DOS ({nq}x{nq}x{nq} q mesh)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = os.path.join(cwd.value, f"{name.value}_phband_phdos.png")
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"img_file": SinglefileData(target), "summary": Dict(summary)}


@calcfunction
def thermo_arrays(thermo_file: SinglefileData) -> ArrayData:
    """{prefix}.thermo -> ArrayData (for anphon nodes made before the parser gave the 'thermo' output)"""
    return thermo_to_arraydata(thermo_file.get_content())


@calcfunction
def thermo_figure(cwd: Str, name: Str, calc_label: Str, ref_label: Str, natom: Int, **thermo) -> dict:
    """C_v(T), S(T) and F(T) per primitive cell (harmonic, from anphon DOS runs); C_v against the
    Dulong-Petit limit 3 N kB.

    thermo: ms and optionally ref (ArrayData: temperatures, heat_capacity [kB], entropy [kB], free_energy [Ry]).
    """
    n = natom.value
    dp = 3.0 * n
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    summary = {"natom_primitive": n, "dulong_petit_kB": dp}
    curves = [("ms", calc_label.value, "C0", "-")]
    if "ref" in thermo:
        curves.insert(0, ("ref", f"{ref_label.value} reference", "0.5", "--"))
    for tag, label, color, ls in curves:
        a = thermo[tag]
        T = a.get_array("temperatures")
        cv = a.get_array("heat_capacity")
        s = a.get_array("entropy")
        f = a.get_array("free_energy") * RY_TO_MEV
        axes[0].plot(T, cv, color=color, ls=ls, lw=1.5, label=label)
        axes[1].plot(T, s, color=color, ls=ls, lw=1.5, label=label)
        axes[2].plot(T, f, color=color, ls=ls, lw=1.5, label=label)
        at = {}
        for t0 in (100.0, 300.0, 1000.0):
            if T.min() <= t0 <= T.max():
                cv0 = float(np.interp(t0, T, cv))
                at[f"{t0:g}K"] = {"Cv_kB": cv0, "Cv_over_dulong_petit": cv0 / dp,
                                  "S_kB": float(np.interp(t0, T, s)), "F_meV": float(np.interp(t0, T, f))}
        # temperature where C_v reaches 90 % of the classical limit (a rough Debye-temperature scale)
        above = np.nonzero(cv >= 0.9 * dp)[0]
        summary[tag] = {"zero_point_energy_meV": float(f[np.argmin(T)]), "at": at,
                        "T_Cv_90pct_dulong_petit_K": float(T[above[0]]) if len(above) else None}

    ax = axes[0]
    ax.axhline(dp, color="k", lw=0.8, ls=":")
    ax.text(T.max(), dp, f"Dulong-Petit 3N = {dp:g}", ha="right", va="bottom", fontsize=8)
    ax.set_ylim(0, dp * 1.12)
    ax.set_ylabel(r"$C_v$ ($k_B$ / primitive cell)")
    ax.set_title(f"{name.value} heat capacity (N = {n} atoms / cell)", fontsize=10)
    right = ax.secondary_yaxis("right", functions=(lambda y: y / dp, lambda r: r * dp))
    right.set_ylabel(r"$C_v$ / $3Nk_B$")
    axes[1].set_ylabel(r"$S$ ($k_B$ / primitive cell)")
    axes[1].set_title(f"{name.value} vibrational entropy", fontsize=10)
    axes[2].set_ylabel("$F$ (meV / primitive cell)")
    axes[2].set_title(f"{name.value} vibrational free energy (incl. zero-point)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Temperature (K)")
        ax.set_xlim(0, T.max())
        ax.legend(fontsize=8, loc="lower right" if ax is not axes[2] else "lower left")
    fig.tight_layout()
    target = os.path.join(cwd.value, f"{name.value}_thermo.png")
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"img_file": SinglefileData(target), "summary": Dict(summary)}


@calcfunction
def decompress_file(compressed: SinglefileData, cwd: Str) -> SinglefileData:
    """bz2/gz -> plain file in cwd (reference IFC xml files of alamode_test are compressed)."""
    import bz2
    import gzip
    name = compressed.filename
    opener = {".bz2": bz2.open, ".gz": gzip.open}[os.path.splitext(name)[1]]
    target = os.path.join(cwd.value, os.path.splitext(name)[0])
    with compressed.open(mode="rb") as fin, opener(fin) as fz, open(target, "wb") as fout:
        fout.write(fz.read())
    return SinglefileData(target)


def _kappa_avg(data):
    """anphon .kl / cumulative .dat: T (or L), then the 3x3 tensor -> (x, trace / 3)"""
    return data[:, 0], (data[:, 1] + data[:, 5] + data[:, 9]) / 3


@calcfunction
def kappa_figure(cwd: Str, name: Str, calc_label: Str, ref_label: Str, temp: Float, rta_qmesh: Int, **files) -> dict:
    """kappa(T), phonon lifetime, cumulative kappa and kappa spectrum at temp; MatterSim and the reference.

    files: kl_{tag}, spec_{tag}, tau_{tag}, cum_{tag}, boundary_{tag} for tag in ms (and ref).
    """
    t0 = temp.value
    nq = rta_qmesh.value
    cases = [("ms", calc_label.value, "C0", "o")]
    if "kl_ref" in files:
        cases.append(("ref", f"{ref_label.value} reference IFCs", "C3", "s"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    summary = {}
    for tag, label, color, marker in cases:
        T, k = _kappa_avg(np.loadtxt(files[f"kl_{tag}"].get_content().splitlines()))
        ax = axes[0, 0]
        ax.loglog(T[T > 0], k[T > 0], color=color, lw=1.5, ls="--" if tag == "ref" else "-", label=label)
        total = float(np.interp(t0, T, k))
        summary[tag] = {"kappa_WmK": {int(t): float(np.interp(t, T, k)) for t in (100, 200, 300, 500, 1000) if t <= T.max()}}

        tau = np.loadtxt(files[f"tau_{tag}"].get_content().splitlines())
        tau = tau[tau[:, 3] > 0]
        axes[0, 1].scatter(tau[:, 2] * CM1_TO_THZ, tau[:, 3], s=6, color=color, marker=marker, alpha=0.6, lw=0, label=label)

        L, kc = _kappa_avg(np.loadtxt(files[f"cum_{tag}"].get_content().splitlines()))
        axes[1, 0].plot(L[1:], kc[1:], color=color, lw=1.5, label=label)
        axes[1, 0].axhline(total, color=color, lw=0.6, ls=":")
        summary[tag]["L50_nm"] = float(np.interp(0.5 * total, kc, L))

        spec = np.loadtxt(files[f"spec_{tag}"].get_content().splitlines())
        spec = spec[np.isclose(spec[:, 0], t0)]
        f = spec[:, 1] * CM1_TO_THZ
        sp = spec[:, 2:5].mean(axis=1) / CM1_TO_THZ
        axes[1, 1].plot(f, sp, color=color, lw=1.5, label=label)
        summary[tag]["spectrum_integral_WmK"] = float(np.trapezoid(sp, f))

        if f"boundary_{tag}" in files:
            Tb, kb = _kappa_avg(np.loadtxt(files[f"boundary_{tag}"].get_content().splitlines()))
            summary[tag]["kappa_boundary_1mm_WmK"] = float(np.interp(t0, Tb, kb))

    ax = axes[0, 0]
    ax.axvline(t0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa_L$ (W/mK)")
    ax.set_title(f"{name.value} lattice thermal conductivity (RTA, {nq}x{nq}x{nq})", fontsize=10)
    ax = axes[0, 1]
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon lifetime (ps)")
    ax.set_title(f"{name.value} phonon lifetime at {t0:.0f} K", fontsize=10)
    ax = axes[1, 0]
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mean free path L (nm)")
    ax.set_ylabel(r"Cumulative $\kappa_L$ (W/mK)")
    ax.set_title(f"{name.value} cumulative thermal conductivity at {t0:.0f} K", fontsize=10)
    ax = axes[1, 1]
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$\kappa_L$ spectrum (W/mK/THz)")
    ax.set_title(f"{name.value} thermal conductivity spectrum at {t0:.0f} K", fontsize=10)
    for ax in axes.ravel():
        ax.legend(fontsize=8, markerscale=2.5)
    fig.tight_layout()
    target = os.path.join(cwd.value, f"{name.value}_kappa.png")
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"img_file": SinglefileData(target), "summary": Dict(summary)}


# ---------------------------------------------------------------- workflow

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="alamode tutorial settings; other options override")
    parser.add_argument("--structure", help="structure file (CIF, POSCAR, ...; any format ASE reads)")
    parser.add_argument("--format", default="", help="ASE format name if it cannot be guessed from the file name")
    parser.add_argument("--supercell", type=int, nargs=3, metavar="N", help="supercell multiplicities of the structure-file cell")
    parser.add_argument("--name", help="label of the run (default: formula of the primitive cell)")
    parser.add_argument("--relax", choices=["volume", "full", "none"], default="volume",
                        help="MatterSim relaxation before the phonons: volume only (cell shape kept), cell and positions, or none")
    parser.add_argument("--idealize", action="store_true",
                        help="symmetrize the relaxed cell (spglib) and use its primitive cell as the unit of the supercell; "
                             "needed after --relax full (alm finds 0 free IFCs on a cell with ~1e-6 noise)")
    parser.add_argument("--mag", type=float, help="displacement [A] (default 0.01)")
    parser.add_argument("--qmesh", type=int, help="DOS q mesh (default 20)")
    parser.add_argument("--emax", type=float, help="DOS EMAX [cm^-1] (default 1000)")
    parser.add_argument("--delta-e", type=float, help="DOS DELTA_E [cm^-1] (default 1.0)")
    parser.add_argument("--nonanalytic", type=int, nargs="+", help="anphon NONANALYTIC values (default 0; 3 needs --borninfo)")
    parser.add_argument("--borninfo", help="BORNINFO file (dielectric tensor and Born charges of the primitive cell, in the spglib species order)")
    parser.add_argument("--borninfo-calculator", metavar="NAME",
                        help="compute the Born charges of the primitive cell with this calculator instead (alamode.bec_ase; "
                             "e.g. sevennet-polar). Needs --dielectric unless the model gives the dielectric tensor.")
    parser.add_argument("--borninfo-kwargs", default="{}", help="JSON kwargs of --borninfo-calculator")
    parser.add_argument("--dielectric-model", metavar="NAME",
                        help="predict eps_inf with this model when the Born-charge calculator has none (e.g. anisonet)")
    parser.add_argument("--dielectric", type=float, nargs="+", metavar="E",
                        help="high-frequency dielectric tensor for --borninfo-calculator: 1 (isotropic), 3 (diagonal) or 9 values")
    parser.add_argument("--na-sigma", type=float, help="NA_SIGMA of NONANALYTIC = 1 (default 0.15)")
    parser.add_argument("--kpath", choices=["auto", "tutorial"], help="band path: auto (ASE bravais lattice) or the tutorial's G-X-G-L")
    parser.add_argument("--cubic", action="store_true", default=None, help="also cubic IFCs and the RTA thermal conductivity")
    parser.add_argument("--cubic-cutoff", type=float, help="cutoff [Bohr] of the cubic IFCs (required with --cubic)")
    parser.add_argument("--cubic-mag", type=float, help="displacement [A] for the cubic IFCs (default 0.04)")
    parser.add_argument("--rta-qmesh", type=int, help="q mesh of the RTA calculation (default 10)")
    parser.add_argument("--temp", type=float, help="temperature [K] of the lifetime / cumulative / spectrum analysis (default 300)")
    parser.add_argument("--ref-cubic-xml", help="reference cubic IFC xml (.bz2 / .gz allowed) for the RTA comparison")
    parser.add_argument("--ref-xml", help="reference IFC xml (e.g. DFT) to compare with, through the same anphon step")
    parser.add_argument("--ref-label", help="legend label of the reference")
    parser.add_argument("--calculator", default="mattersim",
                        help="ASE calculator name (aiida_alamode.ase_runner.CALCULATORS: mattersim, mace, chgnet, sevennet, orb, emt, ...)")
    parser.add_argument("--calculator-kwargs", default="{}", help='JSON kwargs of the calculator, e.g. \'{"model": "medium"}\'')
    parser.add_argument("--calc-label", help="legend label of the calculator")
    parser.add_argument("--computer", default=os.environ.get("AIIDA_ALAMODE_COMPUTER", "localhost"),
                        help="AiiDA computer label of the codes alm@..., anphon@..., ... (env AIIDA_ALAMODE_COMPUTER)")
    parser.add_argument("--gpu", action="store_true", help="request one GPU (#SBATCH --gres=gpu:1) for the MatterSim / SevenNet jobs")
    parser.add_argument("--cores", type=int, default=4, help="cores of the MatterSim jobs")
    parser.add_argument("--njobs", type=int, default=1, help="number of slurm jobs for the displaced structures")
    parser.add_argument("--root", default=os.path.join(HERE, "run_alamode_phonons"))
    parser.add_argument("--force", action="store_true", help="ignore .node.json and recompute everything")
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)

    defaults = dict(mag=0.01, qmesh=20, emax=1000.0, delta_e=1.0, nonanalytic=[0], borninfo=None, kpath="auto",
                    na_sigma=0.15, ref_xml=None, ref_label="reference",
                    cubic=False, cubic_cutoff=None, cubic_mag=0.04, rta_qmesh=10, temp=300.0, ref_cubic_xml=None)
    if args.preset:
        p = PRESETS[args.preset]
        if not os.path.isfile(os.path.join(HERE, p["structure"])):
            ase.io.write(os.path.join(HERE, p["structure"]), p["make"](), format="cif")
        defaults.update(structure=os.path.join(HERE, p["structure"]), supercell=p["supercell"], name=args.preset,
                        mag=p["mag"], emax=p["emax"], delta_e=p["delta_e"], nonanalytic=p["nonanalytic"],
                        kpath=p["kpath"], ref_label=p["ref_label"],
                        borninfo=os.path.join(ALAMODE_TEST, p["borninfo"]) if p["borninfo"] else None,
                        ref_xml=os.path.join(ALAMODE_TEST, p["ref_xml"]) if p["ref_xml"] else None,
                        cubic=p.get("cubic", False), cubic_cutoff=p.get("cubic_cutoff"),
                        cubic_mag=p.get("cubic_mag", 0.04), rta_qmesh=p.get("rta_qmesh", 10),
                        ref_cubic_xml=os.path.join(ALAMODE_TEST, p["ref_cubic_xml"]) if p.get("ref_cubic_xml") else None)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if not args.structure or not args.supercell:
        parser.error("--structure and --supercell are required (or --preset)")
    if any(na > 0 for na in args.nonanalytic) and not (args.borninfo or args.borninfo_calculator):
        parser.error("NONANALYTIC > 0 needs --borninfo or --borninfo-calculator")
    args.borninfo_kwargs = json.loads(args.borninfo_kwargs)
    if args.cubic and not args.cubic_cutoff:
        parser.error("--cubic needs --cubic-cutoff [Bohr]")
    args.calculator_kwargs = json.loads(args.calculator_kwargs)
    if args.calc_label is None:
        if args.calculator == "mattersim":
            model = args.calculator_kwargs.get("load_path", "MatterSim-v1.0.0-1M.pth")
            args.calc_label = os.path.basename(model)[:-4] if model.endswith(".pth") else os.path.basename(model)
        else:
            args.calc_label = args.calculator + ("" if not args.calculator_kwargs else " " + json.dumps(args.calculator_kwargs))
    return args




def main():
    args = parse_args()

    code_alm = load_code(f"alm@{args.computer}")
    code_anphon = load_code(f"anphon@{args.computer}")
    code_displace = load_code(f"displace@{args.computer}")
    code_ase = load_code(f"ase_runner@{args.computer}")   # the alamode-ase-runner script
    code_analyze = load_code(f"analyze_phonons@{args.computer}")    # the compiled analyze_phonons
    # slurm allocates whole cores (CR_Core). MatterSim uses the GPU if available, otherwise `cores` threads.
    opt_ase = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": args.cores},
                     "max_wallclock_seconds": 3600}
    opt_serial = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 2},
                  "max_wallclock_seconds": 3600}
    if args.gpu:   # MatterSim / SevenNet jobs (forces, relax, Born charges) on the GPU
        opt_ase["custom_scheduler_commands"] = "#SBATCH --gres=gpu:1"
        opt_serial["custom_scheduler_commands"] = "#SBATCH --gres=gpu:1"

    # --- 0. input structure. The run directory is named after the formula unless --name is given.
    input_atoms = ase.io.read(args.structure, format=args.format or None)
    formula = ase.formula.Formula(input_atoms.get_chemical_formula()).reduce()[0].format("metal")
    name = args.name or formula + ("" if args.calculator == "mattersim" else f"_{args.calculator}")
    root = os.path.join(args.root, name)
    dirs = {key: os.path.join(root, key) for key in ["relax", "harmonic", "phonons", "reference", "cubic", "rta", "reference_rta"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    bank = NodeBank(root, args.force)
    print(f"{name}: {args.structure} ({len(input_atoms)} atoms), supercell {args.supercell}, relax={args.relax}")

    calc_spec = {"name": args.calculator, "kwargs": args.calculator_kwargs}
    calculator = run_cached(bank, "calculator", lambda: Dict(calc_spec))
    if calculator.get_dict() != calc_spec:
        raise SystemExit(f"{root} was made with the calculator {calculator.get_dict()}; use another --name or --root.")
    print(f"calculator: {calc_spec} ({args.calc_label})")

    structure_file = run_cached(bank, "structure_file", lambda: SinglefileData(os.path.abspath(args.structure)))
    unit0 = run_cached(bank, "unit0", lambda: read_structure(structure_file, Str(args.format)))

    # --- 1. MatterSim relaxation
    if args.relax == "none":
        unit = unit0
    else:
        relax = run_cached(bank, f"relax_{args.relax}",
                           lambda: submit_ase("relax_ase", code_ase, unit0, calculator, opt_serial,
                                                    cwd=Str(dirs["relax"]), hydrostatic_strain=Bool(args.relax == "volume")))
        unit = relax.outputs.structure
        r = relax.outputs.results
        print(f"MatterSim relaxed cell [A]: {np.round(r['cell_lengths'], 5)} (input {np.round(unit0.cell_lengths, 5)}), "
              f"{r['nsteps']} steps")

    symprec = Float(1e-4)
    if args.idealize:
        prim = run_cached(bank, "prim", lambda: idealize_structure(unit, Float(1e-3)))
        unit = prim
    else:
        prim = run_cached(bank, "prim", lambda: find_primitive(unit, symprec))
    diag = List(args.supercell)
    supercell = run_cached(bank, "supercell", lambda: make_supercell_structure(unit, diag))
    print(f"primitive cell: {len(prim.sites)} atoms ({prim.get_formula()}), supercell: {len(supercell.sites)} atoms")

    norder = Int(1)
    prefix = Str(AlmPrefixMaker(name=name, kmesh=args.supercell, norder=1).prefix)
    cwd_harm = Str(dirs["harmonic"])

    # --- 2. alm suggest -> 3. displace.py -> 4. forces + DFSET -> 5. alm opt (harmonic IFCs, xml)
    alm_suggest = run_cached(bank, "alm_suggest",
                             lambda: submit_alm(code_alm, "suggest", supercell, prefix, norder, cwd_harm))
    displace = run_cached(bank, "displace",
                          lambda: submit_displace(code_displace, supercell, alm_suggest.outputs.pattern, args.mag, norder, cwd_harm))
    print(f"{displace.outputs.results['number_of_displacements']} displaced structures (mag = {args.mag} A)")
    forces = run_cached(bank, "forces",
                        lambda: submit_forces(code_ase, displace.outputs.displaced_structures, supercell, calculator,
                                              args.njobs, opt_ase, cwd_harm))
    fmax = np.abs(forces.outputs.arrays.get_array("forces")).max(axis=(1, 2))
    print("max |F| of the displaced structures [eV/A]:", np.round(fmax, 5))
    alm_opt = run_cached(bank, "alm_opt",
                         lambda: submit_alm(code_alm, "opt", supercell, prefix, norder, cwd_harm, dfset=forces.outputs.dfset))
    print("alm opt:", alm_opt.outputs.results["optimization"])

    # --- 6. anphon band / DOS with the MatterSim IFCs, and with the reference IFCs through the same step
    borninfo = None
    if args.borninfo:
        borninfo = run_cached(bank, "borninfo", lambda: SinglefileData(os.path.abspath(args.borninfo)))
    elif args.borninfo_calculator:
        # Z* (alamode.bec_ase) and eps_inf (alamode.epsinf_ase with a dielectric model, or a given value)
        # of the primitive cell -> BORNINFO (BornInfoWorkChain); same atom order as the anphon &position
        bec_calculator = run_cached(bank, "bec_calculator",
                                    lambda: Dict({"name": args.borninfo_calculator, "kwargs": args.borninfo_kwargs}))
        inputs = dict(structure=prim, bec=dict(code=code_ase, calculator=bec_calculator, cwd=Str(dirs["phonons"]), options=Dict(opt_serial)))
        if args.dielectric_model:
            inputs["epsinf"] = dict(code=code_ase, dielectric_model=Dict({"name": args.dielectric_model}), cwd=Str(dirs["phonons"]),
                                    options=Dict(opt_serial))
        if args.dielectric:
            inputs["dielectric"] = List(args.dielectric)
        bec = run_cached(bank, "borninfo_wc", lambda: submit(WorkflowFactory("alamode.borninfo"), **inputs))
        r = bec.outputs.results
        print("Born effective charges (diagonal) [e]:", {s: np.round(d, 3).tolist() for s, d in zip(r["symbols"], r["bec_diagonal"])})
        print(f"dielectric tensor ({r['epsilon_inf_source']}):", np.round(r["epsilon_inf"], 3).tolist())
        borninfo = bec.outputs.borninfo
    targets = [("ms", prim, alm_opt.outputs.input_ANPHON, dirs["phonons"])]
    if args.ref_xml:
        ref_xml = run_cached(bank, "ref_xml", lambda: SinglefileData(os.path.abspath(args.ref_xml)))
        ref_prim = run_cached(bank, "ref_prim", lambda: primitive_from_fcsxml(ref_xml, symprec))
        print(f"reference primitive cell [A]: {np.round(ref_prim.cell_lengths, 5)} ({ref_prim.get_formula()})")
        targets.append(("ref", ref_prim, ref_xml, dirs["reference"]))
    qmesh = List([args.qmesh] * 3)

    def anphon_phonons(structure, fcsxml, cwd, na, phonons_mode):
        general = {"NONANALYTIC": na}
        if na == 1:
            general["NA_SIGMA"] = args.na_sigma
        param = {"general": general}
        if phonons_mode == "band":
            if args.kpath == "tutorial":
                param["kpoint"] = KPATH_TUTORIAL     # otherwise the anphon CalcJob makes the ASE path
        else:
            general.update({"EMIN": 0, "EMAX": args.emax, "DELTA_E": args.delta_e})
        return submit_anphon(code_anphon, structure, fcsxml, "phonons", Str(f"{name}_NA{na}"), Str(cwd),
                             phonons_mode=phonons_mode, qmesh=qmesh, param=Dict(param), borninfo=borninfo,
                             options=opt_serial)

    anphon = {}
    pending = []
    for na in args.nonanalytic:
        for kind in ["band", "dos"]:
            for tag, structure, fcsxml, cwd in targets:
                label = f"{kind}_{tag}_NA{na}"
                node = bank.load(label)
                if node is None:
                    node = anphon_phonons(structure, fcsxml, cwd, na, kind)
                    print(f"submitted {label}: {node}")
                    pending.append((label, node))
                anphon[label] = node
    wait([node for _, node in pending])
    for label, node in pending:
        bank.dump(label, node)

    # --- 7. figure and summary
    files = {}
    for label, node in anphon.items():
        files[label] = node.outputs.phband_file if label.startswith("band") else node.outputs.phdos_file
    figure = run_cached(bank, "figure",
                        lambda: phonon_figure(Str(root), Str(name), Str(args.calc_label), Str(args.ref_label), List(args.nonanalytic),
                                              Int(args.qmesh), **files)["img_file"])
    summary = figure.base.links.get_incoming().one().node.outputs.summary.get_dict()
    print("figure:", os.path.join(root, figure.filename))
    print(json.dumps(summary, indent=1))

    # --- 8. harmonic thermodynamics C_v(T), S(T), F(T) from the DOS runs (largest NONANALYTIC asked for)
    na_thermo = max(args.nonanalytic)
    thermo = {}
    for tag, *_ in targets:
        node = anphon[f"dos_{tag}_NA{na_thermo}"]
        if "thermo" in node.outputs:
            thermo[tag] = node.outputs.thermo
        else:   # anphon node from before the 'thermo' output
            thermo[tag] = run_cached(bank, f"thermo_{tag}_NA{na_thermo}", lambda: thermo_arrays(node.outputs.thermo_file))
    thermo_img = run_cached(bank, "thermo_figure",
                            lambda: thermo_figure(Str(root), Str(name), Str(args.calc_label), Str(args.ref_label),
                                                  Int(len(prim.sites)), **thermo)["img_file"])
    thermo_summary = thermo_img.base.links.get_incoming().one().node.outputs.summary.get_dict()
    print("thermo figure:", os.path.join(root, thermo_img.filename))
    print(json.dumps(thermo_summary, indent=1))
    if not args.cubic:
        print(f"done. provenance: verdi node graph generate {figure.pk}")
        return

    # ================= cubic IFCs and thermal conductivity (alamode tutorial steps 5-7)
    norder2 = Int(2)
    prefix2 = Str(AlmPrefixMaker(name=name, kmesh=args.supercell, norder=2).prefix)
    cwd_cubic = Str(dirs["cubic"])
    cutoff = Dict({"*-*": [None, args.cubic_cutoff]})

    alm2_suggest = run_cached(bank, "alm2_suggest",
                              lambda: submit_alm(code_alm, "suggest", supercell, prefix2, norder2, cwd_cubic, cutoff=cutoff))
    # only the cubic patterns (the tutorial's displace.py -pf *.pattern_ANHARM3)
    pattern2 = run_cached(bank, "pattern2", lambda: select_patterns(alm2_suggest.outputs.pattern, List([1])))
    displace2 = run_cached(bank, "displace2",
                           lambda: submit_displace(code_displace, supercell, pattern2, args.cubic_mag, norder2, cwd_cubic))
    print(f"cubic: {displace2.outputs.results['number_of_displacements']} displaced structures "
          f"(mag = {args.cubic_mag} A, cutoff {args.cubic_cutoff} Bohr)")
    forces2 = run_cached(bank, "forces2",
                         lambda: submit_forces(code_ase, displace2.outputs.displaced_structures, supercell, calculator,
                                               args.njobs, opt_ase, cwd_cubic))
    alm2_opt = run_cached(bank, "alm2_opt",
                          lambda: submit_alm(code_alm, "opt", supercell, prefix2, norder2, cwd_cubic, cutoff=cutoff,
                                             dfset=forces2.outputs.dfset, fc2xml=alm_opt.outputs.input_ANPHON))
    print("alm opt (cubic):", alm2_opt.outputs.results["optimization"])

    # --- RTA and analysis, MatterSim and reference
    rta_targets = [("ms", prim, alm2_opt.outputs.input_ANPHON, dirs["rta"])]
    if args.ref_cubic_xml and args.ref_xml:
        ref_cubic_file = run_cached(bank, "ref_cubic_file", lambda: SinglefileData(os.path.abspath(args.ref_cubic_xml)))
        if os.path.splitext(args.ref_cubic_xml)[1] in (".bz2", ".gz"):
            ref_cubic_xml = run_cached(bank, "ref_cubic_xml",
                                       lambda: decompress_file(ref_cubic_file, Str(dirs["reference_rta"])))
        else:
            ref_cubic_xml = ref_cubic_file
        rta_targets.append(("ref", ref_prim, ref_cubic_xml, dirs["reference_rta"]))
    rta_qmesh = List([args.rta_qmesh] * 3)
    temp = Float(args.temp)

    pending = []
    rta = {}
    for tag, structure, fcsxml, cwd in rta_targets:
        node = bank.load(f"rta_{tag}")
        if node is None:
            general = {"EMIN": 0, "EMAX": args.emax, "DELTA_E": args.delta_e}
            if borninfo is not None:   # LO-TO correction in the RTA too (the largest NONANALYTIC asked for)
                general["NONANALYTIC"] = max(args.nonanalytic)
            node = submit_anphon(code_anphon, structure, fcsxml, "RTA", Str(f"{name}_cubic"), Str(cwd), norder=2,
                                 qmesh=rta_qmesh, kappa_spec=1, param=Dict({"general": general}),
                                 borninfo=borninfo if tag == "ms" else None, options=opt_serial)
            print(f"submitted rta_{tag}: {node}")
            pending.append((f"rta_{tag}", node))
        rta[tag] = node
    wait([node for _, node in pending])
    for label, node in pending:
        bank.dump(label, node)

    analyses = {"tau": {"temp": args.temp}, "cum": {"temp": args.temp, "length": "10000:5"}, "boundary": {"size": 1.0e6}}
    calc_names = {"tau": "tau", "cum": "cumulative", "boundary": "kappa_boundary"}

    def submit_analyze(tag, cwd, key):
        builder = CalculationFactory("alamode.analyze_phonons").get_builder()   # the code may carry an old plugin name
        builder.code = code_analyze
        builder.cwd = Str(cwd)
        builder.prefix = Str(f"{name}_cubic")
        builder.calc = Str(calc_names[key])
        builder.file_result = rta[tag].outputs.result_file
        builder.param = Dict(analyses[key])
        builder.metadata.options = opt_serial
        return submit(builder)

    pending = []
    analyze = {}
    for tag, _, _, cwd in rta_targets:
        for key in analyses:
            label = f"{key}_{tag}"
            node = bank.load(label)
            if node is None:
                node = submit_analyze(tag, cwd, key)
                print(f"submitted {label}: {node}")
                pending.append((label, node))
            analyze[label] = node
    wait([node for _, node in pending])
    for label, node in pending:
        bank.dump(label, node)

    files = {}
    for tag, _, _, _ in rta_targets:
        files[f"kl_{tag}"] = rta[tag].outputs.kl_file
        files[f"spec_{tag}"] = rta[tag].outputs.kl_spec_file
        files[f"tau_{tag}"] = analyze[f"tau_{tag}"].outputs.tau_file
        files[f"cum_{tag}"] = analyze[f"cum_{tag}"].outputs.cumulative_file
        files[f"boundary_{tag}"] = analyze[f"boundary_{tag}"].outputs.kappa_boundary_file
    figure2 = run_cached(bank, "figure_kappa",
                         lambda: kappa_figure(Str(root), Str(name), Str(args.calc_label), Str(args.ref_label), temp, Int(args.rta_qmesh),
                                              **files)["img_file"])
    summary2 = figure2.base.links.get_incoming().one().node.outputs.summary.get_dict()
    print("figure:", os.path.join(root, figure2.filename))
    print(json.dumps(summary2, indent=1))
    print(f"done. provenance: verdi node graph generate {figure2.pk}")


if __name__ == "__main__":
    main()
