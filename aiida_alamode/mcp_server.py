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
"""MCP server of aiida-alamode: phonon calculations with ALAMODE + AiiDA driven by an LLM.

    alamode-mcp                       # stdio transport (what Claude Code / Claude Desktop start)
    alamode-mcp --transport streamable-http   # HTTP on 127.0.0.1:8000

Tools (each returns JSON-serialisable data):
    check_packages   optional ML packages (MatterSim, SevenNet-Polar, Equivar, AnisoNet, ...) on a computer
    list_codes       AiiDA codes of ALAMODE (alm, anphon, displace, analyze_phonons, ase_runner) per computer
    run_phonons      launch example/run_alamode_phonons.py in the background (harmonic phonons, LO-TO, cubic + RTA)
    run_driver       launch any driver (phonons, scph, qha) with raw command-line arguments
    run_status       is the driver alive, its last log lines, the state of every AiiDA node it recorded
    run_results      Z*, eps_inf, Gamma-point frequencies with / without the LO-TO correction, imaginary modes,
                     thermodynamics, figures of a finished run
    list_runs        runs under a root directory
    process_info     one AiiDA process: state, exit code, inputs / outputs, report, remote directory
    kill_run         stop a driver (the AiiDA processes already submitted keep running)
Resources:
    aiida-alamode://skill         the Claude Code skill (how the plugin works, pitfalls, reference numbers)
    aiida-alamode://docs/{name}   docs/<name>.md

Environment: AIIDA_PROFILE (default profile), AIIDA_ALAMODE_COMPUTER (default computer),
AIIDA_ALAMODE_EXAMPLE_DIR (where the drivers are; default <repo>/example), AIIDA_ALAMODE_RUN_ROOT
(default <example>/run_mcp), AIIDA_ALAMODE_MCP_MAX_CHARS (a result longer than this is written to a JSON
file and only a summary with the path is returned; default 20000).
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Optional

import numpy as np

from mcp.server.mcpserver import MCPServer

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)
EXAMPLE_DIR = os.environ.get("AIIDA_ALAMODE_EXAMPLE_DIR", os.path.join(REPO_DIR, "example"))
FIGURE_SUFFIXES = (".png", ".svg", ".pdf")   # figure files of the drivers (--figure-format)
RUN_ROOT = os.environ.get("AIIDA_ALAMODE_RUN_ROOT", os.path.join(EXAMPLE_DIR, "run_mcp"))
RUNS_FILE = os.path.join(RUN_ROOT, ".mcp_runs.json")
# a tool result longer than this (characters of JSON) is written to a file and only a short summary is returned:
# the client's context window is the scarce resource, not the disk
MAX_RESULT_CHARS = int(os.environ.get("AIIDA_ALAMODE_MCP_MAX_CHARS", "20000"))
DRIVERS = {"phonons": "run_alamode_phonons.py", "scph": "run_alamode_scph.py", "qha": "run_alamode_qha.py"}
CM1_TO_THZ = 0.0299792458

server = MCPServer(
    "aiida-alamode",
    instructions=(
        "Phonon calculations with ALAMODE through AiiDA. Typical order: check_packages (the ML packages are optional and "
        "must be on the computer of the ase_runner code) -> run_phonons -> run_status until done -> run_results. "
        "Every number comes with the pk of the AiiDA node it was read from; process_info(pk) shows the provenance. "
        "Read the resource aiida-alamode://skill first for the pitfalls (supercell images, BORNINFO order, 0 free IFCs "
        "after a full relaxation -> idealize=True, imaginary modes)."
    ),
)


# ----------------------------------------------------------------------------------------------- helpers
def _aiida():
    """load the AiiDA profile once (lazily, so that `alamode-mcp --help` works without a profile)"""
    import aiida
    from aiida.manage import get_manager
    if get_manager().get_profile() is None:
        aiida.load_profile(os.environ.get("AIIDA_PROFILE"))
    return aiida


def _computer(computer: Optional[str]) -> str:
    computer = computer or os.environ.get("AIIDA_ALAMODE_COMPUTER")
    if not computer:
        raise ValueError("give computer, or set AIIDA_ALAMODE_COMPUTER")
    return computer


def _load_runs() -> dict:
    if os.path.isfile(RUNS_FILE):
        with open(RUNS_FILE) as f:
            return json.load(f)
    return {}


def _save_runs(runs: dict):
    os.makedirs(RUN_ROOT, exist_ok=True)
    with open(RUNS_FILE, "w") as f:
        json.dump(runs, f, indent=1)


def _deliver(result: dict, path: str, summary: Optional[dict] = None) -> dict:
    """return result as it is when it is small; otherwise write it to path (JSON) and return summary (or the top-level
    keys with the size of each) plus the path, so that the caller can grep / read the file in pieces"""
    text = json.dumps(result, indent=1, default=str)
    if len(text) <= MAX_RESULT_CHARS and summary is None:
        return result
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    out = dict(summary) if summary is not None else {
        "keys": {k: len(json.dumps(v, default=str)) for k, v in result.items()}}
    out["full_result"] = path
    out["full_result_chars"] = len(text)
    if summary is None:
        out["note"] = f"result longer than {MAX_RESULT_CHARS} characters: written to full_result; 'keys' gives the size of each top-level key"
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    with open(f"/proc/{pid}/stat") as f:   # a zombie (finished, not yet reaped) is not alive
        return f.read().split(")")[-1].split()[0] != "Z"


def _tail(path: str, lines: int) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, errors="replace") as f:
        return [line.rstrip("\n") for line in f.readlines()[-lines:]]


def _launch(driver: str, args: list, root: str, name_hint: str) -> dict:
    script = os.path.join(EXAMPLE_DIR, DRIVERS[driver])
    if not os.path.isfile(script):
        raise FileNotFoundError(f"{script} not found; set AIIDA_ALAMODE_EXAMPLE_DIR")
    os.makedirs(root, exist_ok=True)
    run_id = f"{name_hint}_{driver}_{time.strftime('%Y%m%d-%H%M%S')}"
    log = os.path.join(root, f"{run_id}.log")
    command = [sys.executable, "-u", script] + [str(a) for a in args] + ["--root", root]
    with open(log, "w") as f:
        proc = subprocess.Popen(command, cwd=EXAMPLE_DIR, stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    runs = _load_runs()
    runs[run_id] = {"driver": driver, "pid": proc.pid, "log": log, "root": root, "command": command,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_runs(runs)
    return {"run_id": run_id, "pid": proc.pid, "log": log, "root": root, "command": " ".join(command)}


def _run_dir(run: dict) -> Optional[str]:
    """<root>/<name> of a run, from the first log line `<name>: <structure> (...)`"""
    for line in _tail(run["log"], 3)[:1] + _tail(run["log"], 10 ** 6)[:3]:
        m = re.match(r"^([\w.+-]+): .*\(\d+ atoms\)", line)
        if m:
            return os.path.join(run["root"], m.group(1))
    return None


def _node_states(node_json: str) -> dict:
    """label -> {pk, process_label, state, exit_status} for every node in <run>/.node.json"""
    if not os.path.isfile(node_json):
        return {}
    _aiida()
    from aiida.orm import load_node, ProcessNode
    with open(node_json) as f:
        pks = json.load(f)
    out = {}
    for label, pk in pks.items():
        try:
            node = load_node(pk)
        except Exception as err:
            out[label] = {"pk": pk, "error": str(err)}
            continue
        info = {"pk": pk, "type": node.node_type.split(".")[-2] if node.node_type.endswith(".") else node.node_type}
        if isinstance(node, ProcessNode):
            info.update(process_label=node.process_label, state=node.process_state.value if node.process_state else None,
                        exit_status=node.exit_status)
        out[label] = info
    return out


from aiida_alamode.report import gamma_frequencies as _gamma_frequencies   # noqa: E402


# ----------------------------------------------------------------------------------------------- tools
@server.tool()
def check_packages(computer: Optional[str] = None, names: Optional[list[str]] = None) -> dict:
    """Which optional packages the computer of the code ase_runner@<computer> has: mattersim (forces),
    sevennet-polar / equivar (Born charges Z*, 10 elements Ba Ca Hf Li O P Pb Sr Ti Zr), anisonet (eps_inf),
    mace, chgnet, sevennet, orb, emt. The plugin itself needs none of them (ALAMODE + aiida-core only).
    names: subset to check (default all). Returns {name: {ok, detail}}."""
    _aiida()
    from aiida.orm import load_code
    from aiida_alamode.ase_runner import PACKAGES
    from aiida_alamode.optional_packages import check_on_computer
    code = load_code(f"ase_runner@{_computer(computer)}")
    return check_on_computer(code, names or list(PACKAGES))


@server.tool()
def list_codes() -> dict:
    """AiiDA computers and the ALAMODE codes on them (alm, anphon, displace, analyze_phonons, ase_runner).
    A computer is usable by the drivers when it has all five."""
    _aiida()
    from aiida.orm import QueryBuilder, Computer, Code
    out = {}
    for computer in QueryBuilder().append(Computer).all(flat=True):
        out[computer.label] = {"hostname": computer.hostname, "transport": computer.transport_type,
                               "scheduler": computer.scheduler_type, "codes": {}}
    for code in QueryBuilder().append(Code).all(flat=True):
        try:
            label = code.computer.label
        except Exception:
            continue
        out.setdefault(label, {"codes": {}})["codes"][code.label] = {
            "pk": code.pk, "executable": str(getattr(code, "filepath_executable", "")),
            "default_plugin": code.default_calc_job_plugin}
    needed = {"alm", "anphon", "displace", "analyze_phonons", "ase_runner"}
    for label, info in out.items():
        info["complete"] = needed <= set(info["codes"])
    out["_default_computer"] = os.environ.get("AIIDA_ALAMODE_COMPUTER")
    return out


@server.tool()
def run_phonons(structure: str, supercell: list[int], computer: Optional[str] = None, name: Optional[str] = None,
                calculator: str = "mattersim", nonanalytic: Optional[list[int]] = None,
                borninfo_calculator: Optional[str] = None, dielectric_model: Optional[str] = None,
                dielectric: Optional[list[float]] = None, born_charges: Optional[dict[str, list[float]]] = None,
                borninfo: Optional[str] = None, relax: str = "volume", idealize: bool = False,
                cubic: bool = False, cubic_cutoff: Optional[float] = None, emax: Optional[float] = None,
                gpu: bool = False, figure_format: Optional[list[str]] = None, root: Optional[str] = None,
                extra_args: Optional[list[str]] = None) -> dict:
    """Launch run_alamode_phonons.py in the background: relax -> alm suggest -> displace -> forces (ASE calculator)
    -> alm opt -> anphon band / DOS (+ thermodynamics figure); with cubic=True also cubic IFCs and the RTA
    thermal conductivity. Finished steps are reused when the same name is run again.

    structure: CIF / POSCAR path (relative to the example dir, or absolute). supercell: [n1, n2, n3] multiplying
    the cell in the file. nonanalytic: anphon NONANALYTIC values, e.g. [0, 3] for bands without and with the
    LO-TO correction (needs Z* and eps_inf: borninfo_calculator 'sevennet-polar' | 'equivar' with
    dielectric_model 'anisonet' or dielectric [eps]; or born_charges {'Mg': [1.96], 'O': [-1.96]} (1, 3 or 9
    values per species) with dielectric_model / dielectric; or a BORNINFO file). relax: 'volume' | 'full' | 'none';
    use relax='full', idealize=True for low-symmetry cells. figure_format: ['png'] (default), ['svg'], ['pdf'] or
    several; re-running a finished run with another format only redraws the figures. Returns run_id, pid, log;
    poll with run_status."""
    args = ["--structure", structure, "--supercell", *supercell, "--computer", _computer(computer),
            "--calculator", calculator, "--relax", relax]
    if name:
        args += ["--name", name]
    if nonanalytic:
        args += ["--nonanalytic", *nonanalytic]
    if borninfo_calculator:
        args += ["--borninfo-calculator", borninfo_calculator]
    if dielectric_model:
        args += ["--dielectric-model", dielectric_model]
    if dielectric:
        args += ["--dielectric", *dielectric]
    if born_charges:
        args += ["--born-charges"] + [f"{sym}:{','.join(str(v) for v in vals)}" for sym, vals in born_charges.items()]
    if borninfo:
        args += ["--borninfo", borninfo]
    if idealize:
        args += ["--idealize"]
    if cubic:
        args += ["--cubic", "--cubic-cutoff", cubic_cutoff or 8.0]
    if emax:
        args += ["--emax", emax]
    if gpu:
        args += ["--gpu"]
    if figure_format:
        args += ["--figure-format", *figure_format]
    args += extra_args or []
    hint = name or os.path.splitext(os.path.basename(structure))[0].split("_")[0]
    return _launch("phonons", args, root or RUN_ROOT, hint)


@server.tool()
def run_driver(driver: str, args: list[str], root: Optional[str] = None) -> dict:
    """Launch a driver with raw command-line arguments: driver 'phonons' (run_alamode_phonons.py), 'scph'
    (run_alamode_scph.py: MD + LASSO anharmonic IFCs + SCPH, BaTiO3 by default) or 'qha' (run_alamode_qha.py:
    strained IFCs + elastic constants + QHA, ZnO by default). '--root' is added by the server. Give '--computer'
    in args unless AIIDA_ALAMODE_COMPUTER is set. Returns run_id, pid, log."""
    if driver not in DRIVERS:
        raise ValueError(f"driver must be one of {list(DRIVERS)}")
    if "--computer" not in args and os.environ.get("AIIDA_ALAMODE_COMPUTER"):
        args = list(args) + ["--computer", os.environ["AIIDA_ALAMODE_COMPUTER"]]
    hint = args[args.index("--name") + 1] if "--name" in args else driver
    return _launch(driver, list(args), root or RUN_ROOT, hint)


@server.tool()
def run_status(run_id: str, log_lines: int = 15) -> dict:
    """Progress of a launched driver: alive or finished (exit code), the last log lines, the run directory, and
    the state of every AiiDA node the driver recorded in <run>/.node.json (submitted, waiting, finished, failed).
    'done' is true when the driver printed its final 'done. provenance' line."""
    runs = _load_runs()
    if run_id not in runs:
        raise KeyError(f"unknown run_id {run_id}; known: {list(runs)}")
    run = runs[run_id]
    alive = _alive(run["pid"])
    text = "\n".join(_tail(run["log"], 10 ** 6))
    run_dir = _run_dir(run)
    out = {"run_id": run_id, "alive": alive, "done": "done. provenance" in text, "log": run["log"],
           "run_dir": run_dir, "started": run["started"], "log_tail": _tail(run["log"], log_lines)}
    if not alive:
        m = re.findall(r"Traceback|Error|missing on ", text)
        out["failed"] = bool(m) and not out["done"]
    if run_dir:
        out["nodes"] = _node_states(os.path.join(run_dir, ".node.json"))
    return out


@server.tool()
def run_results(run_dir: str) -> dict:
    """Results of a (finished) phonon run directory <root>/<name>: the relaxed cell, Z* and eps_inf that went into
    BORNINFO (with their source: a model or given values), the alm fitting error, the Gamma-point frequencies for
    every NONANALYTIC value (LO-TO splitting = the highest Gamma mode NA0 -> NA3), whether imaginary modes occur,
    the harmonic thermodynamics (ZPE, C_v / 3Nk_B, S; 'thermo'), the band / DOS summary, the RTA kappa if
    computed ('kappa'), and the figure files. Every value carries the pk of the AiiDA node it was read from."""
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(RUN_ROOT, run_dir)
    node_json = os.path.join(run_dir, ".node.json")
    if not os.path.isfile(node_json):
        raise FileNotFoundError(f"{node_json} not found: give <root>/<name> of a run")
    _aiida()
    from aiida.orm import load_node, ProcessNode
    with open(node_json) as f:
        pks = json.load(f)
    out = {"run_dir": run_dir, "nodes": pks}
    name = os.path.basename(run_dir.rstrip("/"))

    def outputs(label):
        """outputs of the process node recorded under label (the driver may record a data node instead of the
        calcfunction that made it, e.g. the figure files: then the creator's outputs)"""
        try:
            node = load_node(pks[label])
            if not isinstance(node, ProcessNode):
                node = node.creator
            return node.outputs if node is not None else None
        except Exception:
            return None

    for label in ("relax_volume", "relax_full"):
        o = outputs(label) if label in pks else None
        if o is not None and "results" in o:
            r = o.results.get_dict()
            out["relaxed_cell"] = {"pk": pks[label], "cell_lengths_A": r.get("cell_lengths"), "steps": r.get("steps")}
    if "borninfo_wc" in pks:
        o = outputs("borninfo_wc")
        if o is not None and "results" in o:
            r = o.results.get_dict()
            out["borninfo"] = {"pk": pks["borninfo_wc"], "symbols": r["symbols"], "bec_diagonal": r["bec_diagonal"],
                               "bec_source": r.get("bec_source") or "model", "asr_residual": r["asr_residual"],
                               "epsilon_inf": r["epsilon_inf"], "epsilon_inf_source": r["epsilon_inf_source"]}
    elif "borninfo" in pks:
        out["borninfo"] = {"pk": pks["borninfo"], "source": "file"}
    if "alm_opt" in pks:
        o = outputs("alm_opt")
        if o is not None and "results" in o:
            out["alm_opt"] = {"pk": pks["alm_opt"], **o.results.get_dict().get("optimization", {})}
    phonons = {}
    for label, pk in pks.items():
        m = re.match(r"band_ms_NA(\d)$", label)
        if m:
            bands = os.path.join(run_dir, "phonons", f"{name}_NA{m.group(1)}_{name}_NA{m.group(1)}.bands")
            if os.path.isfile(bands):
                phonons[f"NA{m.group(1)}"] = {"pk": pk, **_gamma_frequencies(bands)}
    if phonons:
        out["phonons"] = phonons
        if "NA0" in phonons and len(phonons) > 1:
            last = sorted(phonons)[-1]
            out["lo_to_shift_THz"] = {"highest_gamma_mode": [phonons["NA0"]["gamma_highest_THz"], phonons[last]["gamma_highest_THz"]],
                                      "from": "NA0", "to": last}
    for label, key in (("figure", "band_dos_summary"), ("thermo_figure", "thermo"), ("figure_kappa", "kappa")):
        # the figure calcfunctions return a `summary` Dict next to the image; the bank key is `label` for the
        # default PNG and `label[svg,...]` for other --figure-format values
        for stored in (label, *sorted(k for k in pks if k.startswith(label + "["))):
            if stored in pks:
                o = outputs(stored)
                if o is not None and "summary" in o:
                    out[key] = {"pk": pks[stored], **o.summary.get_dict()}
                    break
    if "rta_ms" in pks:
        out.setdefault("kappa", {})["rta_pk"] = pks["rta_ms"]
    out["figures"] = sorted(os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(FIGURE_SUFFIXES))
    return _deliver(out, os.path.join(run_dir, f"{name}_results.json"))


@server.tool()
def run_report(target: str, output: Optional[str] = None) -> dict:
    """Write the HTML report of a calculation, discovered from the AiiDA provenance graph. target: a structure
    (or any node) pk, or a run directory <root>/<name> (its input cell is the root). The report follows the
    structure lineage and every process downstream (relaxation, alm, displacements, forces / MD, Z* and eps_inf,
    anphon band / DOS / RTA / SCPH / QHA, analyze_phonons, figures) and shows: full formula, space group, atoms,
    Wyckoff sites, what was done step by step, the fits, the Gamma frequencies per NONANALYTIC value (LO-TO
    shift), harmonic thermodynamics, RTA kappa, SCPH / QHA results, the SVG figures inline (PNG as base64 when no
    SVG exists) and a process graph. Returns the HTML path, the key numbers (every value with its pk) and the path
    of the JSON file with all collected data (written next to the HTML as <stem>.json). output: HTML path (default
    <run_dir>/<name>_report.html, or <run root>/<formula>_pk<pk>_report.html)."""
    from aiida_alamode.report import collect, render, root_of_run_dir, public, summary
    _aiida()
    if str(target).isdigit():
        data = collect(int(target))
        out = output or os.path.join(RUN_ROOT, f"{data['formula']}_pk{data['root_pk']}_report.html")
    else:
        run_dir = target if os.path.isabs(target) else os.path.join(RUN_ROOT, target)
        data = collect(root_of_run_dir(run_dir), name=os.path.basename(run_dir.rstrip("/")), run_dir=run_dir)
        out = output or os.path.join(run_dir, f"{data['name']}_report.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(data))
    return {"html": out, **_deliver(public(data), os.path.splitext(out)[0] + ".json", summary=summary(data))}


@server.tool()
def list_runs(root: Optional[str] = None) -> dict:
    """Runs (directories with .node.json) under root (default the server's run root), with the figures present
    and the last modification time; plus the drivers launched through this server."""
    root = root or RUN_ROOT
    runs = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isfile(os.path.join(d, ".node.json")):
                runs[name] = {"run_dir": d, "figures": sorted(f for f in os.listdir(d) if f.endswith(FIGURE_SUFFIXES)),
                              "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(d, ".node.json"))))}
    launched = {rid: {**r, "alive": _alive(r["pid"])} for rid, r in _load_runs().items()}
    return _deliver({"root": root, "runs": runs, "launched": launched}, os.path.join(RUN_ROOT, "list_runs.json"))


@server.tool()
def process_info(pk: int, report_lines: int = 20) -> dict:
    """One AiiDA process node: label, state, exit status and message, input and output link labels (with pks),
    the remote working directory of a CalcJob, and the last lines of its report (verdi process report)."""
    _aiida()
    from aiida.orm import load_node, ProcessNode, CalcJobNode
    from aiida.cmdline.utils.common import get_calcjob_report, get_workchain_report
    node = load_node(pk)
    out = {"pk": pk, "uuid": node.uuid, "node_type": node.node_type, "label": node.label, "ctime": str(node.ctime)}
    if isinstance(node, ProcessNode):
        out.update(process_label=node.process_label, process_type=node.process_type,
                   state=node.process_state.value if node.process_state else None,
                   exit_status=node.exit_status, exit_message=node.exit_message)
        out["inputs"] = {link.link_label: {"pk": link.node.pk, "type": link.node.node_type.split(".")[-2]}
                         for link in node.base.links.get_incoming().all()}
        out["outputs"] = {link.link_label: {"pk": link.node.pk, "type": link.node.node_type.split(".")[-2]}
                          for link in node.base.links.get_outgoing().all()}
        try:
            report = get_calcjob_report(node) if isinstance(node, CalcJobNode) else get_workchain_report(node, "REPORT")
            out["report"] = report.splitlines()[-report_lines:]
        except Exception as err:
            out["report"] = [f"(no report: {err})"]
        if isinstance(node, CalcJobNode):
            out["remote_workdir"] = node.get_remote_workdir()
            out["scheduler_state"] = str(node.get_scheduler_state()) if node.get_scheduler_state() else None
    return _deliver(out, os.path.join(RUN_ROOT, f"process_{pk}.json"))


@server.tool()
def kill_run(run_id: str) -> dict:
    """Stop a launched driver (SIGTERM to its process group). AiiDA processes it already submitted keep running;
    kill those with process_info to find the pks and `verdi process kill <pk>`."""
    runs = _load_runs()
    if run_id not in runs:
        raise KeyError(f"unknown run_id {run_id}")
    pid = runs[run_id]["pid"]
    if not _alive(pid):
        return {"run_id": run_id, "killed": False, "reason": "not running"}
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    return {"run_id": run_id, "killed": True, "pid": pid}


# ----------------------------------------------------------------------------------------------- resources
@server.resource("aiida-alamode://skill", name="aiida-alamode skill", mime_type="text/markdown",
                 description="how the plugin works, the drivers, the pitfalls and reference numbers")
def skill_resource() -> str:
    path = os.path.join(REPO_DIR, ".claude", "skills", "aiida-alamode", "SKILL.md")
    with open(path) as f:
        return f.read()


@server.resource("aiida-alamode://docs/{name}", name="aiida-alamode docs", mime_type="text/markdown",
                 description="docs/<name>.md: born_effective_charges, examples_workflow, examples_materials, "
                             "remote_gpu_computer, slurm_invalidaccount, rabbitmq, mcp_server")
def docs_resource(name: str) -> str:
    path = os.path.join(REPO_DIR, "docs", f"{os.path.basename(name).removesuffix('.md')}.md")
    with open(path) as f:
        return f.read()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    args = parser.parse_args()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
