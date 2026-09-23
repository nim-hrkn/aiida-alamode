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
ASE-calculator (MatterSim by default) replacement of the DFT / LAMMPS force calculations.

The code is the console script ``alamode-ase-runner`` (alias ``alamode-mattersim``; aiida_alamode.ase_runner), which runs
inside the scheduler job and reads a json job file.

- MattersimForcesCalculation: energy, forces, stress of displaced structures (TrajectoryData).
- MattersimRelaxCalculation: relax a StructureData (cell volume by default, or cell and positions).
- MattersimMdCalculation: NVT MD of a supercell; sampled snapshots + random displacements -> TrajectoryData
  (replacement of AIMD + displace.py -md --random).
- MattersimBecCalculation: Born effective charges (SevenNet-Polar) of the primitive cell -> BORNINFO file
  of anphon (NONANALYTIC > 0), with the dielectric tensor from the model or given as an input.
- MattersimElasticCalculation: clamped-ion elastic constants (SOEC, TOEC) and the strain-force coupling
  of a cell -> elastic_constants.in, strain_force.in for the anphon QHA structural optimization.

The ``calculator`` input selects the ASE calculator, e.g. {"name": "mace", "kwargs": {"model": "medium"}};
see ase_runner.CALCULATORS.  Without it, ``model`` / ``device`` select MatterSim.

If 'cwd' is given, the retrieved files are saved in that directory (as the other alamode CalcJobs).
"""
import io
import json
import os
from fnmatch import fnmatch

import numpy as np
import ase
import ase.io

from aiida.orm import Str, Int, Float, Bool, Dict, List
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory

from ..common.base import AlamodeBaseCalculation
from ..io.aiida_support import save_output_folder_files

FolderData = DataFactory('core.folder')
SinglefileData = DataFactory('core.singlefile')
ArrayData = DataFactory('core.array')
StructureData = DataFactory('core.structure')
TrajectoryData = DataFactory('core.array.trajectory')

_JOB_FILENAME = "mattersim_job.json"
_RESULT_FILENAME = "mattersim_results.json"
_MODEL = "MatterSim-v1.0.0-1M.pth"
_ARRAY_NAMES = ["energies", "forces", "stresses", "positions", "cells"]


def _write_extxyz(folder: Folder, filename: str, atoms: ase.Atoms):
    """ase.io.write cannot write into a sandbox handle directly: go through a StringIO."""
    buffer = io.StringIO()
    ase.io.write(buffer, atoms, format="extxyz")
    with folder.open(filename, 'w', encoding='utf8') as handle:
        handle.write(buffer.getvalue())


class MattersimBaseCalculation(AlamodeBaseCalculation):
    """common inputs: calculator selection, cwd, job / result files."""
    _PARSER = 'alamode.mattersim'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str, required=False, help='directory where results are saved.')
        spec.input("prefix", valid_type=Str, default=lambda: Str(""),
                   help='string added to the file names saved in cwd.')
        spec.input("model", valid_type=Str, default=lambda: Str(_MODEL), help='MatterSim model file')
        spec.input("device", valid_type=Str, default=lambda: Str("auto"), help="auto, cpu or cuda")
        spec.input("calculator", valid_type=Dict, required=False,
                   help='ASE calculator, e.g. {"name": "mace", "kwargs": {"model": "medium"}}; '
                        'see ase_runner.CALCULATORS. Overrides model / device.')
        spec.inputs['metadata']['options']['parser_name'].default = cls._PARSER
        spec.inputs['metadata']['options']['input_filename'].default = _JOB_FILENAME
        spec.inputs['metadata']['options']['output_filename'].default = 'mattersim.out'
        spec.inputs['metadata']['options']['withmpi'].default = False
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        spec.output('results', valid_type=Dict, help="summary: calculator, timing, ...")
        spec.exit_code(340, 'ERROR_OUTPUT_RESULT_MISSING', message='mattersim_results.json was not retrieved.')
        spec.exit_code(341, 'ERROR_OUTPUT_INCOMPLETE', message='the number of results differs from the inputs.')
        spec.exit_code(342, 'ERROR_NOT_CONVERGED', message='the relaxation did not converge.')

    def _job_base(self, mode: str, files: list, input_format: str) -> dict:
        job = {"mode": mode, "files": files, "input_format": input_format,
               "model": self.inputs.model.value, "device": self.inputs.device.value,
               "output": _RESULT_FILENAME}
        if "calculator" in self.inputs:
            job["calculator"] = self.inputs.calculator.get_dict()
        return job

    def _calcinfo(self, folder: Folder, job: dict, retrieve: list) -> CalcInfo:
        with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
            json.dump(job, handle, indent=1)
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [self.options.input_filename]
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = self.options.withmpi
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['_aiidasubmit.sh', self.options.input_filename,
                                  self.options.output_filename, _RESULT_FILENAME] + retrieve
        return calcinfo


class MattersimForcesCalculation(MattersimBaseCalculation):
    """energy, forces and stress of every structure of a TrajectoryData (or of a subset, indices).

    outputs: arrays (ArrayData: energies [eV], forces [eV/A], stresses [eV/A^3], positions [A], cells [A],
    attribute 'indices'), results (Dict).
    """
    _PARSER = 'alamode.mattersim'
    _FILENAME = "structure{:04d}.extxyz"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structures", valid_type=TrajectoryData, help="displaced structures")
        spec.input("indices", valid_type=List, required=False,
                   help="step indices of structures to compute; all if not given.")
        spec.output('arrays', valid_type=ArrayData,
                    help="energies [eV], forces [eV/A], stresses [eV/A^3], positions [A], cells [A]")

    def _indices(self) -> list:
        if "indices" in self.inputs:
            return [int(i) for i in self.inputs.indices.get_list()]
        return list(range(self.inputs.structures.numsteps))

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        indices = self._indices()
        files = []
        for i in indices:
            filename = self._FILENAME.format(i)
            atoms = self.inputs.structures.get_step_structure(i).get_ase()
            _write_extxyz(folder, filename, atoms)
            files.append(filename)
        job = self._job_base("forces", files, "extxyz")
        job["indices"] = indices
        return self._calcinfo(folder, job, ["*.mattersim.extxyz"])


def results_to_arraydata(structures: list, indices: list) -> ArrayData:
    array = ArrayData()
    array.set_array("energies", np.array([s["energy"] for s in structures]))
    array.set_array("forces", np.array([s["forces"] for s in structures]))
    array.set_array("stresses", np.array([s["stress"] for s in structures]))
    array.set_array("positions", np.array([s["positions"] for s in structures]))
    array.set_array("cells", np.array([s["cell"] for s in structures]))
    array.set_array("indices", np.array(indices, dtype=int))
    array.base.attributes.set("symbols", structures[0]["symbols"] if structures else [])
    return array


class MattersimBaseParser(Parser):

    def _load(self):
        """-> (output_folder, result dict) or an exit code"""
        try:
            output_folder = self.retrieved
        except Exception:
            return None, self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        cwd = self.node.inputs.cwd.value if "cwd" in self.node.inputs else ""
        save_output_folder_files(output_folder, cwd, self.node.inputs.prefix)
        if _RESULT_FILENAME not in output_folder.list_object_names():
            return None, self.exit_codes.ERROR_OUTPUT_RESULT_MISSING
        result = json.loads(output_folder.get_object_content(_RESULT_FILENAME))
        return output_folder, result


class MattersimParser(MattersimBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        structures = result.pop("structures")
        job = json.loads(output_folder.get_object_content(self.node.get_option('input_filename')))
        if len(structures) != len(job["files"]):
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        result["files"] = [s["filename"] for s in structures]
        self.out("results", Dict(dict=result))
        self.out("arrays", results_to_arraydata(structures, job["indices"]))


class MattersimRelaxCalculation(MattersimBaseCalculation):
    """relax the cell and the atomic positions of a structure.

    By default only the cell volume is relaxed (hydrostatic strain), as in the MatterSim
    replacement of the DFT lattice-constant optimization.
    """
    _PARSER = 'alamode.mattersim_relax'
    _INPUT_STRUCTURE_FILENAME = "structure.extxyz"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("fmax", valid_type=Float, default=lambda: Float(1e-4),
                   help="convergence criterion [eV/A] of BFGS on FrechetCellFilter")
        spec.input("steps", valid_type=Int, default=lambda: Int(500))
        spec.input("hydrostatic_strain", valid_type=Bool, default=lambda: Bool(True),
                   help="True: only the volume is relaxed, keeping the cell shape.")
        spec.output('structure', valid_type=StructureData, help="the relaxed structure")

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        _write_extxyz(folder, self._INPUT_STRUCTURE_FILENAME, self.inputs.structure.get_ase())
        job = self._job_base("relax", [self._INPUT_STRUCTURE_FILENAME], "extxyz")
        job.update({"fmax": self.inputs.fmax.value, "steps": self.inputs.steps.value,
                    "hydrostatic_strain": self.inputs.hydrostatic_strain.value})
        return self._calcinfo(folder, job, ["relax.log", "*.mattersim.extxyz"])


class MattersimRelaxParser(MattersimBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        s = result.pop("structures")[0]
        atoms = ase.Atoms(symbols=s["symbols"], positions=s["positions"], cell=s["cell"], pbc=True)
        result.update({"energy": s["energy"], "stress": s["stress"], "cell": s["cell"],
                       "cell_lengths": atoms.cell.lengths().tolist(), "cell_angles": atoms.cell.angles().tolist(),
                       "volume": float(atoms.get_volume())})
        self.out("results", Dict(dict=result))
        self.out("structure", StructureData(ase=atoms))
        if not result.get("converged", False):
            return self.exit_codes.ERROR_NOT_CONVERGED


class MattersimMdCalculation(MattersimBaseCalculation):
    """NVT (Langevin) MD of the supercell; sampled snapshots with random displacements as displaced structures
    (displace.py -md ... -e start:end:interval --random --mag).

    outputs: displaced_structures (TrajectoryData, for ForcesMattersimWorkChain), results (Dict).
    """
    _PARSER = 'alamode.mattersim_md'
    _INPUT_STRUCTURE_FILENAME = "supercell.extxyz"
    _DISP_PREFIX = "disp"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData, help="the supercell")
        spec.input("temperature", valid_type=Float, default=lambda: Float(300.0))
        spec.input("timestep", valid_type=Float, default=lambda: Float(1.0), help="[fs]")
        spec.input("nsteps", valid_type=Int, default=lambda: Int(5000))
        spec.input("sample", valid_type=Str, default=lambda: Str("1001:5000:50"),
                   help="start:end:interval of the sampled steps (displace.py -e)")
        spec.input("random_mag", valid_type=Float, default=lambda: Float(0.04),
                   help="random displacement [A] added to every atom of every snapshot (displace.py --mag)")
        spec.input("random_seed", valid_type=Int, default=lambda: Int(1))
        spec.input("friction", valid_type=Float, default=lambda: Float(0.01), help="Langevin friction [1/fs]")
        spec.output('displaced_structures', valid_type=TrajectoryData)

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        _write_extxyz(folder, self._INPUT_STRUCTURE_FILENAME, self.inputs.structure.get_ase())
        job = self._job_base("md", [self._INPUT_STRUCTURE_FILENAME], "extxyz")
        job.update({"temperature": self.inputs.temperature.value, "timestep_fs": self.inputs.timestep.value,
                    "nsteps": self.inputs.nsteps.value, "sample": self.inputs.sample.value,
                    "random_mag": self.inputs.random_mag.value, "random_seed": self.inputs.random_seed.value,
                    "friction": self.inputs.friction.value, "disp_prefix": self._DISP_PREFIX})
        return self._calcinfo(folder, job, ["md.log", "md_traj.extxyz", f"{self._DISP_PREFIX}*.extxyz"])


class MattersimMdParser(MattersimBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        result.pop("structures", None)
        structures = []
        for filename in result.get("files", []):
            if filename not in output_folder.list_object_names():
                return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
            atoms = ase.io.read(io.StringIO(output_folder.get_object_content(filename)), format="extxyz")
            structures.append(StructureData(ase=atoms))
        if len(structures) == 0:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        self.out("results", Dict(dict=result))
        self.out("displaced_structures", TrajectoryData(structures))


class MattersimElasticCalculation(MattersimBaseCalculation):
    """elastic_constants.in (V*C in Ry, SOEC 81 + TOEC 729 values) and strain_force.in (Ry/Bohr) of a cell,
    the STRAIN_IFC_DIR inputs of the anphon QHA structural optimization."""
    _PARSER = 'alamode.mattersim_elastic'
    _INPUT_STRUCTURE_FILENAME = "cell.extxyz"
    _FILES = ["elastic_constants.in", "strain_force.in"]

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData, help="the (relaxed) primitive cell")
        spec.input("delta", valid_type=Float, default=lambda: Float(0.01), help="finite-difference strain step")
        spec.input("strain_force_delta", valid_type=Float, default=lambda: Float(0.005),
                   help="strain of strain_force.in (tutorial: 0.005)")
        spec.output('strain_ifc_folder', valid_type=FolderData, help="elastic_constants.in, strain_force.in")

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        _write_extxyz(folder, self._INPUT_STRUCTURE_FILENAME, self.inputs.structure.get_ase())
        job = self._job_base("elastic", [self._INPUT_STRUCTURE_FILENAME], "extxyz")
        job.update({"delta": self.inputs.delta.value, "strain_force_delta": self.inputs.strain_force_delta.value})
        return self._calcinfo(folder, job, list(self._FILES))


class MattersimElasticParser(MattersimBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        result.pop("structures", None)
        folderdata = FolderData()
        for filename in MattersimElasticCalculation._FILES:
            if filename not in output_folder.list_object_names():
                return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
            with output_folder.open(filename, "rb") as handle:
                folderdata.put_object_from_filelike(handle, filename)
        self.out("results", Dict(dict=result))
        self.out("strain_ifc_folder", folderdata)


def borninfo_lines(dielectric, bec) -> list:
    """BORNINFO of anphon: the dielectric tensor (3 lines), then Z* of every atom of the primitive cell
    (3 lines each, in the order of &position of the anphon input)."""
    dielectric = np.asarray(dielectric, dtype=float).reshape(3, 3)
    lines = ["%16.8f %16.8f %16.8f" % tuple(row) for row in dielectric]
    for z in np.asarray(bec, dtype=float).reshape(-1, 3, 3):
        lines += ["%14.6f %14.6f %14.6f" % tuple(row) for row in z]
    return lines


class MattersimBecCalculation(MattersimBaseCalculation):
    """Born effective charges of a (primitive) cell with a calculator that provides them
    (calculator {"name": "sevennet-polar", ...}), and the BORNINFO file of anphon.

    The dielectric tensor comes from the model when it provides one (results['dielectric_tensor']),
    otherwise from the input ``dielectric`` (3x3, or 1 value for an isotropic tensor). Without either,
    born_effective_charges is produced but borninfo is not.

    ``enforce_asr`` subtracts the mean of Z* so that sum_i Z*_i = 0 (acoustic sum rule).
    """
    _PARSER = 'alamode.mattersim_bec'
    _INPUT_STRUCTURE_FILENAME = "cell.extxyz"
    _BORNINFO = "BORNINFO"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData,
                   help="the primitive cell, in the atom order of the anphon &position")
        spec.input("dielectric", valid_type=List, required=False,
                   help="high-frequency dielectric tensor (3x3 nested list, or [e] isotropic) if the model has none")
        spec.input("enforce_asr", valid_type=Bool, default=lambda: Bool(True),
                   help="enforce the acoustic sum rule sum_i Z*_i = 0")
        spec.output('born_effective_charges', valid_type=ArrayData,
                    help="'bec' (nat, 3, 3) [e], 'dielectric' (3, 3) if known, 'bec_raw' before the ASR")
        spec.output('borninfo', valid_type=SinglefileData, required=False, help="BORNINFO file of anphon")
        spec.exit_code(343, 'ERROR_NO_BEC', message='the calculator did not return born effective charges.')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        _write_extxyz(folder, self._INPUT_STRUCTURE_FILENAME, self.inputs.structure.get_ase())
        job = self._job_base("bec", [self._INPUT_STRUCTURE_FILENAME], "extxyz")
        return self._calcinfo(folder, job, [])


def _dielectric_matrix(value) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.size == 1:
        return np.eye(3) * float(a.ravel()[0])
    if a.size == 3:
        return np.diag(a.ravel())
    return a.reshape(3, 3)


class MattersimBecParser(MattersimBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        if result.get("born_effective_charges") is None:
            return self.exit_codes.ERROR_NO_BEC
        bec_raw = np.asarray(result.pop("born_effective_charges"), dtype=float)
        bec = bec_raw - bec_raw.mean(axis=0) if self.node.inputs.enforce_asr.value else bec_raw
        dielectric = result.get("dielectric_tensor")
        source = "model"
        if dielectric is None and "dielectric" in self.node.inputs:
            dielectric = _dielectric_matrix(self.node.inputs.dielectric.get_list()).tolist()
            source = "input"
        result.pop("structures", None)
        result.update({"asr_residual_raw": bec_raw.sum(axis=0).tolist(), "asr_enforced": self.node.inputs.enforce_asr.value,
                       "dielectric_tensor": dielectric, "dielectric_source": source if dielectric is not None else None,
                       "bec_diagonal": [np.diag(z).tolist() for z in bec]})
        arrays = ArrayData()
        arrays.set_array("bec", bec)
        arrays.set_array("bec_raw", bec_raw)
        arrays.base.attributes.set("symbols", result.get("symbols", []))
        if dielectric is not None:
            arrays.set_array("dielectric", np.asarray(dielectric, dtype=float))
            lines = borninfo_lines(dielectric, bec)
            self.out("borninfo", SinglefileData(io.BytesIO(("\n".join(lines) + "\n").encode()),
                                                filename=MattersimBecCalculation._BORNINFO))
        self.out("results", Dict(dict=result))
        self.out("born_effective_charges", arrays)
