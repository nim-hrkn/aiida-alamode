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
Force-calculator jobs: energies, forces and stresses of structures (and what is derived from them).

The engine can be a DFT code (VASP, Quantum ESPRESSO, ...) or a machine-learning interatomic potential
(MatterSim, SevenNet, MACE, ... through ASE).  Every implementation derives from
``ForceCalculatorBaseCalculation`` and keeps the output ports below, so the ALAMODE workflow does not care
which engine produced the forces:

    forces  : ``arrays`` (ArrayData: energies [eV], forces [eV/A], stresses [eV/A^3], positions [A], cells [A],
              indices) of the structures of a TrajectoryData            -> ForcesWorkChain, DFSET
    relax   : relaxed ``structure`` (StructureData)
    md      : ``displaced_structures`` (TrajectoryData) sampled from an MD run
    elastic : ``strain_ifc_folder`` (elastic_constants.in, strain_force.in for the anphon QHA)

ASE implementations (engine: alamode-ase-runner, see engine_base.py):
- AseForcesCalculation   entry point alamode.forces_ase
- AseRelaxCalculation    alamode.relax_ase
- AseMdCalculation       alamode.md_ase
- AseElasticCalculation  alamode.elastic_ase
A DFT implementation would be e.g. VaspForcesCalculation (alamode.forces_vasp) with the same ``arrays`` output.

The Born effective charges and the dielectric tensor are a different kind of prediction (dielectric_calcjob.py).
"""
import io
import json

import numpy as np
import ase
import ase.io

from aiida.orm import Str, Int, Float, Bool, Dict, List
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.plugins import DataFactory

from .engine_base import (ExternalCalculatorBaseCalculation, AseRunnerBaseCalculation, AseRunnerBaseParser,
                          _write_extxyz)

FolderData = DataFactory('core.folder')
SinglefileData = DataFactory('core.singlefile')
ArrayData = DataFactory('core.array')
StructureData = DataFactory('core.structure')
TrajectoryData = DataFactory('core.array.trajectory')

_ARRAY_NAMES = ["energies", "forces", "stresses", "positions", "cells"]


class ForceCalculatorBaseCalculation(ExternalCalculatorBaseCalculation):
    """base of the force-type jobs (forces, relax, md, elastic) of any engine; see the module docstring
    for the output-port contract."""


class AseForceCalculatorBaseCalculation(AseRunnerBaseCalculation, ForceCalculatorBaseCalculation):
    """force-type jobs run by the ASE-calculator engine."""


class AseForcesCalculation(AseForceCalculatorBaseCalculation):
    """energy, forces and stress of every structure of a TrajectoryData (or of a subset, indices).

    outputs: arrays (ArrayData: energies [eV], forces [eV/A], stresses [eV/A^3], positions [A], cells [A],
    attribute 'indices'), results (Dict).
    """
    _PARSER = 'alamode.forces_ase'
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
        return self._calcinfo(folder, job, ["*.calc.extxyz"])


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


class AseForcesParser(AseRunnerBaseParser):

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


class AseRelaxCalculation(AseForceCalculatorBaseCalculation):
    """relax the cell and the atomic positions of a structure.

    By default only the cell volume is relaxed (hydrostatic strain), as in the MatterSim
    replacement of the DFT lattice-constant optimization.
    """
    _PARSER = 'alamode.relax_ase'
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
        return self._calcinfo(folder, job, ["relax.log", "*.calc.extxyz"])


class AseRelaxParser(AseRunnerBaseParser):

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


class AseMdCalculation(AseForceCalculatorBaseCalculation):
    """NVT (Langevin) MD of the supercell; sampled snapshots with random displacements as displaced structures
    (displace.py -md ... -e start:end:interval --random --mag).

    outputs: displaced_structures (TrajectoryData, for ForcesWorkChain), results (Dict).
    """
    _PARSER = 'alamode.md_ase'
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


class AseMdParser(AseRunnerBaseParser):

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


class AseElasticCalculation(AseForceCalculatorBaseCalculation):
    """elastic_constants.in (V*C in Ry, SOEC 81 + TOEC 729 values) and strain_force.in (Ry/Bohr) of a cell,
    the STRAIN_IFC_DIR inputs of the anphon QHA structural optimization."""
    _PARSER = 'alamode.elastic_ase'
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


class AseElasticParser(AseRunnerBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        result.pop("structures", None)
        folderdata = FolderData()
        for filename in AseElasticCalculation._FILES:
            if filename not in output_folder.list_object_names():
                return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
            with output_folder.open(filename, "rb") as handle:
                folderdata.put_object_from_filelike(handle, filename)
        self.out("results", Dict(dict=result))
        self.out("strain_ifc_folder", folderdata)
