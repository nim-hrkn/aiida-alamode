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
Bases of the CalcJobs that call an external *engine* (a DFT code such as VASP or Quantum ESPRESSO, or a
machine-learning interatomic potential such as MatterSim or SevenNet through ASE).

- ExternalCalculatorBaseCalculation: engine-independent inputs (cwd, prefix), the ``results`` Dict and the
  shared exit codes.  The two kinds of engine jobs derive from it:
    * force_calcjob.py      ForceCalculatorBaseCalculation      energies / forces / stresses (relax, MD, elastic)
    * dielectric_calcjob.py DielectricCalculatorBaseCalculation Born effective charges and the dielectric tensor
  A DFT engine implements a subclass of the relevant base with the same output ports.
- AseRunnerBaseCalculation / AseRunnerBaseParser: the ASE-calculator engine, run by the console script
  ``alamode-ase-runner`` (aiida_alamode.ase_runner) inside the scheduler job with a json job file; the
  ``calculator`` input selects the potential (mattersim by default; mace, chgnet, sevennet, sevennet-polar,
  orb, emt, ...).
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

_JOB_FILENAME = "ase_job.json"
_RESULT_FILENAME = "ase_results.json"
_MODEL = "MatterSim-v1.0.0-1M.pth"


def _write_extxyz(folder: Folder, filename: str, atoms: ase.Atoms):
    """ase.io.write cannot write into a sandbox handle directly: go through a StringIO."""
    buffer = io.StringIO()
    ase.io.write(buffer, atoms, format="extxyz")
    with folder.open(filename, 'w', encoding='utf8') as handle:
        handle.write(buffer.getvalue())


class ExternalCalculatorBaseCalculation(AlamodeBaseCalculation):
    """base of every CalcJob run by an external engine (DFT code or ML potential).

    Engine-independent inputs (cwd, prefix), the ``results`` Dict output and the shared exit codes.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str, required=False, help='directory where results are saved.')
        spec.input("prefix", valid_type=Str, default=lambda: Str(""),
                   help='string added to the file names saved in cwd.')
        spec.output('results', valid_type=Dict, help="summary: engine, timing, ...")
        spec.exit_code(340, 'ERROR_OUTPUT_RESULT_MISSING', message='the result file of the engine was not retrieved.')
        spec.exit_code(341, 'ERROR_OUTPUT_INCOMPLETE', message='the number of results differs from the inputs.')
        spec.exit_code(342, 'ERROR_NOT_CONVERGED', message='the relaxation did not converge.')


class AseRunnerBaseCalculation(ExternalCalculatorBaseCalculation):
    """ASE-calculator engine (alamode-ase-runner): calculator selection and the job / result json files.

    Used as the engine base of both force-type and dielectric-type jobs (see force_calcjob.py, dielectric_calcjob.py).
    """
    _PARSER = 'alamode.forces_ase'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("model", valid_type=Str, default=lambda: Str(_MODEL), help='MatterSim model file')
        spec.input("device", valid_type=Str, default=lambda: Str("auto"), help="auto, cpu or cuda")
        spec.input("calculator", valid_type=Dict, required=False,
                   help='ASE calculator, e.g. {"name": "mace", "kwargs": {"model": "medium"}}; '
                        'see ase_runner.CALCULATORS. Overrides model / device.')
        spec.inputs['metadata']['options']['parser_name'].default = cls._PARSER
        spec.inputs['metadata']['options']['input_filename'].default = _JOB_FILENAME
        spec.inputs['metadata']['options']['output_filename'].default = 'ase_runner.out'
        spec.inputs['metadata']['options']['withmpi'].default = False
        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1, 'num_mpiprocs_per_machine': 1}

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



class AseRunnerBaseParser(Parser):

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
