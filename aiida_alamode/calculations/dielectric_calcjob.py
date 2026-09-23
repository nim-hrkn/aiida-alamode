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
Dielectric-property jobs: the Born effective charges Z* and the high-frequency dielectric tensor eps_inf
of a cell, i.e. what anphon needs in BORNINFO for the LO-TO correction (NONANALYTIC > 0).

Both are predictions of a different kind than the forces (force_calcjob.py), and they are two separate
jobs, because the engines differ: a Z* model (SevenNet-Polar) does not give eps_inf, an eps_inf model
(AnisoNet) does not give Z*, and a DFT code (VASP LEPSILON, QE ph.x) gives both.  BornInfoWorkChain
(workflows/borninfo_workchain.py) runs the two jobs of any engine combination and assembles the BORNINFO.

Output-port contracts (a DFT implementation subclasses the same bases):
- BornChargesBaseCalculation      -> ``born_effective_charges``: ArrayData 'bec' (nat, 3, 3) [e] in the atom
                                     order of ``structure`` (ASR enforced if ``enforce_asr``), 'bec_raw';
                                     optional ``dielectric_tensor`` when the engine also gives eps_inf
- DielectricTensorBaseCalculation -> ``dielectric_tensor``: ArrayData 'epsilon_inf' (3, 3)

ASE implementations (engine: alamode-ase-runner, see engine_base.py):
- AseBornChargesCalculation      entry point alamode.bec_ase     (calculator {"name": "sevennet-polar"})
- AseDielectricTensorCalculation entry point alamode.epsinf_ase  (dielectric_model {"name": "anisonet"},
                                 or a calculator that returns dielectric_tensor)
"""
import numpy as np

from aiida.orm import Bool, Dict
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.plugins import DataFactory

from .engine_base import (ExternalCalculatorBaseCalculation, AseRunnerBaseCalculation, AseRunnerBaseParser,
                          _write_extxyz)

ArrayData = DataFactory('core.array')
StructureData = DataFactory('core.structure')


class DielectricCalculatorBaseCalculation(ExternalCalculatorBaseCalculation):
    """base of the dielectric-property jobs of any engine: the cell whose properties are computed."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData,
                   help="the primitive cell, in the atom order of the anphon &position")


class BornChargesBaseCalculation(DielectricCalculatorBaseCalculation):
    """Born effective charges Z* of every atom (any engine)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("enforce_asr", valid_type=Bool, default=lambda: Bool(True),
                   help="enforce the acoustic sum rule sum_i Z*_i = 0 (the mean of Z* is subtracted)")
        spec.output('born_effective_charges', valid_type=ArrayData,
                    help="'bec' (nat, 3, 3) [e] in the atom order of structure, 'bec_raw' before the ASR")
        spec.output('dielectric_tensor', valid_type=ArrayData, required=False,
                    help="'epsilon_inf' (3, 3) when the engine gives it too")
        spec.exit_code(343, 'ERROR_NO_BEC', message='the engine did not return born effective charges.')


class DielectricTensorBaseCalculation(DielectricCalculatorBaseCalculation):
    """high-frequency (electronic) dielectric tensor eps_inf of the cell (any engine)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.output('dielectric_tensor', valid_type=ArrayData, help="'epsilon_inf' (3, 3)")
        spec.exit_code(344, 'ERROR_NO_DIELECTRIC', message='the engine did not return the dielectric tensor.')


def dielectric_arraydata(eps, source: str = None) -> ArrayData:
    arrays = ArrayData()
    arrays.set_array("epsilon_inf", np.asarray(eps, dtype=float).reshape(3, 3))
    if source:
        arrays.base.attributes.set("source", source)
    return arrays


class AseBornChargesCalculation(AseRunnerBaseCalculation, BornChargesBaseCalculation):
    """Z* with an ASE calculator that provides born_effective_charges (calculator {"name": "sevennet-polar"});
    runner mode "bec".  If the calculator also returns dielectric_tensor it is output as well."""
    _PARSER = 'alamode.bec_ase'
    _INPUT_STRUCTURE_FILENAME = "cell.extxyz"

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        _write_extxyz(folder, self._INPUT_STRUCTURE_FILENAME, self.inputs.structure.get_ase())
        job = self._job_base("bec", [self._INPUT_STRUCTURE_FILENAME], "extxyz")
        return self._calcinfo(folder, job, [])


class AseBornChargesParser(AseRunnerBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        if result.get("born_effective_charges") is None:
            return self.exit_codes.ERROR_NO_BEC
        bec_raw = np.asarray(result.pop("born_effective_charges"), dtype=float)
        bec = bec_raw - bec_raw.mean(axis=0) if self.node.inputs.enforce_asr.value else bec_raw
        result.pop("structures", None)
        result.update({"asr_residual_raw": bec_raw.sum(axis=0).tolist(),
                       "asr_enforced": self.node.inputs.enforce_asr.value,
                       "bec_diagonal": [np.diag(z).tolist() for z in bec]})
        arrays = ArrayData()
        arrays.set_array("bec", bec)
        arrays.set_array("bec_raw", bec_raw)
        arrays.base.attributes.set("symbols", result.get("symbols", []))
        self.out("born_effective_charges", arrays)
        if result.get("dielectric_tensor") is not None:
            self.out("dielectric_tensor", dielectric_arraydata(result["dielectric_tensor"], "calculator"))
        self.out("results", Dict(dict=result))


class AseDielectricTensorCalculation(AseRunnerBaseCalculation, DielectricTensorBaseCalculation):
    """eps_inf with a dielectric model (dielectric_model {"name": "anisonet"}, ase_runner.DIELECTRIC_MODELS)
    or with an ASE calculator that returns dielectric_tensor; runner mode "dielectric"."""
    _PARSER = 'alamode.epsinf_ase'
    _INPUT_STRUCTURE_FILENAME = "cell.extxyz"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("dielectric_model", valid_type=Dict, required=False,
                   help='eps_inf model, e.g. {"name": "anisonet", "kwargs": {...}}; without it the calculator '
                        'input must return dielectric_tensor')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        _write_extxyz(folder, self._INPUT_STRUCTURE_FILENAME, self.inputs.structure.get_ase())
        job = self._job_base("dielectric", [self._INPUT_STRUCTURE_FILENAME], "extxyz")
        if "dielectric_model" in self.inputs:
            job["dielectric_model"] = self.inputs.dielectric_model.get_dict()
        return self._calcinfo(folder, job, [])


class AseDielectricTensorParser(AseRunnerBaseParser):

    def parse(self, **kwargs):
        output_folder, result = self._load()
        if output_folder is None:
            return result
        if result.get("dielectric_tensor") is None:
            return self.exit_codes.ERROR_NO_DIELECTRIC
        result.pop("structures", None)
        self.out("dielectric_tensor", dielectric_arraydata(result["dielectric_tensor"], result.get("dielectric_source")))
        self.out("results", Dict(dict=result))
