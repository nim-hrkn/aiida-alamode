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
Dielectric-property jobs: Born effective charges Z* (and the high-frequency dielectric tensor eps_inf)
of a cell, i.e. what anphon needs in BORNINFO for the LO-TO correction (NONANALYTIC > 0).

This is a different prediction from the forces (force_calcjob.py): a force calculator does not give Z*,
and a Z* model such as SevenNet-Polar is not a force field.  Engines:
- ASE calculators that provide ``born_effective_charges`` (SevenNet-Polar): AseBornChargesCalculation
  (entry point alamode.bec_ase, alias alamode.mattersim_bec).  eps_inf comes from the model when it
  provides ``dielectric_tensor``, otherwise from the ``dielectric`` input.
- DFT (VASP LEPSILON / LCALCEPS, Quantum ESPRESSO ph.x): a subclass of DielectricCalculatorBaseCalculation
  that parses Z* and eps_inf from the code output (e.g. VaspBornChargesCalculation, alamode.bec_vasp).

Output-port contract of DielectricCalculatorBaseCalculation:
    born_effective_charges : ArrayData 'bec' (nat, 3, 3) [e] in the atom order of ``structure``, 'bec_raw'
                             before the acoustic sum rule, 'dielectric' (3, 3) when known
    borninfo               : SinglefileData BORNINFO for anphon (eps_inf 3 lines, then 3 lines per atom);
                             only when eps_inf is known
    results                : Dict (diagonal Z*, ASR residual, source of eps_inf, ...)
"""
import io

import numpy as np

from aiida.orm import Str, Bool, Dict, List
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.plugins import DataFactory

from .engine_base import (ExternalCalculatorBaseCalculation, AseRunnerBaseCalculation, AseRunnerBaseParser,
                          _write_extxyz)

SinglefileData = DataFactory('core.singlefile')
ArrayData = DataFactory('core.array')
StructureData = DataFactory('core.structure')


class DielectricCalculatorBaseCalculation(ExternalCalculatorBaseCalculation):
    """base of the Born-charge / dielectric-tensor jobs of any engine (inputs and outputs shared by all)."""
    _BORNINFO = "BORNINFO"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData,
                   help="the primitive cell, in the atom order of the anphon &position")
        spec.input("dielectric", valid_type=List, required=False,
                   help="high-frequency dielectric tensor if the engine has none: 3x3 nested list, "
                        "[e_xx, e_yy, e_zz] or [e] (isotropic)")
        spec.input("enforce_asr", valid_type=Bool, default=lambda: Bool(True),
                   help="enforce the acoustic sum rule sum_i Z*_i = 0 (the mean of Z* is subtracted)")
        spec.output('born_effective_charges', valid_type=ArrayData,
                    help="'bec' (nat, 3, 3) [e], 'dielectric' (3, 3) if known, 'bec_raw' before the ASR")
        spec.output('borninfo', valid_type=SinglefileData, required=False, help="BORNINFO file of anphon")
        spec.exit_code(343, 'ERROR_NO_BEC', message='the engine did not return born effective charges.')


def borninfo_lines(dielectric, bec) -> list:
    """BORNINFO of anphon: the dielectric tensor (3 lines), then Z* of every atom of the primitive cell
    (3 lines each, in the order of &position of the anphon input)."""
    dielectric = np.asarray(dielectric, dtype=float).reshape(3, 3)
    lines = ["%16.8f %16.8f %16.8f" % tuple(row) for row in dielectric]
    for z in np.asarray(bec, dtype=float).reshape(-1, 3, 3):
        lines += ["%14.6f %14.6f %14.6f" % tuple(row) for row in z]
    return lines


class AseBornChargesCalculation(AseRunnerBaseCalculation, DielectricCalculatorBaseCalculation):
    """Born effective charges of a (primitive) cell with a calculator that provides them
    (calculator {"name": "sevennet-polar", ...}), and the BORNINFO file of anphon.

    The dielectric tensor comes from the model when it provides one (results['dielectric_tensor']),
    otherwise from the input ``dielectric`` (3x3, or 1 value for an isotropic tensor). Without either,
    born_effective_charges is produced but borninfo is not.

    ``enforce_asr`` subtracts the mean of Z* so that sum_i Z*_i = 0 (acoustic sum rule).
    """
    _PARSER = 'alamode.bec_ase'
    _INPUT_STRUCTURE_FILENAME = "cell.extxyz"

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


class AseBornChargesParser(AseRunnerBaseParser):

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
                                                filename=AseBornChargesCalculation._BORNINFO))
        self.out("results", Dict(dict=result))
        self.out("born_effective_charges", arrays)



# backward-compatible names
MattersimBecCalculation = AseBornChargesCalculation
MattersimBecParser = AseBornChargesParser
