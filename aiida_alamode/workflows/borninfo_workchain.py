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
"""BORNINFO of anphon from a Born-charge job and a dielectric-tensor job of any engines."""
import io

import numpy as np

from aiida.orm import Str, Dict, List
from aiida.engine import WorkChain, calcfunction, if_
from aiida.plugins import DataFactory, CalculationFactory

StructureData = DataFactory('core.structure')
ArrayData = DataFactory('core.array')
SinglefileData = DataFactory('core.singlefile')


def borninfo_lines(dielectric, bec) -> list:
    """BORNINFO of anphon: the dielectric tensor (3 lines), then Z* of every atom of the primitive cell
    (3 lines each, in the order of &position of the anphon input)."""
    dielectric = np.asarray(dielectric, dtype=float).reshape(3, 3)
    lines = ["%16.8f %16.8f %16.8f" % tuple(row) for row in dielectric]
    for z in np.asarray(bec, dtype=float).reshape(-1, 3, 3):
        lines += ["%14.6f %14.6f %14.6f" % tuple(row) for row in z]
    return lines


def dielectric_matrix(value) -> np.ndarray:
    """3x3 from 1 (isotropic), 3 (diagonal) or 9 values"""
    a = np.asarray(value, dtype=float)
    if a.size == 1:
        return np.eye(3) * float(a.ravel()[0])
    if a.size == 3:
        return np.diag(a.ravel())
    return a.reshape(3, 3)


@calcfunction
def dielectric_from_values(values: List) -> ArrayData:
    arrays = ArrayData()
    arrays.set_array("epsilon_inf", dielectric_matrix(values.get_list()))
    arrays.base.attributes.set("source", "input")
    return arrays


@calcfunction
def make_borninfo(structure: StructureData, born_effective_charges: ArrayData, dielectric_tensor: ArrayData) -> dict:
    """BORNINFO (SinglefileData) and a summary; the Z* rows follow the atom order of structure."""
    bec = born_effective_charges.get_array("bec")
    eps = dielectric_tensor.get_array("epsilon_inf")
    if bec.shape[0] != len(structure.sites):
        raise ValueError(f"{bec.shape[0]} Born charges for {len(structure.sites)} atoms")
    lines = borninfo_lines(eps, bec)
    borninfo = SinglefileData(io.BytesIO(("\n".join(lines) + "\n").encode()), filename="BORNINFO")
    summary = {"symbols": [s.kind_name for s in structure.sites],
               "bec_diagonal": [np.diag(z).tolist() for z in bec],
               "asr_residual": bec.sum(axis=0).tolist(),
               "epsilon_inf": eps.tolist(),
               "epsilon_inf_source": dielectric_tensor.base.attributes.get("source", None)}
    return {"borninfo": borninfo, "results": Dict(dict=summary)}


class BornInfoWorkChain(WorkChain):
    """Z* (a BornChargesBaseCalculation of any engine) + eps_inf (a DielectricTensorBaseCalculation of any
    engine, or a given value) -> BORNINFO for anphon.

    The engines are chosen by entry point: ``bec_plugin`` (default alamode.bec_ase with a SevenNet-Polar
    calculator) and ``epsinf_plugin`` (default alamode.epsinf_ase with AnisoNet).  The inputs of each job
    are given in the dynamic namespaces ``bec`` and ``epsinf`` (code, calculator, dielectric_model,
    options Dict for metadata.options, ...).  ``dielectric`` (1, 3 or 9 values) replaces the eps_inf job;
    if the Z* job itself returns a dielectric tensor, that one is used and the eps_inf job is skipped.
    """
    _RESOURCE = {'withmpi': False, 'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData, help="the primitive cell given to anphon")
        spec.input("bec_plugin", valid_type=Str, default=lambda: Str("alamode.bec_ase"),
                   help="entry point of the Born-charge CalcJob")
        spec.input_namespace("bec", dynamic=True, help="inputs of the Born-charge CalcJob (code, calculator, options, ...)")
        spec.input("epsinf_plugin", valid_type=Str, default=lambda: Str("alamode.epsinf_ase"),
                   help="entry point of the dielectric-tensor CalcJob")
        spec.input_namespace("epsinf", dynamic=True, required=False,
                             help="inputs of the dielectric-tensor CalcJob (code, dielectric_model, options, ...)")
        spec.input("dielectric", valid_type=List, required=False,
                   help="eps_inf given by hand (1, 3 or 9 values) instead of the eps_inf job")
        spec.outline(
            cls.run_bec,
            cls.inspect_bec,
            if_(cls.needs_epsinf_job)(cls.run_epsinf, cls.inspect_epsinf),
            cls.assemble,
        )
        spec.output("born_effective_charges", valid_type=ArrayData)
        spec.output("dielectric_tensor", valid_type=ArrayData)
        spec.output("borninfo", valid_type=SinglefileData)
        spec.output("results", valid_type=Dict)
        spec.exit_code(410, 'ERROR_BEC_FAILED', message='the Born-charge job failed.')
        spec.exit_code(411, 'ERROR_EPSINF_FAILED', message='the dielectric-tensor job failed.')
        spec.exit_code(412, 'ERROR_NO_EPSINF', message='no dielectric tensor: give the epsinf inputs or dielectric.')

    def _submit_job(self, plugin: str, namespace: dict):
        builder = CalculationFactory(plugin).get_builder()
        inputs = dict(namespace)
        options = inputs.pop("options", None)
        builder.structure = self.inputs.structure
        for key, value in inputs.items():
            setattr(builder, key, value)
        builder.metadata.options = options.get_dict() if options is not None else dict(self._RESOURCE)
        return self.submit(builder)

    def run_bec(self):
        return {"bec": self._submit_job(self.inputs.bec_plugin.value, self.inputs.bec)}

    def inspect_bec(self):
        if not self.ctx.bec.is_finished_ok:
            return self.exit_codes.ERROR_BEC_FAILED
        self.out("born_effective_charges", self.ctx.bec.outputs.born_effective_charges)
        if "dielectric_tensor" in self.ctx.bec.outputs:
            self.ctx.dielectric = self.ctx.bec.outputs.dielectric_tensor
        elif "dielectric" in self.inputs:
            self.ctx.dielectric = dielectric_from_values(self.inputs.dielectric)

    def needs_epsinf_job(self):
        return "dielectric" not in self.ctx

    def run_epsinf(self):
        if "epsinf" not in self.inputs or "code" not in self.inputs.epsinf:
            return self.exit_codes.ERROR_NO_EPSINF
        return {"epsinf": self._submit_job(self.inputs.epsinf_plugin.value, self.inputs.epsinf)}

    def inspect_epsinf(self):
        if not self.ctx.epsinf.is_finished_ok:
            return self.exit_codes.ERROR_EPSINF_FAILED
        self.ctx.dielectric = self.ctx.epsinf.outputs.dielectric_tensor

    def assemble(self):
        self.out("dielectric_tensor", self.ctx.dielectric)
        out = make_borninfo(self.inputs.structure, self.ctx.bec.outputs.born_effective_charges, self.ctx.dielectric)
        self.out("borninfo", out["borninfo"])
        self.out("results", out["results"])
