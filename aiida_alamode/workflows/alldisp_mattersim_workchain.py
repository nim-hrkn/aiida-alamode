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
"""forces of all the displaced structures with an ASE calculator (MatterSim), and the DFSET."""
import numpy as np

from aiida.orm import Str, Int, Bool, Dict, List, AbstractCode
from aiida.engine import WorkChain, calcfunction, append_
from aiida.plugins import DataFactory, CalculationFactory

from ..io.dfset import make_dfset_lines

StructureData = DataFactory('core.structure')
ArrayData = DataFactory('core.array')
TrajectoryData = DataFactory('core.array.trajectory')

_ARRAY_NAMES = ["energies", "forces", "stresses", "positions", "cells", "indices"]


@calcfunction
def merge_arrays(**kwargs) -> ArrayData:
    """merge the ArrayData of the chunks (chunk0, chunk1, ...) in the order of the structure indices."""
    chunks = [kwargs[key] for key in sorted(kwargs.keys(), key=lambda x: int(x.replace("chunk", "")))]
    indices = np.concatenate([chunk.get_array("indices") for chunk in chunks])
    order = np.argsort(indices, kind="stable")
    merged = ArrayData()
    for name in _ARRAY_NAMES:
        merged.set_array(name, np.concatenate([chunk.get_array(name) for chunk in chunks])[order])
    merged.base.attributes.set("symbols", chunks[0].base.attributes.get("symbols"))
    return merged


@calcfunction
def make_dfset(structure_org: StructureData, arrays: ArrayData, offset: ArrayData = None) -> List:
    """DFSET (List of lines, Rydberg atomic units as extract.py --QE) of the displaced structures.

    offset: arrays of the undisplaced structure_org; its forces are subtracted (extract.py --offset)."""
    labels = [f"structure{i}" for i in arrays.get_array("indices")]
    lines = make_dfset_lines(structure_org.get_ase(), arrays.get_array("positions"), arrays.get_array("forces"),
                             energies=arrays.get_array("energies").tolist(), labels=labels,
                             offset_forces=offset.get_array("forces")[0] if offset is not None else None)
    return List(list=lines)


@calcfunction
def structure_to_trajectory(structure: StructureData) -> TrajectoryData:
    return TrajectoryData([structure])


class ForcesMattersimWorkChain(WorkChain):
    """forces of the displaced structures with MattersimForcesCalculation, split into njobs jobs,
    and the DFSET for alm.

    With subtract_offset, the forces of the undisplaced structure_org are computed too and subtracted
    (needed when structure_org is not at equilibrium, e.g. a strained cell).
    """
    _RESOURCE = {'withmpi': False,
                 'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}}

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=AbstractCode, help="the 'alamode-mattersim' code")
        spec.input("structures", valid_type=TrajectoryData, help='displaced structures')
        spec.input("structure_org", valid_type=StructureData, help='the undisplaced supercell')
        spec.input("model", valid_type=Str, required=False)
        spec.input("device", valid_type=Str, required=False)
        spec.input("calculator", valid_type=Dict, required=False,
                   help="ASE calculator as in MattersimForcesCalculation")
        spec.input("njobs", valid_type=Int, default=lambda: Int(1), help='number of scheduler jobs')
        spec.input("subtract_offset", valid_type=Bool, default=lambda: Bool(False),
                   help="subtract the forces of the undisplaced structure_org")
        spec.input("cwd", valid_type=Str, required=False, help='directory where results are saved.')
        spec.input("prefix", valid_type=Str, required=False, help='string added to filenames')
        spec.input('options', valid_type=Dict, default=lambda: Dict(dict=cls._RESOURCE), help='metadata.options')
        spec.outline(cls.submit_calcjobs, cls.inspect_calcjobs)
        spec.output("arrays", valid_type=ArrayData, help="energies, forces, stresses, positions, cells")
        spec.output("offset_arrays", valid_type=ArrayData, required=False)
        spec.output("dfset", valid_type=List, help="DFSET lines")
        spec.exit_code(400, 'ERROR_CALCJOB_FAILED', message='a MattersimForcesCalculation failed.')

    def _builder(self, structures, indices=None):
        builder = CalculationFactory("alamode.mattersim").get_builder()
        builder.code = self.inputs.code
        builder.structures = structures
        if indices is not None:
            builder.indices = List(list=indices)
        for key in ("model", "device", "calculator", "cwd", "prefix"):
            if key in self.inputs:
                setattr(builder, key, self.inputs[key])
        builder.metadata.options = self.inputs.options.get_dict()
        return builder

    def submit_calcjobs(self):
        nstruct = self.inputs.structures.numsteps
        njobs = max(1, min(self.inputs.njobs.value, nstruct))
        for k in range(njobs):
            indices = list(range(k, nstruct, njobs))
            self.to_context(calcjobs=append_(self.submit(self._builder(self.inputs.structures, indices))))
        if self.inputs.subtract_offset.value:
            trajectory = structure_to_trajectory(self.inputs.structure_org)
            self.to_context(offset=self.submit(self._builder(trajectory)))

    def inspect_calcjobs(self):
        for calcjob in self.ctx.calcjobs:
            if not calcjob.is_finished_ok:
                return self.exit_codes.ERROR_CALCJOB_FAILED
        arrays = merge_arrays(**{f"chunk{i}": calcjob.outputs.arrays for i, calcjob in enumerate(self.ctx.calcjobs)})
        self.out("arrays", arrays)
        extra = {}
        if "offset" in self.ctx:
            if not self.ctx.offset.is_finished_ok:
                return self.exit_codes.ERROR_CALCJOB_FAILED
            extra["offset"] = self.ctx.offset.outputs.arrays
            self.out("offset_arrays", extra["offset"])
        self.out("dfset", make_dfset(self.inputs.structure_org, arrays, **extra))
