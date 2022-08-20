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

import os
import numpy as np
import json

from aiida.engine import WorkChain
from aiida.engine import calcfunction
from aiida.orm import Str,  Int
from aiida.plugins import DataFactory


from ..io.displacement import displacemenpattern_to_lines

from alm import ALM

# load types
StructureData = DataFactory('structure')
FolderData = DataFactory('folder')
SinglefileData = DataFactory('singlefile')
ArrayData = DataFactory('array')
List = DataFactory('list')
Dict = DataFactory('dict')


@calcfunction
def _alm_ALM_sugget(superstructure: StructureData, norder: Int,
                    cutoff_radi: List, nbody: List):
    lavec = superstructure.get_ase().cell
    xcoord = superstructure.get_ase().get_scaled_positions()
    kd = superstructure.get_ase().get_atomic_numbers()
    displacement_patterns = []
    with ALM(lavec, xcoord, kd) as alm:
        alm.define(norder.value, cutoff_radi.get_list(), nbody.get_list())
        alm.suggest()
        for fc_order in range(1, 1+norder.value):
            displacement_patterns.append(
                alm.get_displacement_patterns(fc_order))

    return List(list=displacement_patterns)


if False:
    @calcfunction
    def _alm_ALM_sugget_put_pattern_files(displacement_patterns, cwd, filename_list):
        folderdata = FolderData()
        for displacement_pattern, filename in zip(displacement_patterns, filename_list):
            filepath = os.path.abspath(os.path.join(cwd.value, filename))
            with open(filepath, "w") as f:
                json.dump(displacement_pattern, f)
            folderdata.put_object_from_file(filepath, filename)
        return folderdata


class alm_ALM_suggest(WorkChain):
    """ALM.suggest()
    """
    _CWD = ""

    @ classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str, default=lambda: Str(cls._CWD), help='directory where files are saved.')
        spec.input("structure", valid_type=StructureData, help='structure')
        spec.input("norder", valid_type=Int, help='1 (harmonic) or 2 (harmonic,cubic)')
        spec.input("cutoff_radii", valid_type=List, help='cutoff radii')
        spec.input("nbody", valid_type=List, help='nbody')

        spec.outline(cls.make_patterns)

        spec.output("results", valid_type=Dict)
        spec.output("pattern", valid_type=List)

    def make_patterns(self):
        patterns = _alm_ALM_sugget(self.inputs.structure, self.inputs.norder,
                                   self.inputs.cutoff_radii, self.inputs.nbody)
        cwd = self.inputs.cwd.value
        if len(cwd) > 0:
            os.makedirs(cwd, exist_ok=True)
            keys = ["HARMONIC", "ANHARM3"]
            prefix = "disp"
            for key, pattern in zip(keys, patterns.get_list()):
                filename = f'{prefix}.pattern_{key}'
                filepath = os.path.join(cwd, filename)
                content = displacemenpattern_to_lines(pattern)
                with open(filepath, "w") as f:
                    f.write("\n".join(content))

        self.out("pattern", patterns)
        results = {}  # dummy
        results = Dict(dict=results)
        results.store()
        self.out('results', results)


def _dfset_to_disp_and_force(lines: list):
    """transform DFSET to displacement and forces to pass ALM.optimize.

    The size of arrays are
        displacement(Nstructure, Nsiteperstructure, 3)
        force(Nstructure, Nsiteperstructure, 3)

    Args:
        lines (list): lines of DFSET splitted by "\n".

    Returns:
        tuple containig
        displacement (np.array): 3-dimensional array of displacement.
        force (np.array): 3-dimensional array of forces.
    """
    disp_list = []
    force_list = []
    structure_disp = []
    structure_force = []
    for line in lines[1:]:
        if line.startswith("#"):
            if len(structure_disp) > 0:
                disp_list.append(structure_disp)
                force_list.append(structure_force)
            structure_disp = []
            structure_force = []
            continue
        s = line.split()
        disp = list(map(float, s[:3]))
        force = list(map(float, s[3:]))
        structure_disp.append(disp)
        structure_force.append(force)
    if len(structure_disp) > 0:
        disp_list.append(structure_disp)
        force_list.append(structure_force)

    disp = np.array(disp_list)
    force = np.array(force_list)
    return disp, force


class alm_ALM_opt(WorkChain):
    """ALM.optimize()
    """
    _CWD = ""
    _PREFIX = "disp"
    _SOLVER = "dense"
    _OPTIMIZER_CONTROL = {}

    @ classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("cwd", valid_type=Str, help='directory where files are saved.')
        spec.input("structure", valid_type=StructureData, help='structure')
        spec.input("norder", valid_type=Int, help='1 (harmonic) or 2(cubic)')
        spec.input("cutoff_radii", valid_type=List, help='cutoff radii')
        spec.input("nbody", valid_type=List, help='nbody')
        spec.input("dfset", valid_type=List, help='DFSET')
        spec.input("solver", valid_type=Str,
                   default=lambda: Str(cls._SOLVER),
                   help='optimization solver')
        spec.input("optimizer_control", valid_type=Dict,
                   default=lambda: Dict(dict=cls._OPTIMIZER_CONTROL),
                   help='optimizer_control of ALM')
        spec.outline(cls.opt)

        spec.output("results", valid_type=Dict)
        spec.output("input_ANPHON", valid_type=SinglefileData)

    def opt(self):
        cwd = self.inputs.cwd.value
        norder = self.inputs.norder.value
        atoms = self.inputs.structure.get_ase()
        lavec = atoms.cell
        xcoord = atoms.get_scaled_positions()
        kd = atoms.get_atomic_numbers()
        prefix = self._PREFIX
        disp, force = _dfset_to_disp_and_force(self.inputs.dfset.get_list())
        with ALM(lavec, xcoord, kd) as alm:
            alm.define(norder)
            alm.displacements = disp
            alm.forces = force
            optimizer_control = self.inputs.optimizer_control.get_dict()
            if len(optimizer_control.keys()) > 0:
                alm.optimizer_control = optimizer_control
            solver = self.inputs.solver.value
            _ = alm.optimize(solver=solver)
            filepath = os.path.join(cwd, f'{prefix}.xml')
            alm.save_fc(filepath, format='alamode')

        file = SinglefileData(filepath)
        file.store()
        self.out("input_ANPHON", file)
        results = {}  # dummy
        results = Dict(dict=results)
        results.store()
        self.out('results', results)
