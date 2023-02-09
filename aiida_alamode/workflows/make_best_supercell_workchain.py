

from aiida.orm import Dict, Str
from aiida.engine import WorkChain, if_
from aiida.plugins import DataFactory
from aiida_alamode.workflows import make_best_supercell, structure_to_SinglefileData
import os

StructureData = DataFactory('structure')
ArrayData = DataFactory('array')
SinglefileData = DataFactory('singlefile')


def _make_best_supercell_(refined_structure, **kwargs):
    """make optimal supercell.

    Args:
        refined_structure (ase.atoms.Atom): periodic structure.

    Returns:
        tuples containing,
        ase.atoms.Atoms: peridic super structure.
        np.ndarray: 3x3 array defning supercell.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    refined_pystructure = AseAtomsAdaptor.get_structure(refined_structure)
    # _make_best_supercell argumet is pymatgen.core.Structure
    P = make_best_supercell(refined_pystructure, **kwargs)

    super_pystructure = refined_pystructure.copy()
    super_pystructure.make_supercell(P)

    # convert from pymatgen.core.Structure to ase.atoms.Atoms
    supercell_atoms = AseAtomsAdaptor.get_atoms(super_pystructure)

    return supercell_atoms, P


class BestSupercellWorkChain(WorkChain):
    """make the best supercell.

    The default input parameters is {'min_num_sites':20, 'max_num_sites': 400, 'max_anisotropy': 2.0 }
    """
    _PARAMETERS = {}
    _CWD = ""
    _SUPERCELL_FILENAME = "bestsupercell.{format}"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData, help='primitive structure')
        spec.input("params", valid_type=Dict, default=lambda: Dict(dict=cls._PARAMETERS),
                   help='additional parameters')
        spec.input("cwd", valid_type=Str, required=False,
                   help='directory where files are saved.')
        spec.input("format", valid_type=Str, required=False, help='format of the output file.')

        spec.outline(
            cls.make_supercell,
            if_(cls.must_place_file)(
                cls.place_supercell_structure
            )
        )

        spec.output("superstructure", valid_type=StructureData, help='supercell structure')
        spec.output("superstructure_file", valid_type=SinglefileData,
                    required=False, help='supercell structure file')
        spec.output("result", valid_type=Dict, help='supercell')

    def must_place_file(self):
        flag = "cwd" in self.inputs and "format" in self.inputs
        if flag:
            cwd = self.inputs.cwd.value
            format_ = self.inputs.format.value
            return len(cwd) > 0 and len(format_) > 0
        return False

    def make_supercell(self):
        prim_structure = self.inputs.structure.get_ase()
        param = self.inputs.params.get_dict()
        supercell_atoms, P = _make_best_supercell_(prim_structure, **param)
        # supercell_atoms, P = make_best_supercell(prim_structure)
        supercell_structure_node = StructureData(ase=supercell_atoms)
        supercell_structure_node.store()
        self.ctx.supercell_structure_node = supercell_structure_node
        self.out('superstructure', supercell_structure_node)
        result = {"P": P.tolist()}
        result_node = Dict(dict=result)
        result_node.store()
        self.out('result', result_node)

    def place_supercell_structure(self):
        import copy
        os.makedirs(self.inputs.cwd.value, exist_ok=True)
        filename = copy.deepcopy(self._SUPERCELL_FILENAME)
        filename = filename.replace("{format}", self.inputs.format.value)
        filepath = os.path.join(self.inputs.cwd.value, filename)
        filepath = Str(filepath)
        filepath.store()
        structure_file = structure_to_SinglefileData(atoms=self.ctx.supercell_structure_node,
                                                     path=filepath,
                                                     format=self.inputs.format)
        self.out("superstructure_file", structure_file)


