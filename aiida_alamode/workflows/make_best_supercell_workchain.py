

from aiida.orm import Dict
from aiida.engine import WorkChain
from aiida.plugins import DataFactory
from alamode_aiida.workflows import make_best_supercell


StructureData = DataFactory('structure')
ArrayData = DataFactory('array')


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

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData, help='primitive structure')
        spec.input("parameters", valid_type=Dict, default=lambda: Dict(dict=cls._PARAMETERS),
                   help='additional parameters')

        spec.outline(
            cls.make_supercell,
        )

        spec.output("structure", valid_type=StructureData, help='supercell structure')
        spec.output("result", valid_type=Dict, help='supercell')

    def make_supercell(self):
        prim_structure = self.inputs.structure.get_ase()
        param = self.inputs.parameters.get_dict()
        print("param", param)
        supercell_atoms, P = _make_best_supercell_(prim_structure, **param)
        # supercell_atoms, P = make_best_supercell(prim_structure)
        supercell_structure_node = StructureData(ase=supercell_atoms)
        supercell_structure_node.store()
        self.out('structure', supercell_structure_node)
        result = {"P": P.tolist()}
        result_node = Dict(dict=result)
        result_node.store()
        self.out('result', result_node)
