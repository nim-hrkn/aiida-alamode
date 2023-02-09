from .supercell_tools import _make_best_supercell, symmetrize_atoms, read_qeoutput_from_genid


def make_best_supercell(refined_structure):
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
    P = _make_best_supercell(refined_pystructure)

    super_pystructure = refined_pystructure.copy()
    super_pystructure.make_supercell(P)

    # convert from pymatgen.core.Structure to ase.atoms.Atoms
    supercell_atoms = AseAtomsAdaptor.get_atoms(super_pystructure)

    return supercell_atoms, P


if __name__ == "__main__":

    def load_qe_as_atoms(gen, id_,
                         base_data_dir="/home/max/Documents/tadano_sample/from_Ishikawa/low_convexhulldata"):
        """load qe as ase.atoms.Atoms

        Args:
            gen (int): generation number.
            id_ (int): id number.
            base_data_dir (str, optional): base data directory name. Defaults to "/home/max/Documents/tadano_sample/from_Ishikawa/low_convexhulldata".

        Returns:
            ase.atoms.Atoms: periodic structure.
        """
        structure = read_qeoutput_from_genid(gen, id_, parent_dir=base_data_dir)
        refined_structure, spg_symbol = symmetrize_atoms(structure, symprec=1.0e-3)

        v = spg_symbol.split()
        spg_number = int(v[1].replace("(", "").replace(")", ""))
        return refined_structure, spg_number

    for id_ in range(10):
        print("\n\nid_", id_)
        refined_structure, spg_symbol = load_qe_as_atoms(0, id_)
        print(refined_structure, spg_symbol)
        supercell_atoms, P = make_best_supercell(refined_structure)
        print(P)
        print(supercell_atoms)
