
from ase.calculators.lammps import Prism, convert

def write_lammps_data(fd, atoms, specorder=None, force_skew=False,
                      prismobj=None, velocities=False, units="metal",
                      atom_style='atomic'):
    """Write atomic structure data to a LAMMPS data file.
    
    The original code is from ASE.
    
    オリジナルコードは
    - Massesを書かないので加えた。
    - xy xz yz を書かないので加えた。
    """

    # FIXME: We should add a check here that the encoding of the file object
    #        is actually ascii once the 'encoding' attribute of IOFormat objects
    #        starts functioning in implementation (currently it doesn't do
    #         anything).

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    if hasattr(fd, "name"):
        fd.write("# {0} (written by ASE) \n\n".format(fd.name))
    else:
        fd.write("# (written by ASE) \n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write("{0} \t atoms \n".format(n_atoms))

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
        
   
    n_atom_types = len(species)
    fd.write("{0}  atom types\n".format(n_atom_types))

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    fd.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    fd.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    fd.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if force_skew or p.is_skewed():
        fd.write(
            "{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
    fd.write("\n\n")
    

    masses = atoms.get_masses()
    fd.write("Masses\n\n")
    for _specie_id, _specie in enumerate(species):
        _i = symbols.index(_specie)
        mass = masses[_i]
        fd.write(f"  {_specie_id+1} {mass}\n")
    fd.write("\n\n")
    
    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write("Atoms \n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)

    if atom_style == 'atomic':
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n"
                     .format(*(i + 1, s, q) + tuple(r)))
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        # The label 'mol-id' has apparenlty been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has('mol-id'):
            molecules = atoms.get_array('mol-id')
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " mol-id dtype must be subtype of np.integer, and"
                    " not {:s}.").format(str(molecules.dtype)))
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " each atom must have exactly one mol-id."))
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferableabove assigning all atoms to a single molecule
            # id per default, as done within ase <= v 3.19.1. I.e.,
            # molecules = np.arange(start=1, stop=len(atoms)+1,
            # step=1, dtype=int) However, according to LAMMPS default
            # behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of molecule
            #    assignments.

        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} "
                     "{6:23.17g}\n".format(*(i + 1, m, s, q) + tuple(r)))
    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities \n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            fd.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    fd.flush()
    
    
