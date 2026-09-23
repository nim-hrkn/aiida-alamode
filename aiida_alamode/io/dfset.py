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
"""DFSET of alm from displaced structures and their forces, without extract.py."""
import numpy as np
from ase import Atoms

# the same constants as alamode/interface/QE.py
BOHR_TO_ANGSTROM = 0.5291772108
RYDBERG_TO_EV = 13.60569253


def _refold(x: np.ndarray) -> np.ndarray:
    x = np.where(x >= 0.5, x - 1.0, x)
    return np.where(x < -0.5, x + 1.0, x)


def make_dfset_lines(atoms0: Atoms, positions: np.ndarray, forces: np.ndarray, energies=None,
                     labels=None, offset_forces: np.ndarray = None) -> list:
    """DFSET lines in Rydberg atomic units, as alamode tools/extract.py --QE writes them.

    Args:
        atoms0 (Atoms): the undisplaced supercell.
        positions (ndarray): (ndata, nat, 3) cartesian positions [A] of the displaced structures.
        forces (ndarray): (ndata, nat, 3) forces [eV/A].
        energies (list, optional): potential energies [eV] for the header lines.
        labels (list, optional): names for the header lines.
        offset_forces (ndarray, optional): (nat, 3) forces [eV/A] of the undisplaced supercell; subtracted
            (extract.py --offset), needed when the reference structure is not at equilibrium
            (e.g. a strained cell with fixed fractional coordinates).

    Returns:
        list: DFSET lines (str).
    """
    positions = np.asarray(positions, dtype=float)
    forces = np.asarray(forces, dtype=float)
    ndata = positions.shape[0]
    x0 = np.round(atoms0.get_scaled_positions(wrap=False), 8)
    lavec_transpose = atoms0.cell.array / BOHR_TO_ANGSTROM  # rows are lattice vectors
    inv_cell = np.linalg.inv(atoms0.cell.array)
    force_conversion = BOHR_TO_ANGSTROM / RYDBERG_TO_EV  # eV/A -> Ry/Bohr
    f0 = np.asarray(offset_forces, dtype=float) if offset_forces is not None else 0.0
    if energies is None:
        energies = [0.0] * ndata
    if labels is None:
        labels = [f"structure{i + 1}" for i in range(ndata)]

    lines = []
    for i in range(ndata):
        x = positions[i] @ inv_cell
        disp = _refold(x - x0) @ lavec_transpose
        f = (forces[i] - f0) * force_conversion
        lines.append("# Filename: %s, Snapshot: %d, E_pot (eV): %s" % (labels[i], 1, energies[i]))
        for j in range(len(atoms0)):
            lines.append("%15.7F %15.7F %15.7F %20.8E %15.8E %15.8E" % (disp[j, 0], disp[j, 1], disp[j, 2],
                                                                     f[j, 0], f[j, 1], f[j, 2]))
    return lines
