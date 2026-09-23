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
Diagonal supercells for ALAMODE.

ase.build.make_supercell may place the atoms of the first translation at a periodic image
different from the primitive position (e.g. Te of rocksalt PbTe at (0.875, 0.125, 0.125) of a
4x4x4 supercell instead of (0.125, 0.125, 0.125)). alm then labels that image "translation 1",
while anphon assumes translation 1 is exactly the primitive position. The harmonic phonons
are unaffected, but the Ewald long-range correction (NONANALYTIC = 3) becomes inconsistent.

make_diagonal_supercell puts every atom at x_prim + T with x_prim in [0, 1) and T = 0 first,
so that the atoms of translation 1 coincide with the primitive cell.
"""
from itertools import product

import numpy as np
from ase import Atoms


def make_diagonal_supercell(prim: Atoms, diag, eps: float = 1e-8) -> Atoms:
    """n1 x n2 x n3 supercell of prim. Atom-major order: atom i of translation t is at i*nT + t.

    Args:
        prim (Atoms): the primitive (or conventional) cell.
        diag (list of 3 int): the supercell multiplicities along the cell vectors.
        eps (float): fractional coordinates within eps of 1 are wrapped to 0.

    Returns:
        Atoms: the supercell with translation (0, 0, 0) first for each atom.
    """
    n = np.array(diag, dtype=int)
    if n.shape != (3,) or (n < 1).any():
        raise ValueError(f"diag must be 3 positive integers, not {diag}.")
    frac = prim.get_scaled_positions(wrap=True)
    frac = np.where(frac > 1.0 - eps, 0.0, frac)
    translations = np.array(list(product(range(n[0]), range(n[1]), range(n[2]))), dtype=float)

    numbers = []
    positions = []
    for z, x in zip(prim.get_atomic_numbers(), frac):
        for t in translations:
            positions.append((x + t) / n)
            numbers.append(z)
    supercell = Atoms(numbers=numbers, cell=prim.cell.array * n[:, None], pbc=True)
    supercell.set_scaled_positions(positions)
    return supercell
