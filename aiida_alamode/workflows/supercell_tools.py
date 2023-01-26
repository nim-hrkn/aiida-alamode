#
# Copyright (c) 2022 Terumasa Tadano. This software is released under the MIT license.
#
from ase.io import read
import spglib
import glob
import os
from ase import Atoms
import itertools
import numpy as np


def make_best_supercell(primitive_structure, 
                         min_num_sites=20,
                         max_num_sites=400,
                         max_anisotropy=2.0):
    """make best supercell from primitive cell.

    Args:
        primitive_structure (pymatgen.Structure): periodic structure.
        min_num_sites (int, optional): minimum number of sites. Defaults to 20.
        max_num_sites (int, optional): maximum number of sites. Defaults to 400.
        max_anisotropy (float, optional): maximum anisotropy of the cell. Defaults to 2.0.

    Returns:
        np.ndarray: 3x3 vector defining the supercell.
    """

    a, b, c = primitive_structure.lattice.abc
    num_sites_primitive = primitive_structure.num_sites

    transformation_matrices = []
    anisotropies = []

    if (abs(a / b - 1.0) < 1.0e-4) and (abs(a / c - 1.0) < 1.0e-4):

        min_scale_along_each_axis = int((min_num_sites / num_sites_primitive)**(1.0/3.0))
        max_scale_along_each_axis = int((max_num_sites / num_sites_primitive)**(1.0/3.0))

        min_scale_along_each_axis = 1

        ratios = range(min_scale_along_each_axis, max_scale_along_each_axis+1)

        for ratio in ratios:
            matrix_tmp = np.array([[ratio, 0, 0], [0, ratio, 0], [0, 0, ratio]])

            if np.linalg.det(matrix_tmp) * num_sites_primitive >= min_num_sites \
                    and np.linalg.det(matrix_tmp) * num_sites_primitive <= max_num_sites:
                transformation_matrices.append(matrix_tmp)
                anisotropies.append(1.0)

    elif abs(a / b - 1.0) < 1.0e-4:

        min_scale_along_ab_axes = int((min_num_sites / num_sites_primitive)**0.5)
        max_scale_along_ab_axes = int((max_num_sites / num_sites_primitive)**0.5)
        min_scale_along_ab_axes = 1

        min_scale_along_c_axis = min_num_sites // num_sites_primitive
        max_scale_along_c_axis = max_num_sites // num_sites_primitive
        min_scale_along_c_axis = 1

        ratios_ab = range(min_scale_along_ab_axes, max_scale_along_ab_axes+1)
        ratios_c = range(min_scale_along_c_axis, max_scale_along_c_axis+1)

        for r_ab, r_c in itertools.product(ratios_ab, ratios_c):
            matrix_tmp = np.array([[r_ab, 0, 0], [0, r_ab, 0], [0, 0, r_c]])

            if np.linalg.det(matrix_tmp) * num_sites_primitive >= min_num_sites \
                    and np.linalg.det(matrix_tmp) * num_sites_primitive <= max_num_sites:

                abc_super = [a*r_ab, b*r_ab, c*r_c]
                abc_sorted = sorted(abc_super)
                # print(abc_super)
                if abc_sorted[2] / abc_sorted[0] <= max_anisotropy:
                    transformation_matrices.append(matrix_tmp)
                    anisotropies.append(abc_sorted[2] / abc_sorted[0])

    elif abs(a / c - 1.0) < 1.0e-4:

        min_scale_along_ac_axes = int((min_num_sites / num_sites_primitive)**0.5)
        max_scale_along_ac_axes = int((max_num_sites / num_sites_primitive)**0.5)
        min_scale_along_ac_axes = 1

        min_scale_along_b_axis = min_num_sites // num_sites_primitive
        max_scale_along_b_axis = max_num_sites // num_sites_primitive
        min_scale_along_b_axis = 1

        ratios_ac = range(min_scale_along_ac_axes, max_scale_along_ac_axes+1)
        ratios_b = range(min_scale_along_b_axis, max_scale_along_b_axis+1)

        for r_ac, r_b in itertools.product(ratios_ac, ratios_b):
            matrix_tmp = np.array([[r_ac, 0, 0], [0, r_b, 0], [0, 0, r_ac]])

            if np.linalg.det(matrix_tmp) * num_sites_primitive >= min_num_sites \
                    and np.linalg.det(matrix_tmp) * num_sites_primitive <= max_num_sites:

                abc_super = [a*r_ac, b*r_b, c*r_ac]
                abc_sorted = sorted(abc_super)
                # print(abc_super)
                if abc_sorted[2] / abc_sorted[0] <= max_anisotropy:
                    transformation_matrices.append(matrix_tmp)
                    anisotropies.append(abc_sorted[2] / abc_sorted[0])

    elif abs(b / c - 1.0) < 1.0e-4:

        min_scale_along_bc_axes = int((min_num_sites / num_sites_primitive)**0.5)
        max_scale_along_bc_axes = int((max_num_sites / num_sites_primitive)**0.5)
        min_scale_along_bc_axes = 1

        min_scale_along_a_axis = min_num_sites // num_sites_primitive
        max_scale_along_a_axis = max_num_sites // num_sites_primitive
        min_scale_along_a_axis = 1

        ratios_bc = range(min_scale_along_bc_axes, max_scale_along_bc_axes+1)
        ratios_a = range(min_scale_along_a_axis, max_scale_along_a_axis+1)

        for r_a, r_bc in itertools.product(ratios_a, ratios_bc):
            matrix_tmp = np.array([[r_a, 0, 0], [0, r_bc, 0], [0, 0, r_bc]])

            if np.linalg.det(matrix_tmp) * num_sites_primitive >= min_num_sites \
                    and np.linalg.det(matrix_tmp) * num_sites_primitive <= max_num_sites:

                abc_super = [a*r_a, b*r_bc, c*r_bc]
                abc_sorted = sorted(abc_super)
                # print(abc_super)
                if abc_sorted[2] / abc_sorted[0] <= max_anisotropy:
                    transformation_matrices.append(matrix_tmp)
                    anisotropies.append(abc_sorted[2] / abc_sorted[0])

    else:

        max_scale_along_each_axis = max_num_sites // num_sites_primitive
        ratios = range(1, max_scale_along_each_axis+1)

        print("brute force search", len(ratios), "^3")
        for r_a, r_b, r_c in itertools.product(ratios, ratios, ratios):
            matrix_tmp = np.array([[r_a, 0, 0], [0, r_b, 0], [0, 0, r_c]])

            det = np.linalg.det(matrix_tmp)

            if det * num_sites_primitive < min_num_sites or det * num_sites_primitive > max_num_sites:
                continue

            abc_super = [a*r_a, b*r_b, c*r_c]
            abc_sorted = sorted(abc_super)
            if abc_sorted[2] / abc_sorted[0] <= max_anisotropy:
                transformation_matrices.append(matrix_tmp)
                anisotropies.append(abc_sorted[2] / abc_sorted[0])

    if len(transformation_matrices) == 1:
        return transformation_matrices[0]

    if len(transformation_matrices) == 0:
        return None

    anisotropies = np.array(anisotropies)
    index_sort = np.argsort(anisotropies)

    for i in index_sort:
        matrix_tmp = transformation_matrices[i]
        if np.linalg.det(matrix_tmp) * num_sites_primitive >= 100:
            return matrix_tmp


def symmetrize_atoms(structure, symprec=0.01):
    """symmetrize structure by spglib.standardize_cell

    Args:
        structure (ase.atoms.Atoms): periodic structure.
        symprec (float, optional): symprec of spglib. Defaults to 0.01.

    Returns:
        ase.atoms.Atoms: symmetrized periodic structure.
    """
    cell = (structure.cell[:], structure.get_scaled_positions(), structure.get_atomic_numbers())
    lattice, scaled_positions, numbers = spglib.standardize_cell(cell,
                                                                 to_primitive=True,
                                                                 no_idealize=False,
                                                                 symprec=symprec)
    spacegroup_symbol = spglib.get_spacegroup(cell, symprec=symprec)
    return Atoms(numbers=numbers, cell=lattice, scaled_positions=scaled_positions), spacegroup_symbol


def read_qeoutput_from_genid(gen,
                             id_,
                             parent_dir="data",
                             qeout_filename="Y-Co-B.vcr_1.out"):
    """read qeoutput as ase.atoms.Atoms

    Args:
        gen (int): gen id number.
        id_ (int): id mumber.
        parent_dir (str, optional): parent directory name. Defaults to "data".
        qeout_filename (str, optional): qe.out filename. Defaults to "Y-Co-B.vcr_1.out".

    Raises:
        RuntimeError: failed to load the file.

    Returns:
        ase.atoms.Atoms: periodic structure.
    """
    if gen == 0:
        dir_rule = f"gen0"
    else:
        dir_rule = f"gen{gen}_*"

    id_rule = f"gen{gen}-id{id_}-x*y*"
    result_dir = f"results-gen0"
    dir_prefix = os.path.join(parent_dir, dir_rule, id_rule, result_dir)
    print("dir_prefix", dir_prefix)
    dir_list = glob.glob(dir_prefix)
    if len(dir_list) != 1:
        print(gen, id_)
        raise RuntimeError("This should not happen.")

    path_to_qeout = os.path.join(dir_list[0], qeout_filename)
    structure = read(path_to_qeout, format='espresso-out')
    return structure
