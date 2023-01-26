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
# !/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd

from ase.io.espresso import write_espresso_in
from ase import Atom, Atoms

from tempfile import TemporaryFile
import numpy as np
from itertools import combinations_with_replacement
import fnmatch
import sys

AU2ANG = 0.529177


def alm_norder_name(norder: int):
    if norder == 1:
        name = "harmonic"
    elif norder == 2:
        name = "cubic"
    else:
        raise ValueError(f"unknown norder={norder}.")
    return name


def _list_to_str(value, add="x"):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        value = add.join(list(map(str, value)))
    return value


def check_almprefixtype(value):
    flag = False
    if isinstance(value, float):
        flag = True
    elif isinstance(value, int):
        flag = True
    elif isinstance(value, str):
        flag = True
    elif isinstance(value, list):
        flag = True
    elif isinstance(value, np.ndarray):
        flag = True

    if not flag:
        typename = type(value)
        print(f"invalid type for value={value}")
        raise ValueError(f'invalid type in check_almprefixtype. type={typename}')

    return flag


class AlmBasePrefixMaker(object):
    """This class is obsolete."""
    _order = ["name", "kmesh"]

    def __init__(self, **kwargs):
        self.process_kwargs(**kwargs)

    def process_kwargs(self, **kwargs):
        """norder is converted to "harmonic" or "cubic".
        """
        prefix_list = []
        for _label in self._order:
            if _label in kwargs:
                _value = kwargs.pop(_label)
                check_almprefixtype(_value)
                if _label == "norder":
                    _value = alm_norder_name(_value)
                if isinstance(_value, list) or isinstance(_value, np.ndarray):
                    _value = _list_to_str(_value)
                if _label in ["kmesh", "qmesh"]:
                    _value = _label[0]+_value
                prefix_list.append(_value.__str__())
            else:
                raise ValueError(
                    f"No key={_label} is found in {self.__class__.__name__}:{sys._getframe().f_code.co_name}.")
        for _label, _value in kwargs.items():
            check_almprefixtype(_value)
            if isinstance(_value, list):
                _value = _list_to_str(_value)
            elif _label == "norder" and isinstance(_value, int):
                _value = alm_norder_name(_value)
            if _label in ["kmesh", "qmesh"]:
                _value = _label[0]+_value
            prefix_list.append(_value.__str__())
        self.prefix_list = prefix_list

    @property
    def prefix(self):
        return "_".join(self.prefix_list)


class AlmPrefixMaker(AlmBasePrefixMaker):
    """This class is obsolete.

    make prefix such as "{name}_k{kmesh}_{norder.__str()__}".
    """
    _ORDER = ["name", "kmesh", "norder"]

    def __init__(self, **kwargs):
        self._order = self._ORDER
        super().__init__(**kwargs)


class AlmPathMaker(AlmBasePrefixMaker):
    """This class is obsolete.

    make path name such as "{name}_k{kmesh}_{norder.__str()__}".
    """
    _ORDER = ["name", "kmesh", "norder"]

    def __init__(self, **kwargs):
        self._order = self._ORDER
        super().__init__(**kwargs)



def make_alm_uniform_kmesh(atoms: Atoms, kspacing: float = 0.05, koffset=(0, 0, 0), n=3):
    """

    Args:
        atoms (Atoms): ase.Atoms
        kspacing (float, optional): kspacing of ase.write_espresso_in. Defauls to 0.05.
        koffset ([int,int,int], optional): koffset of ase.write_espresso_in. Defaults to (0,0,0).
        n (int, optional): Only kpoints are returned if n==3. Additonally koffset if returned if n==6. Defaults to 3.

    Returns:
        [int]: kpoints (and optionally koffset).
    """
    params = {
    }

    contents = None
    with TemporaryFile(mode="w+t") as f:
        write_espresso_in(f, atoms,  # pseudopotentials =pseudo,
                          kspacing=kspacing, koffset=koffset,
                          **params)
        f.seek(0)
        contents = f.read().splitlines()
    if contents is None:
        return None

    contents_iter = iter(contents)
    while True:
        line = next(contents_iter)
        if line.startswith("K_POINTS"):
            kpline = next(contents_iter).split()
            return list(map(int, kpline[:n]))

    return None


def make_kspacing_kmeshlist(atoms: Atoms, kspacing_lim=(-1, -2.5), nmesh: int = 20, show=False):
    mesh_data = []
    for kspacing in np.logspace(kspacing_lim[0], kspacing_lim[1], nmesh):
        kspacing_mesh = [kspacing]
        kspacing_mesh.extend(make_alm_uniform_kmesh(atoms, kspacing=kspacing))
        mesh_data.append(kspacing_mesh)

    kmesh_df = pd.DataFrame(mesh_data, columns=["kspacing", "na", "nb", "nc"])
    if show:
        fig, ax = plt.subplots()
        for _ylabel in kmesh_df.columns[1:]:
            kmesh_df.plot(x="kspacing", y=_ylabel, ax=ax)
        ax.set_xscale("log")
    return kmesh_df


def _make_all_bandpath():
    """
    from https://wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html

    Reference

    High-throughput electronic band structure calculations: Challenges and tools

Wahyu Setyawan, Stefano Curtarolo

Computational Materials Science, Volume 49, Issue 2, August 2010, Pages 299–312

doi:10.1016/j.commatsci.2010.05.010

    """
    data = """CUB (primitive cubic)

    GXMGRX,MR

    ../../_images/00.CUB.svg
    FCC (face-centred cubic)

    GXWKGLUWLK,UX

    ../../_images/01.FCC.svg
    BCC (body-centred cubic)

    GHNGPH,PN

    ../../_images/02.BCC.svg
    TET (primitive tetragonal)

    GXMGZRAZ,XR,MA

    ../../_images/03.TET.svg
    BCT1 (body-centred tetragonal)

    GXMGZPNZ1M,XP

    ../../_images/04.BCT1.svg
    BCT2 (body-centred tetragonal)

    GXYSGZS1NPY1Z,XP

    ../../_images/05.BCT2.svg
    ORC (primitive orthorhombic)

    GXSYGZURTZ,YT,UX,SR

    ../../_images/06.ORC.svg
    ORCF1 (face-centred orthorhombic)

    GYTZGXA1Y,TX1,XAZ,LG

    ../../_images/07.ORCF1.svg
    ORCF2 (face-centred orthorhombic)

    GYCDXGZD1HC,C1Z,XH1,HY,LG

    ../../_images/08.ORCF2.svg
    ORCF3 (face-centred orthorhombic)

    GYTZGXA1Y,XAZ,LG

    ../../_images/09.ORCF3.svg
    ORCI (body-centred orthorhombic)

    GXLTWRX1ZGYSW,L1Y,Y1Z

    ../../_images/10.ORCI.svg
    ORCC (base-centred orthorhombic)

    GXSRAZGYX1A1TY,ZT

    ../../_images/11.ORCC.svg
    HEX (primitive hexagonal)

    GMKGALHA,LM,KH

    ../../_images/12.HEX.svg
    RHL1 (primitive rhombohedral)

    GLB1,BZGX,QFP1Z,LP

    ../../_images/13.RHL1.svg
    RHL2 (primitive rhombohedral)

    GPZQGFP1Q1LZ

    ../../_images/14.RHL2.svg
    MCL (primitive monoclinic)

    GYHCEM1AXH1,MDZ,YD

    ../../_images/15.MCL.svg
    MCLC1 (base-centred monoclinic)

    GYFLI,I1ZF1,YX1,XGN,MG

    ../../_images/16.MCLC1.svg
    MCLC3 (base-centred monoclinic)

    GYFHZIF1,H1Y1XGN,MG

    ../../_images/17.MCLC3.svg
    MCLC5 (base-centred monoclinic)

    GYFLI,I1ZHF1,H1Y1XGN,MG

    ../../_images/18.MCLC5.svg
    TRI1a (primitive triclinic)

    XGY,LGZ,NGM,RG

    ../../_images/19.TRI1a.svg
    TRI1b (primitive triclinic)

    XGY,LGZ,NGM,RG

    ../../_images/20.TRI1b.svg
    TRI2a (primitive triclinic)

    XGY,LGZ,NGM,RG

    ../../_images/21.TRI2a.svg
    TRI2b (primitive triclinic)

    XGY,LGZ,NGM,RG

    ../../_images/22.TRI2b.svg
    OBL (primitive oblique)

    GYHCH1XG

    ../../_images/23.OBL.svg
    RECT (primitive rectangular)

    GXSYGS

    ../../_images/24.RECT.svg
    CRECT (centred rectangular)

    GXA1YG

    ../../_images/25.CRECT.svg
    HEX2D (primitive hexagonal)

    GMKG

    ../../_images/26.HEX2D.svg
    SQR (primitive square)

    MGXM

    ../../_images/27.SQR.svg
    LINE (primitive line)

    GX"""
    data = data.splitlines()
    bandpath = {}
    lines = iter(data)
    while True:
        try:
            lat = next(lines)
        except StopIteration:
            break
        _ = next(lines)
        kpath = next(lines)
        lat = lat.split()[0]
        bandpath[lat] = kpath.strip()
        try:
            _ = next(lines)
            _ = next(lines)
        except StopIteration:
            break
    return bandpath


def _make_kpoint_kpmode1(atoms, nkpt=51, use_first_kpath=False):
    bandpath_dic = _make_all_bandpath()

    lat = atoms.cell.get_bravais_lattice()
    special_kpoits = lat.get_special_points()
    kpath = bandpath_dic[lat.name]
    if use_first_kpath:
        if "," in kpath:
            kpath = kpath.split(",")[0]
            if use_first_kpath:
                kpath = kpath[0]  # use only the first kpath

    lines = []
    lines.append('1')
    for k1, k2 in zip(kpath[:-1], kpath[1:]):
        if k1 == "," or k2 == ",":
            continue
        kpoint1 = list(map(str, special_kpoits[k1]))
        kpoint2 = list(map(str, special_kpoits[k2]))
        _s = [k1]
        _s.extend(kpoint1)
        _s.append(k2)
        _s.extend(kpoint2)
        _s.append(str(nkpt))
        lines.append(" ".join(_s))
    return lines


def make_alm_kpoint(atoms, kpmode, kspacing=0.0095, use_first_kpath=False, nkpt=51):
    if kpmode == 1:
        return _make_kpoint_kpmode1(atoms, use_first_kpath=use_first_kpath, nkpt=nkpt)
    elif kpmode == 2:
        kpoints = make_alm_uniform_kmesh(atoms, kspacing=kspacing)
        kpoint = []
        kpoint.append(str(kpmode))
        kpoint.append(" ".join(list(map(str, kpoints))))
        return kpoint


def _make_cutoff_pairs(uniqsymbols, cutoff, norder):
    """
    Make_cutoff_pairs accepting wild card.
    This function uses fnmatch to compare cutoff.keys().

    For example,
    uniqsymbols = ["Mg", "O"]
    norder = 2
    cutoff = {"Mg-*": [None, 5.0]}

    The size of cutoff.value must be norder.
    Both the possibilities of Mg-O and O-Mg are examined for Mg-O.

    Args:
        uniqsymbols (list): a list of unique element symbols.
        cutoff (dict): cutoff dictionary.
        norder (int): NORDER of alm suggest.

    Exception:
        ValueError: when the size of len(cutoff.value) != norder

    Returns:
        [str]: cutoff lines in the alm input format.
    """
    # cutoff
    cutoff_line = []
    for _x in combinations_with_replacement(uniqsymbols, 2):
        key = "-".join(_x)
        key1 = key
        key2 = "-".join(reversed(_x))
        values = None
        if key1 in cutoff.keys():
            values = cutoff[key1]  # must be size of norder
            values = list(map(str, values))
        elif key2 in cutoff.keys():
            values = cutoff[key2]  # must be size of norder
            values = list(map(str, values))
        else:
            for cutoffkey in cutoff.keys():
                if fnmatch.fnmatch(key1, cutoffkey):
                    values = cutoff[cutoffkey]
                    break
                if fnmatch.fnmatch(key2, cutoffkey):
                    values = cutoff[cutoffkey]
                    break
        if values is None:
            values = [None for _i in range(norder)]
        if len(values) != norder:
            raise ValueError(
                f"size of cutoff for key='{cutoffkey}' must be {norder}.")

        values = list(map(str, values))
        cutoff_line.append("{} {}".format(key, " ".join(values[:norder])))
    return cutoff_line


def atoms_to_alm_in(mode: str, superstructure: Atoms,
                    cutoff={}, prefix: str = None, norder: int = 1,
                    dic=None, ):
    """
    convert Atoms to alm.in.

    For example, the cutoff distances (in the unit of a_0) of the element-element pair if NORDER==2 is given by
        cutoff = {"Si-Si": [None, 6.5]}.
    cutoff = {"Si-Si": [None]} if NORDER==1.


    Args:
        mode (str): mode = "suggest",...
        atoms (Atoms): ase.Atoms.
        dic (dict, optional): dict of alm.in. Defaults to None.
        cutoff (dict, optional): dict of cutoff. Defaults to {}.
        prefix (str, optional): alamode prefix. Defaults to None.
        norder (int, optional): alamode norder. Defaults to 1.

    Returns:
        dict: alm_in dict.
    """
    if dic is None:
        dic = {}

    uniqZ = np.unique(superstructure.get_atomic_numbers())
    uniqZ = uniqZ.tolist()
    uniqsymbols = []
    for Z in uniqZ:
        atom = Atom(Z)
        uniqsymbols.append(atom.symbol)

    general = {}
    if prefix is not None:
        general["PREFIX"] = prefix
    else:
        general["PREFIX"] = str(superstructure.symbols)
    general['MODE'] = mode
    general["NKD"] = str(len(uniqsymbols))
    general["KD"] = " ".join(uniqsymbols)
    mass_list = []
    for symbol in uniqsymbols:
        atom = Atom(symbol)
        mass_list.append(atom.mass)
    if mode not in ["suggest", "opt"]:
        general['MASS'] = " ".join(list(map(str, mass_list)))
    if mode in ["suggest", "opt"]:  # delete it if it is defined.
        if "MASS" in general:
            general.remove("MASS")

    nat = superstructure.get_scaled_positions().shape[0]
    if mode not in ["phonons", "RTA"]:
        general["NAT"] = str(nat)

    if "general" not in dic.keys():
        dic["general"] = general
    else:
        general.update(dic["general"])
        dic["general"].update(general)

    if mode in ["phonons"]:
        if "FCSXML" not in dic["general"].keys():
            raise KeyError(f"FCSXML is necessary in mode={mode}.")

    interaction = {"NORDER": norder}
    dic["interaction"] = interaction

    cell = []
    a = superstructure.cell.ravel().max()
    cell.append(str(a/AU2ANG))
    for v in superstructure.cell:
        cell.append(" ".join((v/a).astype(str)))
    dic["cell"] = cell

    # cutoff
    dic["cutoff"] = _make_cutoff_pairs(uniqsymbols, cutoff, norder)

    # make &position
    position = []
    for z, sp in zip(superstructure.get_atomic_numbers(),
                     superstructure.get_scaled_positions()):
        _i = uniqZ.index(z)+1
        spstr = []
        for _x in sp:
            spstr.append("{:15.10f}".format(_x))
        position.append("{} {}".format(_i, " ".join(spstr)))
    dic["position"] = position

    if mode in ["phonons"]:
        if "kpoint" not in dic.keys():
            raise KeyError(f"kpoint is necessary in mode={mode}")
    return dic


def make_alm_in(param: dict, filename: str = None, handle=None):
    """
    write param to filename as alm.in

    Args:
        param (dict): dict of alm.in.
        filename (str): output filename.

    Returns:
        bool: True if suceesss.
    """
    if filename is None and handle is not None:
        # use handle
        mustclose = False
    elif filename is not None and handle is None:
        handle = open(filename, "w", encoding="utf-8")
        mustclose = True

    for key, value in param.items():
        handle.write(f'&{key}\n')
        if isinstance(value, dict):
            for key1, value1 in value.items():
                handle.write(f"{key1} = {value1}\n")
        elif isinstance(value, list):
            for _x in value:
                handle.write(_x+"\n")
        handle.write("/\n")

    if mustclose:
        handle.close()

    return True
