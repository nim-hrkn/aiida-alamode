#!/usr/bin/env python
#
# displace.py
#
# Simple script to generate input files of given displacement patterns.
# Currently, VASP, Quantum-ESPRESSO, LAMMPS, OpenMX, and xTAPP are supported.
#
# Copyright (c) 2014-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

"""
Input structure file generator for displaced configurations.
"""

from __future__ import print_function
import argparse
import numpy as np
from interface.VASP import VaspParser
from interface.QE import QEParser
from interface.xTAPP import XtappParser
from interface.OpenMX import OpenmxParser
from interface.LAMMPS import LammpsParser
from GenDisplacement import AlamodeDisplace

parser = argparse.ArgumentParser()

parser.add_argument('--mag',
                    type=float, default=0.02,
                    help="Magnitude of displacement in units of \
                        Angstrom (default: 0.02)")

parser.add_argument('--prefix',
                    type=str, default="disp",
                    help="Prefix of the files to be created. (default: disp)")

parser.add_argument('--QE',
                    metavar='supercell.pw.in',
                    help="Quantum-ESPRESSO input file with equilibrium atomic positions (default: None)")

parser.add_argument('--VASP',
                    metavar='SPOSCAR',
                    help="VASP POSCAR file with equilibrium atomic \
                        positions (default: None)")

parser.add_argument('--xTAPP',
                    metavar='supercell.cg',
                    help="xTAPP CG file with equilibrium atomic \
                        positions (default: None)")

parser.add_argument('--LAMMPS',
                    metavar='supercell.lammps',
                    help="LAMMPS structure file with equilibrium atomic positions (default: None)")

parser.add_argument('--OpenMX',
                    metavar='supercell.dat',
                    help="dat file with equilibrium atomic \
                        positions (default: None)")

parser.add_argument('-pf', '--pattern_file', metavar='prefix.pattern_*', type=str, nargs='+',
                    help="ALM pattern file(s) generated with MODE = suggest")

parser.add_argument('--random', action="store_true", dest="random", default=False,
                    help="Generate randomly-displaced structures.")

parser.add_argument('--random_normalcoord', action="store_true", dest="random_normalcoord", default=False,
                    help="Generate randomly-displaced structures in normal coordinate basis. "
                         "Please give the --temp option as well.")

parser.add_argument('--temp', type=float, default=100,
                    help="Target temperature of the random distribution of \
                        Q (default: 100). Used if --MD is not given.")

parser.add_argument('--ignore_imag', action="store_true", dest="ignore_imag", default=False,
                    help="Ignore imaginary modes when generating random displacements by"
                         "--random_normalcoord option. By default, imaginary frequency"
                         "will be replaced with its absolute value.")

parser.add_argument('-e', '--every', type=str, default="50", metavar='start:end:interval',
                    help="Specify the range and interval of data sampling. "
                         "--every=1:1000:10 means sampling one structure for every 10 snapshots"
                         "from the 1st step to the 1000th step. Used if --MD is given."
                         "(default: 50)")

parser.add_argument('-md', '--load_mddata', type=str, nargs='+', default=None,
                    help="Specify the file(s) containing displacements of MD trajectories.")

parser.add_argument('--prim', type=str, default=None,
                    help="Specify the file containing structure data of the primitive lattice.")

parser.add_argument('--evec', type=str, default=None,
                    help="Specify the filename containing eigenvalues and eigenvectors.")

parser.add_argument('-nd', '--num_disp', type=int, default=1,
                    help="Specify the number of displacement patterns.")

parser.add_argument('-cl', '--classical', action="store_true", dest="classical", default=False,
                    help="Use classical expectation value for <Q^2>.")

parser.add_argument('-p', '--print', action="store_true", dest="print_disp_stdout", default=False,
                    help="Print displacements to stdout. The unit is Angstrom.")

parser.add_argument('--pes', type=str, default=None, metavar='"q_index branch_index"',
                    help="Specify the target mode to displace atoms for calculating "
                         "the potential energy surface. --pes='5 10' will generate displacements"
                         "that correspond to the phonon mode at the 5th q point and the 10th branch."
                         "Only the real part of the phonon eigenvector is used by default."
                         "To use the imaginary part instead, please add --imag-evec option.")

parser.add_argument('--imag_evec', action="store_true", dest="imag_evec", default=False,
                    help="In the --pes mode, the imaginary part of the phonon eigenvectors"
                         "will be used when --imag_evec option is given. By default, the"
                         "real part of the eigenvectors are used. Please be noted that "
                         "adding this option generates zero displacements for the phonon modes"
                         "at the Brillouin zone center and boundaries because the eigenvectors"
                         "at these points are purely real.")

parser.add_argument('--Qrange', type=str, default=None, metavar='"Qmin Qmax"',
                    help='Range of normal coordinate amplitude Q in units of amu^{1/2}*Angstrom')


def check_code_options(args):
    conditions = [args.VASP is None,
                  args.QE is None,
                  args.xTAPP is None,
                  args.LAMMPS is None,
                  args.OpenMX is None]

    if conditions.count(True) == len(conditions):
        raise RuntimeError(
            "Error : Either --VASP, --QE, --xTAPP, --LAMMPS, "
            "--OpenMX option must be given.")

    elif len(conditions) - conditions.count(True) > 1:
        raise RuntimeError("Error : --VASP, --QE, --xTAPP, --LAMMPS, and "
                           "--OpenMX cannot be given simultaneously.")

    elif args.VASP:
        code = "VASP"
        struct_format = "VASP POSCAR"
        str_outfiles = "%s{counter}.POSCAR" % args.prefix
        file_original = args.VASP

    elif args.QE:
        code = "QE"
        struct_format = "Quantum-ESPRESSO pw.in format"
        str_outfiles = "%s{counter}.pw.in" % args.prefix
        file_original = args.QE

    elif args.xTAPP:
        code = "xTAPP"
        struct_format = "xTAPP cg format"
        str_outfiles = "%s{counter}.cg" % args.prefix
        file_original = args.xTAPP

    elif args.LAMMPS:
        code = "LAMMPS"
        struct_format = "LAMMPS structure format"
        str_outfiles = "%s{counter}.lammps" % args.prefix
        file_original = args.LAMMPS

    elif args.OpenMX:
        code = "OpenMX"
        struct_format = "OpenMX dat format"
        str_outfiles = "%s{counter}.dat" % args.prefix
        file_original = args.OpenMX

    return code, file_original, struct_format, str_outfiles


def check_displace_options(args, code):

    conditions = [args.pattern_file is None,
                  args.load_mddata is None,
                  args.random is False,
                  args.random_normalcoord is False,
                  args.pes is None]

    number_of_matches = conditions.count(True)

    if number_of_matches == len(conditions):
        raise RuntimeError(
            "Either --pattern_file (-pf), --load_mddata (-md), "
            "--random, --random_normalcoord, or --pes must be given.")

    elif (len(conditions) - number_of_matches == 2 and not (args.load_mddata and args.random)) \
            or len(conditions) - number_of_matches > 2:
        raise RuntimeError("The given combination of --pattern_file (-pf), "
                           "--load_mddata (-md), --random, --random_normalcoord, "
                           "and --pes is not allowed. Only combination allowed is "
                           "--load_mddata and --random.")

    if args.pattern_file:
        displacement_mode = "fd"

    elif args.load_mddata:
        if args.random:
            displacement_mode = "md_plus_random"
        else:
            displacement_mode = "md"

    elif args.random_normalcoord:
        displacement_mode = "random_normalcoordinate"

    elif args.pes:
        displacement_mode = "pes"

    else:
        displacement_mode = "random"

    if args.load_mddata and (code == "OpenMX" or code == "LAMMPS" or code == "xTAPP"):
        print("For %s, --load_mddata only accepts displacement file generated by \n"
              "extract.py with --get disp option.\n" % code)

    if (args.pes or displacement_mode == "random_normalcoordinate") and code == "LAMMPS":
        raise RuntimeError("sorry. --random_normalcoord and --pes are not supported for LAMMPS.")

    return displacement_mode


def get_code_object(code):
    if code == "VASP":
        return VaspParser()

    if code == "QE":
        return QEParser()

    if code == "OpenMX":
        return OpenmxParser()

    if code == "xTAPP":
        return XtappParser()

    if code == "LAMMPS":
        return LammpsParser()


def displace(displacement_mode, codeobj, args):

    verbosity = 1
    if args.print_disp_stdout:
        verbosity = 0

    dispobj = AlamodeDisplace(displacement_mode, codeobj,
                              file_primitive=args.prim,
                              file_evec=args.evec,
                              verbosity=verbosity)

    return dispobj.generate(file_pattern=args.pattern_file,
                            file_mddata=args.load_mddata,
                            option_every=args.every,
                            magnitude=args.mag,
                            number_of_displacements=args.num_disp,
                            temperature=args.temp,
                            classical=args.classical,
                            option_pes=args.pes,
                            option_qrange=args.Qrange,
                            ignore_imag=args.ignore_imag,
                            imag_evec=args.imag_evec)


def print_displacement_stdout(disp_list, codeobj):

    lavec_transpose = codeobj.lattice_vector.transpose()

    for disp in disp_list:
        disp_tmp = np.dot(disp, lavec_transpose)
        for i in range(codeobj.nat):
            print("%15.7f %15.7f %15.7f" % (disp_tmp[i, 0],
                                            disp_tmp[i, 1],
                                            disp_tmp[i, 2]))
        print('')


if __name__ == '__main__':

    args = parser.parse_args()

    if not args.print_disp_stdout:
        print("*****************************************************************")
        print("    displace.py --  Generator of displaced configurations        ")
        print("                      Version. 1.2.1                             ")
        print("*****************************************************************")
        print("")

    print(args)
    print(type(args))

    code, file_original, struct_format, str_outfiles = check_code_options(args)
    displacement_mode = check_displace_options(args, code)

    codeobj = get_code_object(code)
    codeobj.load_initial_structure(file_original)

    if not args.print_disp_stdout:
        print(" Output format                  : %s" % struct_format)
        print(" Structure before displacements : %s" % file_original)
        print(" Output file names              : %s" % str_outfiles)
        print(" Magnitude of displacements     : %s Angstrom" % args.mag)
        print(" Number of atoms                : %i" % codeobj.nat)
        print("")

    header_list, disp_list = displace(displacement_mode, codeobj, args)

    if not args.print_disp_stdout:
        print(" Number of displacements        : %i" % len(disp_list))
        print("-----------------------------------------------------------------")
        print("")
        codeobj.generate_structures(args.prefix, header_list, disp_list)
        print("All input files are created.")
        import sys
        sys.exit(0)
    else:
        print_displacement_stdout(disp_list, codeobj)
