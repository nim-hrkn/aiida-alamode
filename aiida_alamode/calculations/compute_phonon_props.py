#!/usr/bin/env python

import argparse
import collections
import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase.io import read
from ase.units import Bohr, Ry
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from ase import Atoms

anphon_bin = os.path.join(os.environ['HOME'], 'src', 'alamode', '_build', 'anphon', 'anphon')


class PhononCalculator(object):
    """_summary_

        Anphon executable file path must be set by .set_anphon_bin() 
        or by setting the environmental variable ANPHON_BIN.
        If not, RuntimeError: "Path to anphon binary is not found." will be caused.

        fname_primitive is read in __init_ through ase.io.read, which accepts a file handler.
        fname_fcs is read by anphon. fname_fcs must be placed in the aiida folder.
    """

    def __init__(self, fname_primitive='POSCAR', fname_fcs='supercell.xml',
                 qspacing=0.1, format='vasp', tmin=0, tmax=2000, dt=10,
                 dos_energy_step=1.0, cwd=None):
        """initialization.

        if cwd is given, figures will be made.

        Args:
            fname_primitive (str|Atoms, optional): primitive structure. Defaults to 'POSCAR'.
            fname_fcs (str, optional): used for fcs in the anphon input file. Defaults to 'supercell.xml'.
            qspacing (float, optional): qspacing in the anphon input file. Defaults to 0.1.
            format (str, optional): format of fname_primitive if it is string. Defaults to 'vasp'.
            tmin (int, optional): tmin in the anphon input file. Defaults to 0.
            tmax (int, optional): tmax in the anphon input file. Defaults to 2000.
            dt (int, optional): dt in the anphon input file. Defaults to 10.
            dos_energy_step (float, optional): dos energy step in the anphon input file. Defaults to 1.0.
            cwd (str, optional): folder to place figures. Defaults to None.
        Raises:
            TypeError: unknown type for fname_primitive
        """
        if isinstance(fname_primitive, str):
            self._primitive = read(fname_primitive, format=format)
        elif isinstance(fname_primitive, Atoms):
            self._primitive = fname_primitive
        else:
            raise TypeError(f"unknown type for fname_primitive, type={type(fname_primitive)}.")
        self.natom = len(self._primitive.get_atomic_numbers())
        self._fname_fcs = fname_fcs
        self._qspacing = qspacing
        self._prefix = 'phonon'
        self._suffix = ''
        self._anphon_bin = None
        self.free_energy = None
        self.heat_capacity = None
        self.internal_energy = None
        self.entropy = None
        self.temperatures = None
        self.dos = None
        self._imaginary_branches = None
        self._frequencies = None
        self._tmin = tmin
        self._tmax = tmax
        self._dt = dt
        self._delta_e = dos_energy_step
        self._qmesh = None
        self._qpoints_irred = None
        self._weight_q = None
        self._cwd = cwd

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix):
        self._prefix = prefix

    @property
    def suffix(self):
        return self._suffix

    @suffix.setter
    def suffix(self, suffix):
        self._suffix = suffix

    @property
    def frequcncies(self):
        return self._frequencies

    def set_anphon_bin(self, anphon_bin=None):
        """set anphon binary file path.

        Args:
            anphon_bin (str, optional): anphon binary file path. Defaults to None.

        Raises:
            RuntimeError: Path to anphon binary is not found.
        """
        if anphon_bin:
            self._anphon_bin = anphon_bin
        elif os.getenv('ANPHON_BIN') is not None:
            self._anphon_bin = os.getenv('ANPHON_BIN')
        else:
            raise RuntimeError("Path to anphon binary is not found.")

    def _gen_bzpath(self):
        cell = (self._primitive.get_cell(),
                self._primitive.get_scaled_positions(),
                self._primitive.get_atomic_numbers())
        path_info = seekpath.get_path(cell, True, "hpkot", 1.0e-3, 1.0e-3, 1.0)

        return path_info

    def _generate_qmesh_from_qspacing(self):
        aa = self._primitive.get_cell()
        bb = np.linalg.inv(aa)
        b_norms = np.linalg.norm(bb, axis=0)
        qmesh = [int(max(1, np.ceil(2.0 * np.pi * x / self._qspacing))) for x in b_norms]

        return qmesh

    def gen_anphon_input(self,
                         filename,
                         kpmode=1,
                         npoints=51,
                         dos=True):
        """generate a file, filemame as anphon input.

        Args:
            filename (str): anphon input filename.
            kpmode (int, optional): kpoint mode. Defaults to 1.
            npoints (int, optional): the number of k points. Defaults to 51.
            dos (bool, optional): also calculate DOS or not. Defaults to True.
        """
        with open(filename, "w") as f:
            self.gen_anphon_input_(f, kpmode, npoints, dos)

    def gen_anphon_input_(self,
                          handler,
                          kpmode=1,
                          npoints=51,
                          dos=True):
        """generate a file, filemame as anphon input.

        Args:
            handler (TextIOWrapper): anphon input file handler.
            kpmode (int, optional): kpoint mode. Defaults to 1.
            npoints (int, optional): the number of k points. Defaults to 51.
            dos (bool, optional): also calculate DOS or not. Defaults to True.
        """
        chemical_symbols_uniq = list(
            collections.OrderedDict.fromkeys(self._primitive.get_chemical_symbols()))
        # Make input for ANPHON
        f = handler
        if True:
            f.write("&general\n")
            f.write(" PREFIX = %s\n" % self._prefix)
            f.write(" MODE = phonons\n")
            str_spec = ""
            for entry in chemical_symbols_uniq:
                str_spec += entry + " "
            f.write(" NKD = %i; KD = %s\n" % (len(chemical_symbols_uniq), str_spec))
            f.write(" TOLERANCE = 1.0e-3\n")
            f.write(" FCSXML = %s\n" % self._fname_fcs)
            f.write(" TMIN = %f; TMAX = %f; DT = %f\n" % (self._tmin, self._tmax, self._dt))
            f.write(" DELTA_E = %f\n" % self._delta_e)
            f.write("/\n\n")
            f.write("&cell\n")
            f.write("%20.14f\n" % (1.0 / Bohr))
            for i in range(3):
                for j in range(3):
                    f.write("%20.13f" % self._primitive.get_cell()[i][j])
                f.write("\n")
            f.write("/\n\n")

            if kpmode == 1:
                f.write("&kpoint\n")
                f.write(" 1\n")

                path_info = self._gen_bzpath()
                kpath = path_info["path"]
                point_coords = path_info["point_coords"]

                for line in kpath:
                    f.write(" %s" % line[0])
                    coord_s = point_coords[line[0]]
                    coord_e = point_coords[line[1]]
                    for coord in coord_s:
                        f.write(" %15.8f" % coord)

                    f.write(" %s" % line[1])
                    for coord in coord_e:
                        f.write("%15.8f" % coord)
                    f.write(" %d\n" % npoints)
                f.write("/\n\n")

            elif kpmode == 2:
                qmesh = self._generate_qmesh_from_qspacing()
                self._qmesh = qmesh
                f.write("&kpoint\n")
                f.write(" 2\n")
                f.write(' %d %d %d\n' % (qmesh[0], qmesh[1], qmesh[2]))
                f.write("/\n")

                if not dos:
                    f.write("&analysis\n")
                    f.write(" DOS = 0\n")
                    f.write("/\n")

    def run_anphon(self, fname_in, fname_out):
        """run anphon with subprocess.call.

        Args:
            fname_in (str): input filename
            fname_out (str): output filename

        Raises:
            RuntimeError: "Anphon binary is not set properly."
        """
        if not self._anphon_bin:
            raise RuntimeError("Anphon binary is not set properly.")
        fout = open(fname_out, 'w')
        subprocess.call([self._anphon_bin, fname_in], stdout=fout)
        fout.close()

    def compute(self, joblist):
        """compute band and/or thermo.

        Args:
            joblist (str): "band" or "dos+thermo" or "thermo".
        """
        for job in joblist:
            if job == 'band':
                self.gen_anphon_input('phband.in', kpmode=1)
                self.run_anphon('phband.in', 'phband%s.log' % self._suffix)

            elif job == 'dos+thermo':
                self.gen_anphon_input('phdos.in', kpmode=2)
                self.run_anphon('phdos.in', 'phdos%s.log' % self._suffix)
                self._parse_dos()
                self._parse_thermo()
                self._parse_frequencies('phdos%s.log' % self._suffix)
                self._detect_imaginary_branches()

            elif job == 'thermo':
                self.gen_anphon_input('phdos.in', kpmode=2, dos=False)
                self.run_anphon('phdos.in', 'phdos%s.log' % self._suffix)
                self._parse_thermo()
                self._parse_frequencies('phdos%s.log' % self._suffix)
                self._detect_imaginary_branches()

    def _parse_dos(self):
        """load the content of the dos file.
        """
        if self.dos is None:
            fname_dos = self._prefix + '.dos'
            if os.path.exists(fname_dos):
                self.parse_dos_(fname_dos)

    def parse_dos_(self, f):
        """load the content of the dos file.

        Args:
            f (TextIOWrapper): file handler.
        """
        data = np.loadtxt(f)
        self.dos = data

    def _parse_thermo(self):
        """load the content of the thermo file.
        """
        if self.temperatures is None:
            fname_thermo = self._prefix + '.thermo'
            if os.path.exists(fname_thermo):
                with open(fname_thermo) as f:
                    self.parse_thermo_(f)

    def parse_thermo_(self, handle):
        """load the content of the thermo file.

        """
        data = np.loadtxt(handle)
        self.temperatures = data[:, 0]
        self.heat_capacity = data[:, 1]
        self.entropy = data[:, 2]
        self.internal_energy = data[:, 3]
        self.free_energy = data[:, 4]

    def _parse_frequencies(self, fname_log):
        """parse the log file to get phonon frequencies.

        Args:
            fname_log (str): log file name.
        """
        if self._frequencies is None:
            if os.path.exists(fname_log):
                with open(fname_log, 'r') as f:
                    self.parse_frequencies_(f)

    def parse_frequencies_(self, handle):
        """parse the log file to get phonon frequencies.

        Args:
            handle (TextIOWrapper): log file handler.
        """
        with handle:
            lines = handle.read().splitlines()
            lines_iter = iter(lines)
            # parse number of irred. q points
            nq_irred = 0
            try:
                while True:
                    line = next(lines_iter)
                    if "Number of irreducible k points" in line:
                        nq_irred = int(line.split()[-1])
                        break
            except StopIteration:
                pass
            # parse q points and weights
            xq = []
            weight_q = []
            lines_iter = iter(lines)
            try:
                while True:
                    line = next(lines_iter)
                    if "List of irreducible k points (reciprocal coordinate, weight) :" in line:
                        for iq in range(nq_irred):
                            line = next(lines_iter)
                            entry = [float(t) for t in line.split()[1:]]
                            xq.append(entry[:3])
                            weight_q.append(entry[3])
                        break
            except StopIteration:
                pass
            self._qpoints_irred = np.array(xq)
            self._weight_q = np.array(weight_q)

            # parse frequencies
            frequencies = []
            lines_iter = iter(lines)
            try:
                while True:
                    line = next(lines_iter)
                    if "THz )" in line:
                        frequencies.append(float(line.split()[1]))
            except StopIteration:
                pass
            frequencies = np.array(frequencies)

            self._frequencies = np.reshape(frequencies, (len(self._qpoints_irred),
                                                         3 * len(self._primitive.get_atomic_numbers())))

    def _detect_imaginary_branches(self):
        """detect the disperson branches containing imaginary frequencies
           if self._imaginary_branches and self._frequencies  is set.
           The resulting (banch number, kpoint number and frequency) will be set to ._imaginry_branches.
        """
        if self._imaginary_branches is None and self._frequencies is not None:
            imaginary_list = []
            nq, nb = np.shape(self._frequencies)
            for iq in range(nq):
                for ib in range(nb):
                    if self._frequencies[iq, ib] < 0.0:
                        imaginary_list.append([iq, ib, self._frequencies[iq, ib]])

            self._imaginary_branches = np.array(imaginary_list)

    def get_imaginary_ratio(self, threshold=0.0):
        """calculate ratio of imaginary phonon frequency among all.

        Args:
            threshold (float, optional): thredhold to take the imaginary phonon. Defaults to 0.0.

        Returns:
            float: ratio of imaginary phonon frequency.
        """
        if self._imaginary_branches is not None:
            if len(self._imaginary_branches) == 0:
                return 0.0

            nq, nb = np.shape(self._frequencies)
            ratio = 0.0
            for entry in self._imaginary_branches:
                iq = round(entry[0])
                freq = entry[2]
                if freq < threshold:
                    ratio += self._weight_q[iq]

            ratio /= nb

            return ratio

        else:
            return None

    def get_lowest_frequency(self):
        """get the lowest phonon frequency to check having imaginary phonon frequency.
        The imaginary phonon freq. is outputed as mimus energy.

        Returns:
            float: the lowest frequency or None.
        """
        if self._frequencies is not None:
            tmp = self._frequencies[:, 0]
            return np.sort(tmp)[0]
        else:
            return None

    def make_figures(self, joblist):
        """make figures depending on joblist.
        joblist can be 'band', 'dos+thermo', 'thermo'.

        {prefix}_(band,dos,themo).pdf will be made.

        Args:
            joblist (str): job string.
        """
        if self._cwd is None:
            return
        for job in joblist:
            if job == 'band':
                self._plot_band()

            elif job == 'dos+thermo':
                self._plot_thermo()
                self._plot_dos()

            elif job == 'thermo':
                self._plot_thermo()

    def _plot_band(self):
        """plot band dispersion

        {prefix}_band.pdf will be made.
        """
        if self._cwd is None:
            return
        unit = 'meV'
        fname_to_plot = '%s.bands' % self._prefix
        files = [fname_to_plot]
        normalize_xaxis = False

        nax, xticks_ax, xticklabels_ax, xmin_ax, xmax_ax, ymin, ymax, data_merged_ax \
            = self._preprocess_data(files, unit, normalize_xaxis)

        gs = GridSpec(1, 1)
        self._setformat_plot(xtick_size=12, ytick_size=16)
        self._add_plot(gs[0], nax, xticks_ax, xticklabels_ax,
                       xmin_ax, xmax_ax, ymin, ymax, data_merged_ax, unit, files)

        os.makedirs(self._cwd, exist_ok=True)
        plt.savefig(os.path.join(self._cwd, '%s_band.pdf' % self._prefix), bbox_inches='tight')
        plt.clf()

    def _plot_dos(self):
        """plot DOS if self.dos is True.

        {prefix}_dos.pdf will be made.
        """
        if self._cwd is None:
            return
        if self.dos is None:
            return
        gs = GridSpec(1, 1)
        self._setformat_plot(xtick_size=12, ytick_size=12)
        kayser_to_mev = 0.0299792458 * 1.0e+12 * \
            6.62606896e-34 / 1.602176565e-19 * 1000

        ax = plt.subplot(gs[0])
        ax.plot(self.dos[:, 0] * kayser_to_mev, self.dos[:, 1] / kayser_to_mev)
        ax.set_ylim(0)
        ax.set_xlabel('Frequency (meV)')
        ax.set_ylabel('DOS (states/meV/cell)')
        os.makedirs(self._cwd, exist_ok=True)
        plt.savefig(os.path.join(self._cwd, '%s_dos.pdf' % self._prefix), bbox_inches='tight')
        plt.clf()

    def _plot_thermo(self):
        """plot thermo if self.temperatures is not None.

        {prefix}_thermo.pdf will be made.
        """
        if self._cwd is None:
            return
        if self.temperatures is None:
            return
        gs = GridSpec(2, 2)
        gs.update(hspace=0.25, wspace=0.4)
        self._setformat_plot(xtick_size=12, ytick_size=12)

        ax = plt.subplot(gs[0, 0])
        ax.plot(self.temperatures, self.heat_capacity)
        ax.axhline(y=3 * len(self._primitive.get_atomic_numbers()),
                   linestyle=':', color='black')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_ylabel('$C_v/k_\mathrm{B}$')

        ax = plt.subplot(gs[0, 1])
        ax.plot(self.temperatures, self.entropy, color='C1')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_ylabel('$S/k_\mathrm{B}$')

        ax = plt.subplot(gs[1, 0])
        ax.plot(self.temperatures, self.internal_energy * Ry, color='C2')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('$U_{\mathrm{vib}}$/cell (eV)')

        ax = plt.subplot(gs[1, 1])
        ax.plot(self.temperatures, self.free_energy * Ry, color='C3')
        ax.set_xlim(0)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('$F_{\mathrm{vib}}$/cell (eV)')

        os.makedirs(self._cwd, exist_ok=True)
        plt.savefig(os.path.join(self._cwd, '%s_thermo.pdf' % self._prefix), bbox_inches='tight')
        plt.clf()

    def _setformat_plot(self, xtick_size=16, ytick_size=16):
        # font styles
        mpl.use('Agg')
        mpl.rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Helvetica']})
        mpl.rc('xtick', labelsize=xtick_size)
        mpl.rc('ytick', labelsize=ytick_size)
        mpl.rc('axes', labelsize=16)
        mpl.rc('lines', linewidth=1.5)
        mpl.rc('legend', fontsize='small')

    def _change_scale(self, array, str_scale):
        str_tmp = str_scale.lower()

        if str_tmp == 'kayser':
            return array

        elif str_tmp == 'mev':
            kayser_to_mev = 0.0299792458 * 1.0e+12 * \
                6.62606896e-34 / 1.602176565e-19 * 1000

            for i in range(len(array)):
                for j in range(len(array[i])):
                    for k in range(1, len(array[i][j])):
                        array[i][j][k] *= kayser_to_mev

            return array

        elif str_tmp == 'thz':
            kayser_to_thz = 0.0299792458

            for i in range(len(array)):
                for j in range(len(array[i])):
                    for k in range(1, len(array[i][j])):
                        array[i][j][k] *= kayser_to_thz

            return array

        else:
            print("Unrecognizable option for --unit %s" % str_scale)
            print("Band structure will be shown in units of cm^{-1}")
            return array

    def _normalize_to_unity(self, array, array_axis):
        for i in range(len(array)):
            max_val = array[i][-1][0]

            factor_normalize = 1.0 / max_val

            for j in range(len(array[i])):
                array[i][j][0] *= factor_normalize

        max_val = array_axis[-1]
        factor_normalize = 1.0 / max_val

        for i in range(len(array_axis)):
            array_axis[i] *= factor_normalize

        return array, array_axis

    def _get_xy_minmax(self, array):
        xmin, xmax, ymin, ymax = [0, 0, 0, 0]

        for i in range(len(array)):
            xtmp = array[i][-1][0]
            xmax = max(xmax, xtmp)

        for i in range(len(array)):
            for j in range(len(array[i])):
                for k in range(1, len(array[i][j])):
                    ytmp = array[i][j][k]
                    ymin = min(ymin, ytmp)
                    ymax = max(ymax, ytmp)

        return xmin, xmax, ymin, ymax

    def _gridspec_setup(self, data_merged, xtickslabels, xticksvars):
        xmaxs = []
        xmins = []

        xticks_grids = []
        xticklabels_grids = []
        xticklabels_tmp = []
        xticks_tmp = []

        for i in range(len(xtickslabels)):

            if i == 0:
                xmins.append(xticksvars[0])
            else:
                if xticksvars[i] == xticksvars[i - 1]:
                    xmaxs.append(xticksvars[i - 1])
                    xmins.append(xticksvars[i])

                    xticks_grids.append(xticks_tmp)
                    xticklabels_grids.append(xticklabels_tmp)
                    xticklabels_tmp = []
                    xticks_tmp = []

            xticklabels_tmp.append(xtickslabels[i])
            xticks_tmp.append(xticksvars[i])

        xticks_grids.append(xticks_tmp)
        xticklabels_grids.append(xticklabels_tmp)
        xmaxs.append(xticksvars[-1])

        naxes = len(xticks_grids)
        nfiles = len(data_merged)

        data_all_axes = []

        for i in range(naxes):
            data_ax = []

            xmin_ax = xmins[i]
            xmax_ax = xmaxs[i]

            for j in range(nfiles):

                kval = np.array(data_merged[j][0:, 0])
                ix_xmin_arr = np.where(kval <= xmin_ax)
                ix_xmax_arr = np.where(kval >= xmax_ax)

                if len(ix_xmin_arr[0]) > 0:
                    ix_xmin = int(ix_xmin_arr[0][-1])
                else:
                    ix_xmin = 0

                if len(ix_xmax_arr[0]) > 0:
                    ix_xmax = int(ix_xmax_arr[0][0])
                else:
                    ix_xmax = -2

                data_ax.append(data_merged[j][ix_xmin:(ix_xmax + 1), :])

            data_all_axes.append(data_ax)

        return naxes, xticks_grids, xticklabels_grids, xmins, xmaxs, data_all_axes

    def _preprocess_data(self, files, unitname, normalize_xaxis):
        xtickslabels, xticksvars = self._get_kpath_and_kval(files[0])

        data_merged = []

        for file in files:
            data_tmp = np.loadtxt(file, dtype=float)
            data_merged.append(data_tmp)

        data_merged = self._change_scale(data_merged, unitname)

        if normalize_xaxis:
            data_merged, xticksvars = self._normalize_to_unity(data_merged, xticksvars)

        xmin, xmax, ymin, ymax = self._get_xy_minmax(data_merged)

        factor = 1.05
        ymin *= factor
        ymax *= factor

        naxes, xticks_grids, xticklabels_grids, xmins, xmaxs, data_merged_grids \
            = self._gridspec_setup(data_merged, xtickslabels, xticksvars)

        return naxes, xticks_grids, xticklabels_grids, \
            xmins, xmaxs, ymin, ymax, data_merged_grids

    def _get_kpath_and_kval(self, file_in):
        kpath_list, kval_float = [], []
        with open(file_in, 'r') as ftmp:
            kpath_list, kval_float = self._get_kpath_and_kval_(ftmp)
        return kpath_list, kval_float

    def _get_kpath_and_kval_(self, ftmp):
        kpath = ftmp.readline().rstrip('\n').split()
        kval = ftmp.readline().rstrip('\n').split()

        if kpath[0] == '#' and kval[0] == '#':
            kval_float = [float(val) for val in kval[1:]]
            kpath_list = []
            for i in range(len(kpath[1:])):
                if kpath[i + 1] == 'GAMMA':
                    kpath_list.append('$\Gamma$')
                elif kpath[i + 1] == 'SIGMA_0':
                    kpath_list.append('$\Sigma_0$')
                else:
                    kpath_list.append("$\mathrm{%s}$" % kpath[i + 1])

            return kpath_list, kval_float
        else:
            return [], []

    def _add_plot(self, ax, nax, xticks_ax, xticklabels_ax, xmin_ax, xmax_ax, ymin, ymax, data_merged_ax,
                  unitname, files):
        color = ['b', 'g', 'r', 'm', 'k', 'c', 'y', 'r']
        lsty = ['-', '-', '-', '-', '--', '--', '--', '--']

        width_ratios = []
        for xmin, xmax in zip(xmin_ax, xmax_ax):
            width_ratios.append(xmax - xmin)

        gs = GridSpecFromSubplotSpec(nrows=1, ncols=nax, wspace=0.1,
                                     width_ratios=width_ratios, subplot_spec=ax)

        for iax in range(nax):
            ax = plt.subplot(gs[iax])

            for i in range(len(data_merged_ax[iax])):

                if len(data_merged_ax[iax][i]) > 0:
                    ax.plot(data_merged_ax[iax][i][0:, 0], data_merged_ax[iax][i][0:, 1],
                            linestyle=lsty[i], color=color[i], label=files[i])

                    for j in range(2, len(data_merged_ax[iax][i][0][0:])):
                        ax.plot(data_merged_ax[iax][i][0:, 0], data_merged_ax[iax][i][0:, j],
                                linestyle=lsty[i], color=color[i])

            if iax == 0:
                if unitname.lower() == "mev":
                    ax.set_ylabel("Frequency (meV)", labelpad=20)
                elif unitname.lower() == "thz":
                    ax.set_ylabel("Frequency (THz)", labelpad=20)
                else:
                    ax.set_ylabel("Frequency (cm${}^{-1}$)", labelpad=10)

            else:
                ax.set_yticklabels([])
                ax.set_yticks([])

            plt.axis([xmin_ax[iax], xmax_ax[iax], ymin, ymax])
            ax.set_xticks(xticks_ax[iax])
            ax.set_xticklabels(xticklabels_ax[iax])
            ax.xaxis.grid(True, linestyle='-')


def run(args):
    joblist = []

    if args.mode == 'all':
        joblist = ['dos+thermo', 'band']
    elif args.mode == 'band':
        joblist = ['band']
    elif args.mode == 'dos':
        joblist = ['dos+thermo']
    elif args.mode == 'thermo':
        joblist = ['thermo']

    calculator = PhononCalculator(fname_primitive=args.VASP,
                                  fname_fcs=args.fc,
                                  qspacing=args.qspacing)
    calculator.set_anphon_bin()
    calculator.prefix = args.prefix
    calculator.compute(joblist)

    if args.savefig:
        calculator.make_figures(joblist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--VASP',
                        metavar='POSCAR',
                        type=str,
                        default='POSCAR',
                        help="VASP POSCAR file of the primitive cell with equilibrium atomic positions")

    parser.add_argument('--mode',
                        default='all',
                        type=str,
                        help="Types of calculation. DOS, band, thermo, or all")

    parser.add_argument('--qspacing',
                        metavar='0.1',
                        type=float,
                        default=0.1,
                        help="The spacing of q point sampling in units of A^{-1}.")

    parser.add_argument('--fc',
                        metavar='supercell.xml',
                        type=str,
                        default='supercell.xml',
                        help="The FCSXML file used for phonon calculations")

    parser.add_argument('--savefig', action="store_true", dest="savefig", default=False,
                        help="Save figures to pdf files.")

    parser.add_argument('--prefix',
                        metavar='phonon',
                        type=str,
                        default='phonon',
                        help='File name prefix of anphon outputs')

    args = parser.parse_args()
    run(args)
