#!/usr/bin/env python

import argparse

import numpy as np
from alm import ALM
from ase.io import read
from ase.units import Ry, Bohr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from ase import Atoms
import pandas as pd


class ForceConstantModelSelector(object):

    def __init__(self, fname_supercell='SPOSCAR', format='vasp', maxorder=1):
        """initialization function.

        fname_supercell file is read using ase.io.read() with format.

        Args:
            fname_supercell (str|Atoms, optional): file path with equilibrium atomic positions.. Defaults to 'SPOSCAR'.
            format (str, optional): file format of fname_supercell. Defaults to 'vasp'.
            maxorder (int, optional): maximum order of phonon calculation passed to ALM, =1 means then harminic term. 
            Defaults to 1.
        """
        if isinstance(fname_supercell, str):
            self._supercell = read(fname_supercell, format=format)
        elif isinstance(fname_supercell, Atoms):
            self._supercell = fname_supercell
        else:
            raise TypeError(f"unknown type for fname_supercell, type={type(fname_supercell)}.")
        self._lavec = self._supercell.get_cell() / Bohr
        self._xcoord = self._supercell.get_scaled_positions()
        self._numbers = self._supercell.get_atomic_numbers()
        self._crystal = (self._lavec, self._xcoord, self._numbers)
        self._disp = None
        self._force = None
        dist_min, dist_max, dist_unique = self._get_distance_info(unit='bohr')
        self.dist_min = dist_min
        self.dist_max = dist_max
        self._dist_unique = dist_unique
        self._maxorder = maxorder
        self._cutoff_3rd = 5.0

        numbers_uniq = list(set(np.sort(self._numbers)))
        self._nspecies = len(numbers_uniq)
        self._amat = None
        self._bvec = None

    def load_training_data(self, num_snapshots=None, fname_dfset='DFSET'):
        """load training data.

        Args:
            num_snapshots (int, optional): len(force). Defaults to None.
            fname_dfset (str|np.ndarray, optional): DFSET filename. Defaults to 'DFSET'.
        """
        numbers = len(self._numbers)
        if isinstance(fname_dfset, str):
            dfset = np.loadtxt(fname_dfset).reshape((-1, numbers, 6))
        elif isinstance(fname_dfset, np.ndarray):
            dfset = fname_dfset
        else:
            raise TypeError(f"unknown type for fname_dfset, {type(fname_dfset)}")
        force = dfset[:, :, 3:]
        disp = dfset[:, :, :3]

        if num_snapshots:
            if num_snapshots > len(force):
                num_snapshots = len(force)
        else:
            num_snapshots = len(force)

        self._disp = disp[:num_snapshots]
        self._force = force[:num_snapshots]

    @property
    def displacement(self):
        """returns displacement.

        Returns:
            _type_: displacement vector, disp[:num_snapshots].
        """
        return self._disp

    def _get_distance_info(self, unit='bohr'):
        """get distance information

        Args:
            unit (str, optional): the unit of returns. Defaults to 'bohr'.

        Returns:
            tuple containing,
            float: dist_min.
            float: dist_max.
            float: dist_unique.
        """
        distances = self._supercell.get_all_distances(mic=True)
        dist_list = []
        for entry in distances:
            for entry2 in entry:
                if entry2 > 0.0:
                    dist_list.append(entry2)

        dist_list = np.array(dist_list)
        dist_min = np.min(dist_list)
        dist_max = np.max(dist_list)

        dist_sorted = np.sort(dist_list)

        dist_unique = []
        dist_prev = 0.0
        for entry in dist_sorted:
            if (entry - dist_prev) > 0.01:
                dist_unique.append(entry)
                dist_prev = entry

        dist_unique = np.array(dist_unique)

        if unit == 'bohr':
            dist_min /= Bohr
            dist_max /= Bohr
            dist_unique /= Bohr

        return dist_min, dist_max, dist_unique

    def gen_sensing_matrix(self, maxorder=None, cutoff=None, nbody=None):
        """generate sensing matrix.

        Args:
            maxorder (int, optional): maximum order of phonon calculation. Defaults to None.
            cutoff (float, optional): cutoff of phonon calculation. Defaults to None.
            nbody (int, optional): upto nth boday term. Defaults to None.
        """
        with ALM(*self._crystal) as alm:
            alm.verbosity = 0
            alm.define(maxorder, cutoff, nbody)
            alm.set_constraint(translation=True)
            alm.displacements = self._disp
            alm.forces = self._force
            X, y = alm.get_matrix_elements()
            self._amat = X
            self._bvec = y

    @property
    def amat(self):
        """returns amat.

        Returns:
            float: amat
        """
        return self._amat

    @property
    def bvec(self):
        """returns bvec.

        Returns:
            np.ndarray: bvector
        """
        return self._bvec

    def run_ols_cross_validation(self, n_splits=5):
        """train a linear model with CV and return CV scores.

        Args:
            n_splits (int, optional): _description_. Defaults to 5.

        Returns:
            np.ndarray: (rmse_mean, rmse_std, cvc_mean, cv_std) in meV/atom.
        """
        ndata = len(self._disp)
        nat = len(self._numbers)
        groups = np.repeat(np.arange(ndata), 3 * nat)
        gkf = GroupKFold(n_splits=n_splits)

        rms_train = np.zeros(n_splits)
        rms_test = np.zeros(n_splits)

        for i, (train, test) in enumerate(gkf.split(self._amat, self._bvec, groups=groups)):
            # Set displacement and force data
            X_train = self._amat[train, :]
            X_test = self._amat[test, :]
            y_train = self._bvec[train]
            y_test = self._bvec[test]

            reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)

            y_model_train = reg.predict(X_train)
            y_model_test = reg.predict(X_test)

            y_diff_train = y_model_train - y_train
            y_diff_test = y_model_test - y_test

            residual_train = np.dot(y_diff_train, y_diff_train)
            residual_test = np.dot(y_diff_test, y_diff_test)

            rms_train[i] = np.sqrt(residual_train / len(y_diff_train))
            rms_test[i] = np.sqrt(residual_test / len(y_diff_test))

        rmse_mean = np.mean(rms_train)
        rmse_std = np.std(rms_train, ddof=1)
        cv_mean = np.mean(rms_test)
        cv_std = np.std(rms_test, ddof=1)

        data = np.zeros(4, dtype=np.float64)
        data[0] = rmse_mean
        data[1] = rmse_std
        data[2] = cv_mean
        data[3] = cv_std

        data *= Ry * 1000.0 / Bohr

        return data

    def ols_find_best_cutoff(self, cutoff_list, cv=5,
                             save_history=True,
                             fname_history='ols_history.txt',
                             make_history_dataframe=False,
                             verbosity=0):
        """find the best cutoff from cutoff_list.

        The output file specified by fname_history contains, 
            # %d-fold cross-validation using ordinary least squares
            # cutoff (bohr), force RMSE (meV/A) [train mean, train std, test mean, test std]
            ...

        Args:
            cutoff_list (list): a list containing cutoff.
            cv (int, optional): the number of CV. Defaults to 5.
            save_history (bool, optional): save the history file if True. Defaults to True.
            fname_history (str, optional): file name of the history file. Defaults to 'ols_history.txt'.
            make_history_dataframe (str, optional): make the history dataframe if True. Defaults to False.
            verbosity (int, optional): verbosity level. Defaults to 0.

        Raises:
            RuntimeError: "cv must be a positive integer.

        Returns:
            tuple containing,
            list[float,float]: a list of [cutoff, cv_mean] from the lowest to the highest.
        """
        nbreak_increase_inarow = 2

        history = []
        min_test_mean = 10000000.0
        icount_increase = 0

        if cv <= 0:
            raise RuntimeError("cv must be a positive integer.")

        if save_history:
            f = open(fname_history, 'w')
            f.write('# %d-fold cross-validation using ordinary least squares\n' % cv)
            f.write('# cutoff (bohr), force RMSE (meV/A) [train mean, train std, test mean, test std]\n')
        if make_history_dataframe:
            columns = ["cutoff (bohr)", "force RMSE (meV/A) train_mean",
                       "force RMSE (meV/A) train_std",
                       "force RMSE (meV/A) test_mean",
                       "force RMSE (meV/A) test_std"]
            df_history = pd.DataFrame(columns=columns)

        for cutoff in cutoff_list:
            if verbosity > 0:
                print(" cutoff: %f " % cutoff, end='')
            cutoff_mat = np.ones((self._nspecies, self._nspecies)) * cutoff * 1.001
            if self._maxorder == 2:
                cutoff_3rd = np.ones((self._nspecies, self._nspecies)) * self._cutoff_3rd
                cutoff_mat = [cutoff_mat, cutoff_3rd]
            self.gen_sensing_matrix(maxorder=self._maxorder, cutoff=cutoff_mat)

            nrows, ncols = np.shape(self._amat)
            # break the loop if the sensing matrix becomes underdetermined
            if nrows * (cv - 1) / cv < ncols:
                break

            if verbosity > 0:
                print("Amat shape : ", np.shape(self._amat))

            data = self.run_ols_cross_validation(n_splits=cv)
            if save_history:
                f.write('%f %f %f %f %f\n' % (cutoff, data[0], data[1], data[2], data[3]))
                f.flush()
            if make_history_dataframe:
                newline = pd.Series([cutoff, data[0], data[1], data[2], data[3]], index=columns)
                df_history = df_history.append(newline,  ignore_index=True)

            history.append([cutoff, data[2]])

            if data[2] > min_test_mean:
                icount_increase += 1
            else:
                icount_increase = 0
                min_test_mean = data[2]

            if icount_increase == nbreak_increase_inarow:
                break

        if save_history:
            f.close()

        index_sorted = np.argsort(np.array(history)[:, 1])

        if make_history_dataframe:
            return history[index_sorted[0]][0], df_history
        else:
            return history[index_sorted[0]][0]

    def ols_optimize(self, rc_best, fname_fcs='supercell_best_rc.xml'):
        """optimization of the model?

        Args:
            rc_best (float): the best rc value.
            fname_fcs (str, optional): filename of the linear model. Defaults to 'supercell_best_rc.xml'.
        """
        nspecies = self._nspecies
        cutoff_mat = np.ones((nspecies, nspecies)) * rc_best * 1.001
        if self._maxorder == 2:
            cutoff_3rd = np.ones((nspecies, nspecies)) * self._cutoff_3rd
            cutoff_mat = [cutoff_mat, cutoff_3rd]

        with ALM(*self._crystal) as alm:
            alm.verbosity = 0
            alm.define(self._maxorder, cutoff_mat)
            alm.set_constraint(translation=True)
            alm.displacements = self._disp
            alm.forces = self._force
            X, y = alm.get_matrix_elements()
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            alm.set_fc(reg.coef_)
            alm.save_fc(filename=fname_fcs, format="alamode")

    def ols_optimize_and_get_fc(self, rc_best):
        """optimization of the model?

        Args:
            rc_best (float): the best rc value.
        """
        nspecies = self._nspecies
        cutoff_mat = np.ones((nspecies, nspecies)) * rc_best * 1.001
        if self._maxorder == 2:
            cutoff_3rd = np.ones((nspecies, nspecies)) * self._cutoff_3rd
            cutoff_mat = [cutoff_mat, cutoff_3rd]

        fc, indices = None, None
        with ALM(*self._crystal) as alm:
            alm.verbosity = 0
            alm.define(self._maxorder, cutoff_mat)
            alm.set_constraint(translation=True)
            alm.displacements = self._disp
            alm.forces = self._force
            X, y = alm.get_matrix_elements()
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            alm.set_fc(reg.coef_)
            # alm.save_fc(filename=fname_fcs, format="alamode")
            fc, indices = alm.get_fc(1, mode='all')
        return fc, indices


def run(args):
    fcmodel = ForceConstantModelSelector(fname_supercell=args.VASP, maxorder=1)

    fcmodel.load_training_data(num_snapshots=args.ntrain,
                               fname_dfset=args.dfset)

    cutoff_grid = np.arange(fcmodel.dist_min, fcmodel.dist_max + 0.1, 0.5)
    rc_best = fcmodel.ols_find_best_cutoff(cutoff_grid, cv=4, verbosity=1)
    fcmodel.ols_optimize(rc_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrain',
                        default=None,
                        type=int,
                        help="Number of training snapshots to use for extracting force constants")

    parser.add_argument('--VASP',
                        metavar='SPOSCAR',
                        type=str,
                        default='SPOSCAR',
                        help="VASP POSCAR file with equilibrium atomic positions")

    parser.add_argument('--dfset',
                        default='DFSET_random',
                        type=str,
                        help="DFSET filename to use for training")

    args = parser.parse_args()
    run(args)
