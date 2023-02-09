import argparse
import os
from pathlib import Path

import numpy as np
from ase.units import Ry

from compute_phonon_props import PhononCalculator
from .model_selection import ForceConstantModelSelector
from .dfhistoryfile import DfHistoryData
from io import TextIOWrapper
import pandas as pd

ANPHON_BIN_DEFAULT = os.path.join(os.environ['HOME'], 'src', 'alamode', '_build', 'anphon', 'anphon')

_HISTORY_FILE = 'free_energy_history.txt'


def file_exists_or_hander(filename):
    if isinstance(filename, TextIOWrapper):
        return True
    elif os.path.exists(filename):
        return True
    elif isinstance(filename, np.ndarray):
        return True
    else:
        return False


def sampling_fvib(temperatures, fvib_dense, natoms):
    """extract data the temperatures of which are at 100,200,300,...

    The output, free energies are in meV/atom.

    Args:
        temperatures (np.ndarray): temperatures.
        fvib_dense (np.ndarray): free energies.
        natoms (int): number of atoms.

    Returns:
        tuple containing,
        np.ndarray: sampled temperatures.
        np.ndarray: free energies.
    """
    temperatures_sample = temperatures[np.where(np.mod(temperatures, 100.0) == 0)]
    data = fvib_dense[np.where(np.mod(temperatures, 100.0) == 0)]
    data[:] *= Ry * 1000 / float(natoms)
    return temperatures_sample, data


def new_status(df_history: pd.DataFrame, fvib_temp: np.ndarray):
    """update status from history_data

    The column corresponding to round(fvib_temp / 100) + 5 is used as fvib.

    Args:
        df_history (pd.DataFrame): history data assuming the shape id 2D.
        fvib_temp (np.ndarray): temperature data.

    Returns:
        dict: status dictionary.
    """
    dic_status = {}
    # history_data = np.loadtxt(fname_history, usecols=[0, 1, 2, 3, 4, round(fvib_temp / 100) + 5])
    history_data = df_history.values
    history_data = history_data[:, [0, 1, 2, 3, 4, round(fvib_temp / 100) + 5]]
    if len(history_data) > 0:
        # history_data = np.reshape(history_data, (len(np.ravel(history_data)) // 6, 6))
        # skip it because the shape of the history data is 2D now.
        valid_fe_data = history_data[history_data[:, 3] >= -1.0, :]
        iter_start = int(history_data[-1, 0])

        ntrain = int(history_data[-1, 1])
        ninc_used = ntrain // iter_start
        dic_status['exist_history'] = True
        dic_status['iter_start'] = iter_start
        dic_status['ninc_used'] = ninc_used
        dic_status['cutoffs'] = history_data[:, 2].tolist()
        dic_status['lowest_omegas'] = history_data[:, 3].tolist()
        dic_status['imaginary_ratios'] = history_data[:, 4].tolist()
        if len(valid_fe_data) > 0:
            fvib_prev = valid_fe_data[-1, 5]
            dic_status['previous_fvib'] = fvib_prev
    else:
        dic_status['exist_history'] = False

    return dic_status


def infer_status(df_history: pd.DataFrame, fvib_temp, cwd=None):
    """make status dict from files and their contents.

    If UNSTALE file exists, {'unstable':True}
    elseif supercell_converged.xml exists, {'convereged': True}
    elseif fname_history exists and contains appropriate data, a number of parameters will set set
    else {'exist_history': False}

    Args:
        df_history (pd.DataFrame): history data.

    Returns:
        dict: various status.
    """
    dic_status = {'converged': False,
                  'unstable': False,
                  'exist_history': True,
                  'previous_fvib': None,
                  'iter_start': 0,
                  'ninc_used': None,
                  'cutoffs': None,
                  'lowest_omegas': None,
                  'imaginary_ratios': None}
    """
    if file_exists_or_hander('UNSTABLE'):
        print("The system is likely to be unstable. Skip force constant calculations ...")
        dic_status['unstable'] = True

    elif file_exists_or_hander('supercell_converged.xml'):
        print("The free-energy already converged. Skip force constant calculations...")
        dic_status['converged'] = True

    el
    """
    if df_history.size > 0:

        dic_status.update(new_status(df_history, fvib_temp))

    else:
        dic_status['exist_history'] = False

    return dic_status


def run_main_(superstructure, dfset, df_history, primstructure, args,
              phase=[1, 2, 3], anphon_bin=None,
              fname_fcs='supercell_best_rc_iter.xml',  # for phase 2
              rc_best=None,
              temperatures=None, free_energy=None, omega_lowest=None, ratio_negative=None  # phase 3
              ):
    """automatic calculation of vibrational free energy and dispersion. 
    This function runs the anphon program. The path of anphon must be set before running this function at phase=2.

    If success, "Convergence criterion of |Delta_Fvib| < %.3f has been satisfied!" will be shown.


    args:
        qspacing: The spacing of q point sampling in units of A^{-1}.
        conv_thr: Convergence threshold of free energy in units of meV/atom.
        temp: "Temperature (K) at which the convergence of the free energy will be examined.
        cv: k-fold cross-validation will be performed for finding best cutoff radius.
            If not specified, the same value as --ninc will be used if --ninc > 1;
            otherwise, k = 2 will be used.
        ninc: The number of training data added at each iteration. Must be a positive integer.
            If not specified, the default value of 5 will be used.
        verbosity: 'Log level (0, 1, or 2).

    Args:
        superstructure (Atoms): supercell crystal structure.
        dfset (np.ndarray): DFSET of alamode.
        df_history (pd.DataFrame): history data.
        anphon_bin (str): anphon execution binary file path.
        primstructure (Atoms): primitive crystal structure.
        args (argparse): options.
        phase (list): calculation phase. Defaults to [1,2,3].
        anphon_bin (str): anphon binary path used at phase=2. Defaults to None.
        rc_best (float): the best value of rc used at phase=3. Defaults to None.
        temperatures (np.ndarray): temperatures used at phase=3. Defaults to None.
        free_energy (np.ndarray): free energies used at phase=3. Defaults to None.
        omega_lowest (float): the lowest value of omega used at phase=3. Defaults to None.
        ratio_negative (float): negative frequency ratio used at phase=3. Defaults to None.

    Raises:
        RuntimeError: --ninc must be a positive integer.
        RuntimeError: --cv must be larger than 1.
        RuntimeError: Stop the job because the --ninc parameter is inconsistent.


    Returns:
        If 1 in phase, tuple containig,
        float: rc_best
        pd.DataFrame: df_ols_history
        np.ndarray: fc2
        np.ndarray: fc_indices

        If 2 in phase, tuple containig,
        np.ndarray: temperatures
        np.ndarray: free_energy
        float: omega_lowest
        float: ratio_negative

        If 3 in phase, tuple containing,
        dict: status
        pd.DataFrame: df_history

    """
    fcmodel = ForceConstantModelSelector(fname_supercell=superstructure, maxorder=1)
    cutoff_grid = np.arange(fcmodel.dist_min, fcmodel.dist_max + 0.1, 0.5)
    fcmodel.load_training_data(fname_dfset=dfset)
    ndata_max = len(fcmodel.displacement)

    joblist = ['thermo', 'band']

    nstep = args["ninc"]
    if nstep < 1:
        raise RuntimeError("--ninc must be a positive integer.")

    ncv = 2
    if args["cv"] is not None:
        ncv = args["cv"]
    else:
        ncv = nstep

    if ncv < 2:
        raise RuntimeError("--cv must be larger than 1.")

    maxiter = ndata_max // nstep
    fvib_temp = args["temp"]
    ntrain = 0
    iter_start = 0
    thr_imaginary = -1.0
    nbreak_loop_imaginary = 3
    ncount_unstable_consecutive = 0
    cutoff_prev = 0.0

    status = infer_status(df_history, fvib_temp)
    # refresh_history_file = not status['exist_history']  # not used
    iter_start = status['iter_start']
    if status['previous_fvib'] is None:
        fvib_prev = 1.0e+20
    else:
        fvib_prev = status['previous_fvib']

    if status['ninc_used'] is not None:
        if status['ninc_used'] != nstep:
            raise RuntimeError('Stop the job because the --ninc parameter is inconsistent.')
        ntrain = nstep * iter_start

    if status['cutoffs'] is not None:
        cutoffs = np.array(status['cutoffs'])
        omegas = np.array(status['lowest_omegas'])
        cutoff_prev = cutoffs[0]
        if omegas[0] < thr_imaginary:
            ncount_unstable_consecutive = 1

        for i, (rc, omega0) in enumerate(zip(cutoffs, omegas)):
            if i == 0:
                continue

            if omega0 < thr_imaginary:
                if rc <= cutoff_prev:
                    ncount_unstable_consecutive += 1
                else:
                    ncount_unstable_consecutive = 1
                    cutoff_prev = rc
            else:
                ncount_unstable_consecutive = 0
                if rc > cutoff_prev:
                    cutoff_prev = rc
            if ncount_unstable_consecutive >= nbreak_loop_imaginary:
                status['unstable'] = True
                break

    if not status['converged']:

        # must be "iter_start < maxiter"
        maxiter = iter_start+1  # force for loop
        for iter in range(iter_start, maxiter):
            iter = iter_start
            ntrain = ndata_max  # force max.

            if 1 in phase:
                rc_best, df_ols_history = fcmodel.ols_find_best_cutoff(cutoff_grid,
                                                                       cv=ncv,
                                                                       save_history=False,
                                                                       # fname_history='ols_history_iter.txt',
                                                                       make_history_dataframe=True,
                                                                       verbosity=(args["verbosity"] > 1))
                # print("df_ols_history")
                # print(df_ols_history)
                # fcmodel.ols_optimize(rc_best, fname_fcs=fname_fcs)
                fc2, fc_indices = fcmodel.ols_optimize_and_get_fc(rc_best)
                return rc_best, df_ols_history, fc2, fc_indices

            if 2 in phase:
                from ..io import Fcsxml
                fcsxml = Fcsxml(superstructure.cell, superstructure.get_scaled_positions(),
                                superstructure.numbers)
                fcsxml.set_force_constants(fc2, fc_indices)
                fcsxml.write(fname_fcs)

                calculator = PhononCalculator(fname_primitive=primstructure,
                                              fname_fcs=fname_fcs,
                                              qspacing=args["qspacing"])
                calculator.set_anphon_bin(anphon_bin)
                calculator.prefix = 'iter'
                calculator.compute(joblist)
                omega_lowest = calculator.get_lowest_frequency()
                ratio_negative = calculator.get_imaginary_ratio(threshold=0.0)
                return calculator.temperatures, calculator.free_energy, \
                    omega_lowest, ratio_negative

            if 3 in phase:
                natom = len(superstructure.numbers)
                temp, fvib = sampling_fvib(temperatures,
                                           free_energy,
                                           natom)
                historyData = DfHistoryData(df_history)
                historyData.append(iter + 1, ntrain, rc_best, omega_lowest, ratio_negative, fvib)
                df_history = historyData.df

                if iter == 0:
                    cutoff_prev = rc_best
                    if omega_lowest < thr_imaginary:
                        ncount_unstable_consecutive = 1
                        if args["verbosity"] > 0:
                            print("Iter #%d: unstable mode exists, w = %.2f cm^-1" % ((iter + 1), omega_lowest))
                else:
                    if omega_lowest < thr_imaginary:
                        if rc_best <= cutoff_prev:
                            ncount_unstable_consecutive += 1
                            print("Iter #%d: unstable mode exists with the same cutoff radius %d-times "
                                  "in a row, w = %.2f cm^-1" % ((iter + 1),
                                                                ncount_unstable_consecutive,
                                                                omega_lowest))
                        else:
                            ncount_unstable_consecutive = 1
                            cutoff_prev = rc_best
                            print("Iter #%d: unstable mode exists but with a larger cutoff radius, w = %.2f cm^-1. "
                                  "Reset the counter to 1" % ((iter + 1),
                                                              omega_lowest))
                    else:
                        ncount_unstable_consecutive = 0
                        if rc_best > cutoff_prev:
                            cutoff_prev = rc_best

                if omega_lowest >= thr_imaginary:
                    fvib_now = fvib[temp == fvib_temp][0]

                    if args["verbosity"] > 0:
                        if iter == 0:
                            print("Iter #%d: Fvib (meV/atom) = %.3f" % ((iter + 1), fvib_now))
                        else:
                            print("Iter #%d: Fvib (meV/atom) = %.3f, "
                                  "Delta_Fvib (meV/atom) = %.3f" % ((iter + 1),
                                                                    fvib_now,
                                                                    (fvib_now - fvib_prev)))

                    if np.abs(fvib_now - fvib_prev) < args["conv_thr"]:
                        status['converged'] = True
                        if args["verbosity"] > 0:
                            print("Convergence criterion of |Delta_Fvib| < %.3f has been satisfied!" % args["conv_thr"])
                        break
                    else:
                        fvib_prev = fvib_now

                if ncount_unstable_consecutive >= nbreak_loop_imaginary:
                    status['unstable'] = True
                    if args["verbosity"] > 0:
                        print("Stop the iteration because the system is unlikely to become stable.")
                    break

                status.update(new_status(df_history, fvib_temp))
                break

    if status['unstable']:
        if "cwd" in status:
            os.makedirs(status["cwd"])
            Path(status["cwd"], 'UNSTABLE').touch()

    if False:
        if status['converged']:
            # shutil.copy2('supercell_best_rc_iter.xml', 'supercell_converged.xml')

            if "cwd" in args.keys():
                calculator = PhononCalculator(fname_primitive=primstructure,
                                              fname_fcs=fname_fcs,
                                              qspacing=args["qspacing"], cwd=args["cwd"])
                calculator.set_anphon_bin(anphon_bin)
                calculator.prefix = 'converged'
                joblist = ['dos+thermo', 'band']
                calculator.compute(joblist)
                calculator.make_figures(joblist)

    return status, df_history


def run_main(args):
    # make crystal structure
    poscar_file = args.VASP
    from ase import io
    superstructure = io.read(poscar_file, format="vasp")

    poscar_file = args.prim
    from ase import io
    primstructure = io.read(poscar_file, format="vasp")

    # make dfset
    dfset_file = args.dfset
    dfset = np.loadtxt(dfset_file)
    numatoms = len(superstructure.numbers)
    dfset = dfset.reshape(-1, numatoms, 6)

    # make history data
    import copy
    history_file = copy.deepcopy(args.history)
    history = DfHistoryData().from_file(history_file)

    # anphon binary
    if 'ANPHON_BIN' in os.environ:
        anphon_bin = os.environ['ANPHON_BIN']
    else:
        anphon_bin = ANPHON_BIN_DEFAULT

    # params
    params = {"qspacing": args.qspacing, "conv_thr": args.conv_thr, "temp": args.temp, "cv": args.cv,
              "ninc": args.ninc, "verbosity": args.verbosity}

    # figure folder
    params["cwd"] = os.getcwd()

    rc_best, temperatures, free_energy, \
        omega_lowest, ratio_negative = run_main_(superstructure, dfset, history.df, primstructure, params,
                                                 phase=1, anphon_bin=None)
    result, df_history = run_main_(superstructure, dfset, history.df, anphon_bin, primstructure, params,
                                   phase=2, anphon_bin=anphon_bin,
                                   rc_best=rc_best, temperatures=temperatures, free_energy=free_energy,
                                   omega_lowest=omega_lowest, ratio_negative=ratio_negative)

    history = DfHistoryData(df_history)
    history.to_file(history_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--VASP',
                        metavar='SPOSCAR',
                        type=str,
                        default='SPOSCAR',
                        help="VASP POSCAR file with equilibrium atomic positions")

    parser.add_argument('--history',
                        default=_HISTORY_FILE,
                        type=str,
                        help="history file.")

    parser.add_argument('--dfset',
                        default='DFSET_random',
                        type=str,
                        help="DFSET filename to use for training")

    parser.add_argument('--prim',
                        metavar='POSCAR',
                        type=str,
                        default='POSCAR',
                        help="VASP POSCAR file of the primitive cell with equilibrium atomic positions")

    parser.add_argument('--qspacing',
                        metavar='0.1',
                        type=float,
                        default=0.1,
                        help="The spacing of q point sampling in units of A^{-1}.")

    parser.add_argument('--conv_thr',
                        metavar='0.1',
                        type=float,
                        default=0.1,
                        help="Convergence threshold of free energy in units of meV/atom.")

    parser.add_argument('--temp',
                        metavar='1000',
                        type=float,
                        default=1000,
                        help="Temperature (K) at which the convergence of the free energy will be examined.")

    parser.add_argument('--cv',
                        metavar='k',
                        type=int,
                        default=None,
                        help="k-fold cross-validation will be performed for finding best cutoff radius.\n"
                             "If not specified, the same value as --ninc will be used if --ninc > 1;\n"
                             "otherwise, k = 2 will be used.")

    parser.add_argument('--ninc',
                        type=int,
                        default=5,
                        help="The number of training data added at each iteration. Must be a positive integer. \n"
                             "If not specified, the default value of 5 will be used.")

    parser.add_argument('--verbosity',
                        type=int,
                        default=0,
                        help='Log level (0, 1, or 2)')

    args = parser.parse_args()

    run_main(args)
