import numpy as np
import os
import pandas as pd

_HISTORY_FILE = 'free_energy_history.txt'


class DfHistoryData:

    def __init__(self, df=None):
        if df is not None:
            self._df = df
            
    def make_columns(self, n: int = 1):
        _columns = ["Iteration", "No. training data", "best cuoff [bohr]", "lowest omega",
                    "ratio imaginary"]
        for i in range(n):
            _columns.append(f"Fvib (mev/atom) {i}")
        return _columns

    def from_file(self, historyfile: str = _HISTORY_FILE):
        self._historyfile = historyfile
        if os.path.isfile(historyfile):
            self._df = pd.read_csv(historyfile)
        else:
            _columns = self.make_columns()
            self._df = pd.DataFrame(columns=_columns)
        return self

    @property
    def df(self):
        return self._df

    @df.setter
    def data(self, df: pd.DataFrame):
        self._df = df

    def to_file(self, historyfile: str):
        self._df.to_csv(historyfile, index=False)

    def append(self, iter, ntrain, rc_best, omega_lowest, ratio_negative, fvib: np.ndarray):
        newdata = [iter, ntrain, rc_best, omega_lowest, ratio_negative]
        newdata.extend(fvib.tolist())
        nfvib = fvib.size
        newcolumns = self.make_columns(nfvib)
        if self._df.size == 0:
            self._df = pd.DataFrame([newdata], columns=newcolumns)
        else:
            if len(newcolumns) == self._df.shape[1]:
                series = pd.Series(newdata, index=self._df.columns)
                self._df = self._df.append(series, ignore_index=True)
            else:
                raise RuntimeError('df history size mismtach')
        return self._df
