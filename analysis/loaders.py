"""Load one archived run: its data.csv and its parameters block."""

from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import clean_path, parse_master_log
from models import MODELS, by_system
from snippets import get_param, get_vector_param

MASTER_LOG_FILENAME = "logfile.txt"


class plot_data:
    """Wraps <ID>.csv plus the logfile entry that identifies it.

    Replaces the three divergent copies of this class that lived in
    four_band.ipynb, prl_case.ipynb and test.ipynb.
    """

    def __init__(self, working_data_id, data_path, system=None):
        data_path = Path(clean_path(str(data_path)))
        self.id = str(working_data_id)
        self.data_path = data_path

        if system is None:
            system = _system_of(data_path)
        self.schema = by_system(system)

        master = parse_master_log(data_path / MASTER_LOG_FILENAME, system)
        try:
            self.parameters = master[self.id]
        except KeyError:
            raise KeyError(
                f"run {self.id!r} is not in {data_path / MASTER_LOG_FILENAME}") from None

        csv = data_path / f"{self.id}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"logfile has {self.id} but {csv} is missing")

        d = pd.read_csv(csv)
        self._data = d
        self.Time = d["Time"].values
        self.Ax = d["Ax"].values
        self.Ay = d["Ay"].values
        self.Jx = d["Jx"].values
        self.Jy = d["Jy"].values

        self.dt = self.Time[1] - self.Time[0]
        self.Time_dot = (self.Time[:-1] + self.Time[1:]) / 2.0
        self.Jx_dot = np.diff(self.Jx) / self.dt
        self.Jy_dot = np.diff(self.Jy) / self.dt

    def param(self, key, cast=float):
        return get_param(self.parameters, key, cast)

    def vector(self, key):
        return get_vector_param(self.parameters, key)

    def __repr__(self):
        return (f"<plot_data {self.schema.name} id={self.id} "
                f"nt={self.Time.size} dt={self.dt:.3g}>")


def _system_of(data_path):
    """Read the System line out of the logfile so the caller need not repeat it."""
    log = Path(data_path) / MASTER_LOG_FILENAME
    with open(log) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith(("*", "#")):
                return s
    raise ValueError(f"could not read a System line from {log}")
