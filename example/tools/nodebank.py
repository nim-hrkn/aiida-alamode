import json
import os
from aiida.orm import load_node

from .aiida_support import wait_for_node_finished
from aiida.engine import submit

from datetime import datetime


class NodeBank:
    """実行するごとにノードをstoreせずに過去のノードをloadするために用いる。

    """
    _KEY = "pk"

    def __init__(self, parent_dir, force=False):
        """initialization.

        if force=True, force calling dump().

        """
        self.filename = ".node.json"
        self.parent_dir = parent_dir
        self.force = force
        self.filepath = os.path.join(self.parent_dir, self.filename)

    def set_force(self, force):
        self.force = force

    def load_json(self):
        filepath = self.filepath
        self.dic = {}
        if os.path.isfile(filepath):
            with open(filepath) as f:
                self.dic = json.load(f)

    def dump_json(self):
        filepath = self.filepath
        with open(filepath, "w") as f:
            json.dump(self.dic, f)

    def load(self, label: str, raise_error: bool = True):

        self.load_json()
        if label in list(self.dic.keys()):
            if self._KEY == "pk":
                pk = self.dic[label]
                return load_node(pk)
            else:
                uuid = self.dic[label]
                return load_node(uuid)
        else:
            if raise_error:
                raise ValueError(f"failed to find key={label}")
            return None

    def dump(self, label: str, node):

        node.store()
        self.load_json()
        if self._KEY == "pk":
            self.dic[label] = node.pk
            self.dump_json()
        else:
            self.dic[label] = node.uuid
            self.dump_json()
        return node

    def load_or_dump(self, label: str, node):
        print("debug, node", node)
        _t = self.load(label, raise_error=False)
        print("debug, load", _t)
        if _t is None or self.force:
            print("debug, force dump")
            self.dump(label, node)
            return node
        else:
            return _t

    def load_code_or_wait_for_node(self, key,  builder, sec=2):
        starttime = datetime.now()

        g_alm_suggest_future = self.load(key, raise_error=False)
        print(g_alm_suggest_future)
        if g_alm_suggest_future is None:
            # if True:
            g_alm_suggest_future = submit(builder)  # temporary run(), use submit()
            print(g_alm_suggest_future)
            wait_for_node_finished(g_alm_suggest_future, sec)
            if g_alm_suggest_future.is_finished_ok:
                self.dump(key, g_alm_suggest_future)
            else:
                raise
        endtime = datetime.now()
        print(endtime-starttime)
        return g_alm_suggest_future

    def load_workchain_or_wait_for_node(self, key, workchain, inputs, sec=2):
        starttime = datetime.now()

        lammps_all = self.load(key, raise_error=False)
        if lammps_all is None:
            lammps_all = submit(workchain, **inputs)
            wait_for_node_finished(lammps_all, sec)
            print(lammps_all.is_finished_ok)
            if lammps_all.is_finished_ok:
                self.dump(key, lammps_all)
            else:
                raise
        endtime = datetime.now()
        print(endtime-starttime)
        return lammps_all
