import json
import os
from aiida.orm import load_node


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

    def load(self, label: str, force=None, raise_error=False):

        self.load_json()
        if label in self.dic:
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
        _t = self.load(label, node, raise_error=False)
        print("debug, load", _t)
        if _t is None or self.force:
            print("debug, force dump")
            self.dump(label, node)
            return node
        else:
            return _t
