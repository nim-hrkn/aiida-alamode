"""Compact provenance graph: only the processes (CalcJob / WorkChain / calcfunction) that lead to a node,
connected where an output of one is an input of another. usage: python provenance_processes.py <pk> <out.png|out.svg|out.pdf>"""
import os
import sys
from aiida import load_profile
load_profile()
from aiida.orm import load_node, ProcessNode, CalcJobNode, WorkChainNode
from aiida.common.links import LinkType
from aiida.tools.graph.graph_traversers import traverse_graph
import graphviz

target = int(sys.argv[1]); out = sys.argv[2]
res = traverse_graph([target], max_iterations=None,
                     links_backward=[LinkType.INPUT_CALC, LinkType.INPUT_WORK, LinkType.CREATE, LinkType.RETURN,
                                     LinkType.CALL_CALC, LinkType.CALL_WORK])
procs = {pk: load_node(pk) for pk in res["nodes"] if isinstance(load_node(pk), ProcessNode)}
g = graphviz.Digraph(format="png", graph_attr={"rankdir": "TB", "nodesep": "0.3", "ranksep": "0.5"})
for pk, n in procs.items():
    plugin = n.process_type.split(":")[-1].replace("aiida_alamode.", "")
    if isinstance(n, CalcJobNode):
        shape, color = "box", "#f28e8e"
    elif isinstance(n, WorkChainNode):
        shape, color = "box3d", "#f7b26b"
    else:
        shape, color = "ellipse", "#a8d8a8"; plugin = plugin.split(".")[-1]
    g.node(str(pk), f"{plugin}\npk {pk}", shape=shape, style="filled", fillcolor=color, fontsize="10")
edges = set()
for pk, n in procs.items():
    for link in n.base.links.get_incoming(link_type=(LinkType.INPUT_CALC, LinkType.INPUT_WORK)).all():
        creator = link.node.base.links.get_incoming(link_type=(LinkType.CREATE, LinkType.RETURN)).all()
        for c in creator:
            if c.node.pk in procs and c.node.pk != pk:
                edges.add((c.node.pk, pk, f"{link.node.__class__.__name__}\n{link.link_label}"))
    for link in n.base.links.get_incoming(link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)).all():
        if link.node.pk in procs:
            edges.add((link.node.pk, pk, "CALL"))
for a, b, lab in sorted(edges):
    g.edge(str(a), str(b), label=lab, fontsize="8", style="dashed" if lab == "CALL" else "solid")
base, ext = os.path.splitext(out)
g.render(base, format=(ext or ".png").lstrip("."), cleanup=True)   # .png / .svg / .pdf
print(f"{len(procs)} processes, {len(edges)} edges -> {out}")
