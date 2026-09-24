"""
Equivar: Born effective charges with the pretrained equivariant GCNN of Kutana, Shimizu, Watanabe & Asahi,
Sci. Rep. 15, 16687 (2025) (github.com/equivar/equivar_eval; weights BM1.pt / BM2.pt on Mendeley Data
10.17632/hx8kcpxh84.1).

The weights are TorchScript archives that only need torch (the graph is built here with e3nn's spherical
harmonics, as in equivar_eval.process.AtomsToGraphs).  The archives reference the custom operators of
torch_scatter / torch_sparse, which have no wheels for recent torch; pure-torch stand-ins are registered
under the same names before the archive is loaded, so neither package is needed.

Training data (DFPT): ABO3 perovskites (A = Ba, Ca, Sr, Pb; B = Ti, Zr, Hf), Li3PO4 and ZrO2, so the
elements the model has learned are Ba Ca Hf Li O P Pb Sr Ti Zr.  Other elements go through an untrained
embedding row and give meaningless numbers.  Only Z* is predicted (no dielectric tensor).

    from aiida_alamode.equivar import EquivarCalculator
    atoms.calc = EquivarCalculator(model="~/models/equivar/BM1.pt")
    atoms.calc.calculate(atoms, properties=["born_effective_charges"])
    atoms.calc.results["born_effective_charges"]     # nat x 3 x 3, e

Registered in aiida_alamode.ase_runner.CALCULATORS as "equivar" (runner mode "bec", CalcJob alamode.bec_ase).
"""
import math
import os

import numpy as np

KNOWN_ELEMENTS = ("Ba", "Ca", "Hf", "Li", "O", "P", "Pb", "Sr", "Ti", "Zr")
DEFAULT_MODEL = os.environ.get("EQUIVAR_MODEL", os.path.expanduser("~/models/equivar/BM1.pt"))

# the hyper-parameters of the graph the BM1 / BM2 archives were trained with (equivar_eval defaults)
GRAPH_MAX_RADIUS = 3.0     # A
NUM_RADIAL = 32
EDGE_SH_LMAX = 2

_SQ2_1 = 1.0 / math.sqrt(2.0)
_SQ3_1 = 1.0 / math.sqrt(3.0)
_SQ23_1 = _SQ2_1 * _SQ3_1
# model output (irreducible components) -> Cartesian Z_11, Z_12, ..., Z_33 (equivar_eval.scripts.evaluate)
_COB = np.array([
    [_SQ3_1, 0.0, 0.0, -_SQ23_1, 0.0, -_SQ2_1, 0.0, 0.0, 0.0],
    [0.0, 0.0, _SQ2_1, 0.0, 0.0, 0.0, 0.0, 0.0, _SQ2_1],
    [0.0, _SQ2_1, 0.0, 0.0, 0.0, 0.0, 0.0, -_SQ2_1, 0.0],
    [0.0, 0.0, _SQ2_1, 0.0, 0.0, 0.0, 0.0, 0.0, -_SQ2_1],
    [_SQ3_1, 0.0, 0.0, 2.0 * _SQ23_1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, _SQ2_1, 0.0, _SQ2_1, 0.0, 0.0],
    [0.0, _SQ2_1, 0.0, 0.0, 0.0, 0.0, 0.0, _SQ2_1, 0.0],
    [0.0, 0.0, 0.0, 0.0, _SQ2_1, 0.0, -_SQ2_1, 0.0, 0.0],
    [_SQ3_1, 0.0, 0.0, -_SQ23_1, 0.0, _SQ2_1, 0.0, 0.0, 0.0],
]).T


_ops_registered = False


def register_scatter_ops():
    """torch_scatter / torch_sparse operators referenced by the TorchScript archives, in plain torch."""
    global _ops_registered
    if _ops_registered:
        return
    import torch
    if hasattr(torch.ops, "torch_scatter") and hasattr(torch.ops.torch_scatter, "segment_sum_csr"):
        try:
            torch.ops.torch_scatter.segment_sum_csr   # the real package is installed
            _ops_registered = True
            return
        except (AttributeError, RuntimeError):
            pass

    def _index_of_csr(indptr, n_src):
        counts = indptr[1:] - indptr[:-1]
        return torch.repeat_interleave(torch.arange(counts.numel(), device=indptr.device), counts)

    def _segment(src, indptr, reduce):
        index = _index_of_csr(indptr, src.shape[0])
        n = indptr.numel() - 1
        if reduce == "sum":
            out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
            return out.index_add_(0, index, src)
        if reduce == "mean":
            out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
            out.index_add_(0, index, src)
            counts = (indptr[1:] - indptr[:-1]).clamp(min=1).to(src.dtype)
            return out / counts.reshape((-1,) + (1,) * (src.dim() - 1))
        out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        out, arg = _scatter_minmax(src, index, 0, out, n, reduce)
        return out, arg

    def _scatter_minmax(src, index, dim, out, dim_size, reduce):
        if dim < 0:
            dim += src.dim()
        size = list(src.shape)
        size[dim] = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
        if index.dim() == 1:
            shape = [1] * src.dim(); shape[dim] = -1
            index = index.reshape(shape).expand_as(src)
        fill = torch.finfo(src.dtype).min if reduce == "max" else torch.finfo(src.dtype).max
        res = torch.full(size, fill, dtype=src.dtype, device=src.device)
        res = res.scatter_reduce(dim, index, src, reduce="amax" if reduce == "max" else "amin", include_self=True)
        res = torch.where(res == fill, torch.zeros_like(res), res)
        arg = torch.full(size, src.shape[dim], dtype=torch.long, device=src.device)
        return res, arg

    lib = torch.library.Library("torch_scatter", "DEF")
    lib.define("segment_sum_csr(Tensor src, Tensor indptr, Tensor? out=None) -> Tensor")
    lib.define("segment_mean_csr(Tensor src, Tensor indptr, Tensor? out=None) -> Tensor")
    lib.define("segment_min_csr(Tensor src, Tensor indptr, Tensor? out=None) -> (Tensor, Tensor)")
    lib.define("segment_max_csr(Tensor src, Tensor indptr, Tensor? out=None) -> (Tensor, Tensor)")
    lib.define("scatter_mul(Tensor src, Tensor index, int dim, Tensor? out, int? dim_size) -> Tensor")
    lib.define("scatter_min(Tensor src, Tensor index, int dim, Tensor? out, int? dim_size) -> (Tensor, Tensor)")
    lib.define("scatter_max(Tensor src, Tensor index, int dim, Tensor? out, int? dim_size) -> (Tensor, Tensor)")
    lib.impl("segment_sum_csr", lambda src, indptr, out=None: _segment(src, indptr, "sum"), "CompositeExplicitAutograd")
    lib.impl("segment_mean_csr", lambda src, indptr, out=None: _segment(src, indptr, "mean"), "CompositeExplicitAutograd")
    lib.impl("segment_min_csr", lambda src, indptr, out=None: _segment(src, indptr, "min"), "CompositeExplicitAutograd")
    lib.impl("segment_max_csr", lambda src, indptr, out=None: _segment(src, indptr, "max"), "CompositeExplicitAutograd")

    def _scatter_mul(src, index, dim, out, dim_size):
        if dim < 0:
            dim += src.dim()
        size = list(src.shape)
        size[dim] = dim_size if dim_size is not None else int(index.max()) + 1
        if index.dim() == 1:
            shape = [1] * src.dim(); shape[dim] = -1
            index = index.reshape(shape).expand_as(src)
        res = torch.ones(size, dtype=src.dtype, device=src.device) if out is None else out
        return res.scatter_reduce(dim, index, src, reduce="prod", include_self=True)
    lib.impl("scatter_mul", _scatter_mul, "CompositeExplicitAutograd")
    lib.impl("scatter_min", lambda src, index, dim, out, dim_size: _scatter_minmax(src, index, dim, out, dim_size, "min"),
             "CompositeExplicitAutograd")
    lib.impl("scatter_max", lambda src, index, dim, out, dim_size: _scatter_minmax(src, index, dim, out, dim_size, "max"),
             "CompositeExplicitAutograd")

    lib_sparse = torch.library.Library("torch_sparse", "DEF")
    lib_sparse.define("ind2ptr(Tensor ind, int M) -> Tensor")
    lib_sparse.define("ptr2ind(Tensor ptr, int E) -> Tensor")
    lib_sparse.impl("ind2ptr", lambda ind, M: torch.cumsum(torch.bincount(ind, minlength=M), 0).new_zeros(M + 1)
                    .index_copy_(0, torch.arange(1, M + 1, device=ind.device),
                                 torch.cumsum(torch.bincount(ind, minlength=M), 0)), "CompositeExplicitAutograd")
    lib_sparse.impl("ptr2ind", lambda ptr, E: torch.repeat_interleave(torch.arange(ptr.numel() - 1, device=ptr.device),
                                                                      ptr[1:] - ptr[:-1]), "CompositeExplicitAutograd")
    globals()["_lib_scatter"], globals()["_lib_sparse"] = lib, lib_sparse    # keep the registrations alive
    _ops_registered = True


def _neighbors(atoms, cutoff, mic):
    """(i, j, D) of the pairs with 0.1 < |D| < cutoff, D = r_j - r_i (+ lattice shift): every periodic image
    (mic=False, ase.neighborlist), or only the nearest image as equivar_eval does (mic=True)."""
    if mic:
        vec = atoms.get_all_distances(mic=True, vector=True)
        r = np.linalg.norm(vec, axis=2)
        i, j = np.where((r < cutoff) & (r > 0.1))
        return i, j, vec[i, j]
    from ase.neighborlist import primitive_neighbor_list
    i, j, D = primitive_neighbor_list("ijD", atoms.pbc, atoms.cell, atoms.positions, cutoff, self_interaction=True)
    r = np.linalg.norm(D, axis=1)
    keep = r > 0.1
    return i[keep], j[keep], D[keep]


def build_graph(atoms, cutoff=GRAPH_MAX_RADIUS, num_radial=NUM_RADIAL, lmax=EDGE_SH_LMAX, mic=False, device="cpu"):
    """{"Z", "edge_index", "edge_attr"} of one structure, as equivar_eval.process.AtomsToGraphs builds it."""
    import torch
    from e3nn import o3
    i, j, D = _neighbors(atoms, cutoff, mic)
    r = np.linalg.norm(D, axis=1)
    offset = torch.linspace(0.0, cutoff, num_radial)
    gamma = -0.5 / (offset[1] - offset[0]).item() ** 2
    radial = torch.exp(gamma * (torch.tensor(r, dtype=torch.float).view(-1, 1) - offset.view(1, -1)) ** 2)
    sh = o3.SphericalHarmonics(o3.Irreps.spherical_harmonics(lmax), normalize=True)
    angular = sh(torch.tensor(D / r[:, None], dtype=torch.float))
    combined = torch.einsum("bi,bj->bij", radial, angular)
    edge_attr = torch.cat([torch.flatten(combined[:, :, sl], start_dim=-2)
                           for sl in o3.Irreps.spherical_harmonics(lmax).slices()], dim=-1)
    return {"Z": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device),
            "edge_index": torch.tensor(np.array([j, i], dtype=np.int64), device=device),
            "edge_attr": edge_attr.to(device)}


class EquivarModel:
    """A loaded BM1 / BM2 archive.  predict(atoms) -> nat x 3 x 3 Z* (acoustic sum rule enforced)."""

    def __init__(self, model=DEFAULT_MODEL, device="cpu", mic=False, enforce_asr=True):
        import torch
        register_scatter_ops()
        self.path = os.path.expanduser(model)
        self.device = device
        self.mic = mic
        self.enforce_asr = enforce_asr
        self.model = torch.jit.load(self.path, map_location=torch.device(device)).eval()

    def predict(self, atoms):
        import torch
        unknown = sorted(set(atoms.get_chemical_symbols()) - set(KNOWN_ELEMENTS))
        if unknown:
            print(f"WARNING: Equivar has not been trained on {' '.join(unknown)}; the Born charges are not meaningful",
                  flush=True)
        data = build_graph(atoms, mic=self.mic, device=self.device)
        with torch.no_grad():
            out = self.model(data).cpu().double().numpy()
        bec = out @ _COB
        if self.enforce_asr:
            bec = bec - bec.mean(axis=0)
        return bec.reshape(len(atoms), 3, 3)


def EquivarCalculator(model=DEFAULT_MODEL, device="cpu", mic=False, enforce_asr=True, **kwargs):
    """ASE calculator with the property "born_effective_charges" (nat x 3 x 3, e; the training data's
    convention, rows as VASP BORN EFFECTIVE CHARGES).  No energy or forces."""
    from ase.calculators.calculator import Calculator, all_changes

    engine = EquivarModel(model, device=device, mic=mic, enforce_asr=enforce_asr)

    class _EquivarCalculator(Calculator):
        implemented_properties = ["born_effective_charges"]

        def __init__(self, **kw):
            super().__init__(**kw)
            self.engine = engine

        def calculate(self, atoms=None, properties=("born_effective_charges",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["born_effective_charges"] = self.engine.predict(self.atoms)

    return _EquivarCalculator(**kwargs)
