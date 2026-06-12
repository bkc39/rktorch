"""Classify ATen schema entries into the v1 marshalling IR.

The IR admits exactly the signature shapes the hand-written v1 shim
marshals: Tensor, Scalar -> double, float -> double, int64_t (incl.
SymInt), bool, IntArrayRef -> (s64*, len), TensorList -> (ptr*, len),
and a single Tensor return. Everything else is skipped with a reason --
the generator reports skips instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchgen.model import (
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    Variant,
)

# Param kinds (the manifest spells these out for the parity battery).
TENSOR = "tensor"
SCALAR = "scalar"  # at::Scalar, marshalled as double
DOUBLE = "double"
INT64 = "int64"
BOOL = "bool"
INT_ARRAY = "int-array"
TENSOR_LIST = "tensor-list"


@dataclass(frozen=True)
class Param:
    name: str
    kind: str


@dataclass(frozen=True)
class Op:
    aten_name: str  # e.g. "sum.dim_IntList", "matmul"
    base: str  # e.g. "matmul" (the at::* callable name)
    c_name: str  # e.g. "tr_gen_sum_dim_intlist"
    racket_name: str  # e.g. "matmul", "sum-dim-intlist"
    python_name: str  # torch.<attr> for the parity battery
    params: tuple[Param, ...]
    shard: str


@dataclass(frozen=True)
class Skip:
    aten_name: str
    reason: str
    shard: str


# Op names that collide with bindings the public facade shadow-dispatches
# (racket/base, racket/math tanh, racket/list argmax/flatten). Emitting
# these from generated.rkt would hand consumers a tensor-only binding
# without the numeric fast path, so they skip — promote by hand with the
# dispatch shim instead. Extend as collisions surface.
_RACKET_COLLISIONS = frozenset({
    "exp", "log", "sqrt", "tanh", "max", "min", "argmax", "flatten",
    "abs", "round", "floor", "ceiling", "truncate",
})

_BASE_KINDS = {
    BaseTy.Tensor: TENSOR,
    BaseTy.Scalar: SCALAR,
    BaseTy.float: DOUBLE,
    BaseTy.int: INT64,
    BaseTy.SymInt: INT64,
    BaseTy.bool: BOOL,
}


def _param_kind(ty) -> str | None:
    if isinstance(ty, BaseType):
        return _BASE_KINDS.get(ty.name)
    if isinstance(ty, ListType) and isinstance(ty.elem, BaseType):
        if ty.elem.name in (BaseTy.int, BaseTy.SymInt):
            return INT_ARRAY
        if ty.elem.name is BaseTy.Tensor:
            return TENSOR_LIST
    return None


def _c_name(aten_name: str) -> str:
    return "tr_gen_" + aten_name.replace(".", "_").lower()


def _racket_name(func_name) -> str:
    base = func_name.name.base.replace("_", "-")
    overload = func_name.overload_name
    name = base
    if overload:
        name += "-" + overload.lower().replace("_", "-")
    if func_name.name.inplace:
        name += "!"
    return name


def classify(f: NativeFunction, shard: str) -> Op | Skip:
    func = f.func
    aten_name = str(func.name)

    def skip(reason: str) -> Skip:
        return Skip(aten_name, reason, shard)

    if func.arguments.out:
        return skip("out variant")
    if func.name.name.inplace:
        return skip("in-place op: the C-side mutation convention "
                    "(mutable handle + status, like tr_tensor_sub_) "
                    "is a tranche-2 design decision (#3)")
    rets = func.returns
    if len(rets) != 1 or not (
        isinstance(rets[0].type, BaseType)
        and rets[0].type.name is BaseTy.Tensor
    ):
        return skip("return is not a single Tensor")

    if Variant.function not in f.variants:
        return skip("method-only op: the C++ shim calls at::* free "
                    "functions and the parity battery calls torch.* — "
                    "revisit alongside the in-place convention (#3)")

    params: list[Param] = []
    for a in func.arguments.flat_non_out:
        kind = _param_kind(a.type)
        if kind is None:
            return skip(f"unsupported arg type: {a.name}: {a.type}")
        params.append(Param(a.name, kind))

    rkt_name = _racket_name(func.name)
    if func.name.name.base in _RACKET_COLLISIONS:
        return skip(f"name collides with a shadow-dispatched binding "
                    f"({rkt_name}); promote by hand with the dispatch shim")

    base = func.name.name.base
    return Op(
        aten_name=aten_name,
        base=base,
        c_name=_c_name(aten_name),
        racket_name=rkt_name,
        python_name=base,
        params=tuple(params),
        shard=shard,
    )
