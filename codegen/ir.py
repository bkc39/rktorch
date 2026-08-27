"""Classify ATen schema entries into the v1 marshalling IR.

The IR admits exactly the signature shapes the hand-written v1 shim
marshals: Tensor, Scalar -> double, float -> double, int64_t (incl.
SymInt), bool, IntArrayRef -> (s64*, len), TensorList -> (ptr*, len),
and a single Tensor return. The tranche-2 (#3) additions widen this to
optional Tensor/int/IntArrayRef/ScalarType (marshalled as a NULL pointer
or a sentinel) and in-place ops (a mutable receiver + integer status,
the tr_tensor_sub_ shape). Everything else is skipped with a reason --
the generator reports skips instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchgen.model import (
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
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
# Optional kinds (tranche 2): a NULL pointer (tensor/int-array) or a
# sentinel (int64 -> has-value flag, dtype -> -1) marshals c10::nullopt.
OPTIONAL_TENSOR = "optional-tensor"
OPTIONAL_INT64 = "optional-int64"
OPTIONAL_INT_ARRAY = "optional-int-array"
OPTIONAL_DTYPE = "optional-dtype"


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
    # In-place ops mutate params[0] (the `self` receiver) and return an
    # integer status instead of a fresh handle; the C++ body calls the
    # `<base>_` method on the receiver rather than the at::* free function.
    inplace: bool = False
    # RNG ops draw from the global generator stream; the Racket emitter
    # gives them the no-retry allocator wrap so an OOM collect-and-retry
    # can never double-draw and break seeded parity (allowlist `rng` flag).
    rng: bool = False


@dataclass(frozen=True)
class Skip:
    aten_name: str
    reason: str
    shard: str


# Op names that collide with bindings the public facade shadow-dispatches
# (racket/base, racket/math tanh, racket/list argmax/flatten). Emitting
# these from generated.rkt under their bare names would hand consumers a
# tensor-only binding without the numeric fast path. Two tiers:
# - _RACKET_COLLISIONS: a hand-written tr_* implementation already
#   exists; codegen skips the op entirely.
# - _BASE_SHADOWED: fully generated, but the racket binding is emitted
#   as <name>-tensor so generated.rkt stays collision-free; the public
#   dispatch shim lives in promoted.rkt. Extend as collisions surface.
_RACKET_COLLISIONS = frozenset({
    "exp", "log", "sqrt", "tanh", "max", "min", "argmax", "flatten",
    "round", "floor", "ceiling", "truncate",
})

_BASE_SHADOWED = frozenset({"abs", "cos", "sin"})

_BASE_KINDS = {
    BaseTy.Tensor: TENSOR,
    BaseTy.Scalar: SCALAR,
    BaseTy.float: DOUBLE,
    BaseTy.int: INT64,
    BaseTy.SymInt: INT64,
    BaseTy.bool: BOOL,
}


def _int_list(ty) -> bool:
    return (
        isinstance(ty, ListType)
        and isinstance(ty.elem, BaseType)
        and ty.elem.name in (BaseTy.int, BaseTy.SymInt)
    )


def _param_kind(ty) -> str | None:
    if isinstance(ty, BaseType):
        return _BASE_KINDS.get(ty.name)
    if isinstance(ty, ListType) and isinstance(ty.elem, BaseType):
        if ty.elem.name in (BaseTy.int, BaseTy.SymInt):
            return INT_ARRAY
        if ty.elem.name is BaseTy.Tensor:
            return TENSOR_LIST
    if isinstance(ty, OptionalType):
        elem = ty.elem
        if isinstance(elem, BaseType):
            if elem.name is BaseTy.Tensor:
                return OPTIONAL_TENSOR
            if elem.name in (BaseTy.int, BaseTy.SymInt):
                return OPTIONAL_INT64
            if elem.name is BaseTy.ScalarType:
                return OPTIONAL_DTYPE
        if _int_list(elem):
            return OPTIONAL_INT_ARRAY
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
    inplace = func.name.name.inplace
    rets = func.returns
    if len(rets) != 1 or not (
        isinstance(rets[0].type, BaseType)
        and rets[0].type.name is BaseTy.Tensor
    ):
        return skip("return is not a single Tensor")

    # Functional ops are emitted as at::<base> free-function calls, so they
    # must expose the function variant. In-place ops are emitted as a method
    # call on the mutable receiver, so the method variant (which they all
    # have) suffices -- no torch.* free function is needed.
    if not inplace and Variant.function not in f.variants:
        return skip("method-only op: the C++ shim calls at::* free "
                    "functions and the parity battery calls torch.*")

    params: list[Param] = []
    for a in func.arguments.flat_non_out:
        kind = _param_kind(a.type)
        if kind is None:
            return skip(f"unsupported arg type: {a.name}: {a.type}")
        params.append(Param(a.name, kind))

    if inplace and not (params and params[0].kind == TENSOR):
        return skip("in-place op without a Tensor receiver as the "
                    "first argument")

    rkt_name = _racket_name(func.name)
    if func.name.name.base in _RACKET_COLLISIONS:
        return skip(f"name collides with a shadow-dispatched binding "
                    f"({rkt_name}); promote by hand with the dispatch shim")
    if func.name.name.base in _BASE_SHADOWED:
        rkt_name = f"{rkt_name}-tensor"

    base = func.name.name.base
    return Op(
        aten_name=aten_name,
        base=base,
        c_name=_c_name(aten_name),
        racket_name=rkt_name,
        # the full overload path, so the parity battery calls the explicit
        # torch.ops.aten.<base>.<overload> (e.g. sum.dim_IntList) instead of
        # relying on arg-count overload disambiguation. For an in-place op
        # this already carries the trailing-underscore base (add_.Tensor).
        python_name=aten_name,
        params=tuple(params),
        shard=shard,
        inplace=inplace,
    )
