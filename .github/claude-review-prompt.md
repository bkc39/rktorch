Start with .review-context/context.md. It says whether this is
a full review of the pull request or an incremental review of
one push, and names the stack when the pull request is one
layer of one. For an incremental review, review only
.review-context/changes.patch and comment only on lines it
changes; the rest of the pull request was reviewed before. In a
stack, review only this layer's diff against its base branch.

.review-context/threads.md holds every earlier review thread
on this pull request, resolved or open, with the replies. Do
not open a thread on a point already raised there unless this
push changes the code it concerns. If you still disagree with
a reply, say why in your summary comment, not in a new inline
thread.

This repository is torchrkt: Racket bindings to libtorch via a
hand-written extern "C" shim. Read AGENTS.md first for the
project's conventions; its Code Review Rules section is written
for you. Then review the diff for:
- correctness bugs (memory ownership across the FFI boundary,
  the integer-status + tr_last_error contract, GC finalizer
  registration on tensor-returning bindings)
- violations of repo conventions (files <= 500 lines, ops.cpp
  boundary helpers in detail/op_call.hpp, allocator-wrapped raw
  bindings, contracts at the definition site, the
  name-shadowing dispatch convention)
- missing test or parity coverage for new ops (gtest goldens,
  example triples, generated-parity-test recipes for allowlist
  ops, python-cross-test checks for hand-written ops)

Generated files (everything under a generated/ directory,
cpp/include/torchrkt/c_api/generated.h, torch/generated.rkt and
torch/tests/generated-parity.rktd) carry a DO-NOT-EDIT header:
review the generator in codegen/ and codegen/allowlist.txt, the
diffs that produce them, and do not review the generated bodies.

Report genuine issues only: name the input or state that goes
wrong and what happens. Do not restate the diff or praise
style. Use inline comments for line-specific findings and a
single summary comment for overall feedback.
