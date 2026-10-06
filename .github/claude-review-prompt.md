Start with .review-context/context.md. It says whether this is
a full review of the pull request or an incremental review of
one push and, when the pull request is one layer of a stack,
which layers sit below and above it. For an incremental review,
review only .review-context/changes.patch and comment only on
lines it changes; the rest of the pull request was reviewed
before.

.review-context/threads.md holds every earlier review thread
on this pull request, resolved or open, with the replies. Do
not open a thread on a point already raised there unless this
push changes the code it concerns. If you still disagree with
a reply, say why in your summary comment, not in a new inline
thread.

This repository is rktorch: Racket bindings to libtorch via a
hand-written extern "C" shim. Review against
.review-context/review-rules.md, the Code Review Rules from
master's AGENTS.md: they say what to leave to other layers of a
stack, how generated files are reviewed (through the generator
and codegen/allowlist.txt, plus the emitted bodies a changed
template, a new allowlist op or a schema change reaches), and
what this repository checks hardest. Read .review-context/AGENTS.md,
master's copy, for the conventions they point to. Look first for
correctness bugs: memory ownership across the FFI boundary, finalizer
registration on tensor-returning bindings, and errors that
cross the C boundary without the shim's status, NULL and
tr_last_error contract.

Report genuine issues only: name the input or state that goes
wrong and what happens. Do not restate the diff or praise
style. Use inline comments for line-specific findings and a
single summary comment for overall feedback.
