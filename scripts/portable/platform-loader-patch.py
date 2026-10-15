"""Rewrite a copy of the torch collection for the platform-package layout.

The loader looks for libtorchrkt in collection "torch-native" first and falls
back to torch/native-libs; info.rkt gains the #:platform dependency.
    python3 -I platform-loader-patch.py <torch-collection-copy>
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1])


def replace(rel, old, new):
    path = root / rel
    text = path.read_text()
    if old not in text:
        sys.exit(f"{rel}: anchor not found: {old!r}")
    path.write_text(text.replace(old, new, 1))


replace(
    "foreign/raw/syntax.rkt",
    '(define-runtime-path native-libs-dir "../../native-libs")',
    """(define-runtime-path bundled-native-libs-dir "../../native-libs")
(define native-libs-dir
  (cond
    [(collection-file-path "libtorchrkt.so" "torch-native"
                           #:fail (lambda (_) #f))
     => (lambda (p) (let-values ([(dir _name _dir?) (split-path p)]) dir))]
    [else bundled-native-libs-dir]))""",
)
replace(
    "info.rkt",
    """(define deps '("base" "threading-lib"))""",
    """(define deps
  '("base" "threading-lib"
    ("torch-x86_64-linux" #:platform "x86_64-linux")))""",
)
replace(
    "info.rkt",
    '(define pre-install-collection "private/install-torchrkt-native.rkt")',
    "",
)
