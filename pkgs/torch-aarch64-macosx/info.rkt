#lang info

#|review: ignore|#

(define collection "torch-native")
(define version "0.1")
(define deps '("base"))
(define pkg-desc "libtorchrkt and libtorch 2.14.0 (CPU and MPS) for torch on macOS arm64")
(define pkg-authors (list (string-append "bkschemer" "@" "gmail" "." "com")))
(define license
  '(Apache-2.0 AND BSD-3-Clause AND BSD-2-Clause AND BSL-1.0 AND MIT
    AND (MIT OR Unlicense)
    AND (Apache-2.0 WITH LLVM-exception) AND (NCSA OR MIT)
    AND LGPL-2.1-or-later AND LGPL-2.1-only AND LGPL-2.0-or-later))
