#lang info

#|review: ignore|#

(define collection "torch-native")
(define version "0.1")
(define deps '("base"))
(define pkg-desc "libtorchrkt and libtorch 2.14.0 (CPU) for torch on x86_64 Linux")
(define pkg-authors (list (string-append "bkschemer" "@" "gmail" "." "com")))
(define license
  '(((Apache-2.0 AND (BSD-3-Clause AND BSD-2-Clause))
     AND ((BSL-1.0 AND MIT) AND (Zlib AND (MIT OR Unlicense))))
    AND (((GPL-3.0-or-later WITH GCC-exception-3.1)
          AND (LGPL-2.1-or-later AND (LGPL-2.1-only AND LGPL-2.0-or-later)))
         AND LicenseRef-Intel-Simplified-Software-License)))
