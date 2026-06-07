#lang racket/base

;; Live cross-validation of the Racket bindings against upstream PyTorch.
;;
;; For each reference example we run the adjacent Python file (examples/python/
;; NN-*.py, which prints {"shape": [...], "values": [...]}) and check our Racket
;; result agrees within a float tolerance.  The `nix develop` shell provides
;; Python `torch`; when python3 can't `import torch` (e.g. the sandboxed
;; `nix build`, or the lean `.#ci` shell) this test SKIPS -- keeping
;; `raco test` and `nix build` green without it.
;;
;; Tolerance, not bit-exactness: v0 builds the C++ side against libtorch-bin,
;; which may differ in patch version from the Python torch here.  Seeded CPU
;; randn is stable across recent versions, but the tolerance absorbs any drift.
;; Flip the flake's `torchSource` to "python" for guaranteed bit-exact parity.
;;
;; Run for real:  raco test torchrkt/tests/python-cross-test.rkt

(module+ test
  (require racket/port
           racket/runtime-path
           racket/system
           json
           rackunit
           "../main.rkt")

  (define-runtime-path examples-dir "../../examples")

  (define python (find-executable-path "python3"))

  (define (python-torch-available?)
    (and python
         (parameterize ([current-output-port (open-output-nowhere)]
                        [current-error-port (open-output-nowhere)])
           (system* python "-c" "import torch"))))

  ;; Run a Python reference file and return its parsed JSON hash.
  (define (python-result rel-path)
    (define py (build-path examples-dir rel-path))
    (define out (open-output-string))
    (define ok?
      (parameterize ([current-output-port out]
                     [current-error-port (open-output-nowhere)])
        (system* python (path->string py))))
    (unless ok?
      (error 'python-cross-test "python failed for ~a" rel-path))
    (read-json (open-input-string (get-output-string out))))

  (define tol 1e-4)

  (cond
    [(not (python-torch-available?))
     (printf "[python-cross-test] skipped: python3 `torch` not available ~a\n"
             "(run inside `nix develop`)")]
    [else
     ;; 00 — seeded randn 2x2
     (let ()
       (define j (python-result "python/00_randn.py"))
       (define py-shape (hash-ref j 'shape))
       (define py-values (hash-ref j 'values))
       (manual-seed! 0)
       (define t (randn 2 2))
       (check-equal? (tensor-shape t) py-shape "00_randn: shape parity")
       (define rkt-values (tensor->list t))
       (check-equal? (length rkt-values) (length py-values)
                     "00_randn: value count")
       (for ([r (in-list rkt-values)]
             [p (in-list py-values)]
             [i (in-naturals)])
         (check-= r p tol (format "00_randn: value ~a parity" i)))
       ;; The headline: the Racket REPL repr string matches the Python one
       ;; byte-for-byte (racket repl == python repl).
       (check-equal? (tensor->repr t) (hash-ref j 'repr)
                     "00_randn: repr parity"))]))
