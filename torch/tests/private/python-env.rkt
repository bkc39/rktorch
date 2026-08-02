#lang racket/base

;; Shared python-subprocess infrastructure for the PyTorch parity suites
;; (python-cross-test.rkt and generated-parity-test.rkt, split per #29):
;; locating python3, the CUDA LD_LIBRARY_PATH pinning, torch-availability
;; probes, the reference-program runners, and the shared float tolerance.
;;
;; Tolerance, not bit-exactness: v0 builds the C++ side against libtorch-bin,
;; which may differ in patch version from the Python torch here.  Seeded CPU
;; randn is stable across recent versions, but the tolerance absorbs any drift.
;; Flip the flake's `torchSource` to "python" for guaranteed bit-exact parity.

(require json
         racket/port
         racket/runtime-path
         racket/system)

(provide python
         call-with-python-env
         with-python-env
         python-torch-available?
         python-cuda-available?
         python-result
         python-check
         tol)

(define-runtime-path examples-dir "../../../examples")
(define-runtime-path python-checks-dir "../python")

(define python (find-executable-path "python3"))

;; Under the .#cuda shell the process LD_LIBRARY_PATH carries cudaTorch/lib
;; (libtorch 2.9, there for Racket's cuDNN), which would shadow the Python
;; torch-bin wheel's own 2.12 libtorch and break `import torch`
;; (libtorch_python.so ABI clash). The cudaHook exports
;; RKTORCH_CUDA_DRIVER_PATH (the host-driver farm only); pin each python
;; child's LD_LIBRARY_PATH to it so the wheel loads its own libs + the driver.
;; Unset (default/ci shell): leave the env alone so the CPU torch works.
(define cuda-driver-path (getenv "RKTORCH_CUDA_DRIVER_PATH"))

;; Run `thunk` with the python child's environment adjusted on a *copy* (never
;; the process-wide env): the CUDA driver-farm LD_LIBRARY_PATH pin when the
;; cuda shell set RKTORCH_CUDA_DRIVER_PATH, plus any `extra` (name . value)
;; string pairs (e.g. RKTORCH_PARITY_DEVICE). Nested uses compose — the inner
;; copy inherits the outer's parameterized vars.
(define (call-with-python-env thunk #:env [extra '()])
  (define env (environment-variables-copy (current-environment-variables)))
  (when cuda-driver-path
    (environment-variables-set! env #"LD_LIBRARY_PATH"
                                (string->bytes/utf-8 cuda-driver-path)))
  (for ([nv (in-list extra)])
    (environment-variables-set! env (string->bytes/utf-8 (car nv))
                                (string->bytes/utf-8 (cdr nv))))
  (parameterize ([current-environment-variables env]) (thunk)))

(define-syntax-rule (with-python-env body ...)
  (call-with-python-env (lambda () body ...)))

(define (python-torch-available?)
  (and python
       (with-python-env
        (parameterize ([current-output-port (open-output-nowhere)]
                       [current-error-port (open-output-nowhere)])
          (system* python "-c" "import torch")))))

;; Does the Python torch on PATH see a CUDA device? True only under the cuda
;; dev shell (cu130 torch-bin) on a GPU host; gates the accelerator parity.
(define (python-cuda-available?)
  (and python
       (with-python-env
        (parameterize ([current-output-port (open-output-nowhere)]
                       [current-error-port (open-output-nowhere)])
          (system* python "-c"
                   "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)")))))

;; Run a Python reference file (relative to examples/) and return its parsed
;; JSON hash.
(define (python-result rel-path)
  (define py (build-path examples-dir rel-path))
  (define out (open-output-string))
  (define ok?
    (with-python-env
     (parameterize ([current-output-port out]
                    [current-error-port (open-output-nowhere)])
       (system* python (path->string py)))))
  (unless ok?
    (error 'python-cross-test "python failed for ~a" rel-path))
  (read-json (open-input-string (get-output-string out))))

;; Run one of the standalone reference programs in torch/tests/python/
;; (each prints one JSON line) and parse its output. Captures stderr so a
;; crashing program surfaces its traceback rather than a bare "failed".
(define (python-check name)
  (define py (build-path python-checks-dir name))
  (define out (open-output-string))
  (define err (open-output-string))
  (define ok?
    (with-python-env
     (parameterize ([current-output-port out]
                    [current-error-port err])
       (system* python (path->string py)))))
  (unless ok?
    (error 'python-cross-test "python check ~a failed:\n~a"
           name (get-output-string err)))
  (read-json (open-input-string (get-output-string out))))

(define tol 1e-4)
